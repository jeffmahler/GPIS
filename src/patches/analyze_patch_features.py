import numpy as np
import argparse
import yaml
import os
import logging
import matplotlib
matplotlib.use('Agg')#prevents using X server backend for matplotlib
import matplotlib.pyplot as plt

import sys
_grasp_selection_path = os.path.join(os.path.dirname(__file__), '..', 'grasp_selection')
_data_analysis_path = os.path.join(os.path.dirname(__file__), '..', 'data_analysis')
sys.path.append(_grasp_selection_path)
sys.path.append(_data_analysis_path)
import plotting
from csv_statistics import CSVStatistics
import wrap_text

from patches_data_loader import PatchesDataLoader as PDL

import IPython

def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def _plot_save_hist_bin(config, data, labels, feature_name, label_name, output_path, hs):
    if len(np.unique(data)) != 2:
        logging.warn("Plotting histograms for binary data can only take in data with values 1 and 0. Skipping {0}".format(feature_name))
        return
    
    #read plot config
    font_size = config['plotting']['font_size']
    num_bins = config['plotting']['num_bins']
    dpi = config['plotting']['dpi']
    
    #get data and compute statistics
    positive_metrics = np.take(labels, np.argwhere(data == 1).flatten())
    negative_metrics = np.take(labels, np.argwhere(data == 0).flatten()) 
    
    #plotting
    textbox_props = {'boxstyle':'square', 'facecolor':'white'}
    
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Metric {0} Density'.format(label_name), fontsize=font_size)
    
    ax = plt.subplot("121")
    ax.set_title("{0}=1".format(feature_name), fontsize=font_size)
    plt.ylabel('Normalized Density', fontsize=font_size)
    plt.xlabel(wrap_text.wrap(label_name), fontsize=font_size)
    plotting.plot_histogram(positive_metrics, num_bins=num_bins, normalize=True, show_stats=True)
    
    ax = plt.subplot("122")
    ax.set_title("{0}=0".format(feature_name), fontsize=font_size)
    plt.ylabel('Normalized Density', fontsize=font_size)
    plt.xlabel(wrap_text.wrap(label_name), fontsize=font_size)
    plotting.plot_histogram(negative_metrics, num_bins=num_bins, normalize=True, show_stats=True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    
    name = 'metric_{0}{1}'.format(label_name, feature_name)
    figname = '{0}histogram.pdf'.format(name)
    logging.info("Saving {0}".format(figname))
    plt.savefig(os.path.join(output_path, figname), dpi=dpi)
    plt.close()
    
    #save stats in csv
    pos_name = '{0}=1'.format(name)
    neg_name = '{0}=0'.format(name)
    
    hs.append_data(pos_name, positive_metrics)
    hs.append_data(neg_name, negative_metrics)

def _plot_save_scatter_pair(config, data, labels, features_pair, label_name, output_path, ss):
    if data.shape[1] != 2:
        logging.warn("Scatter plot pair can only accept two-dimensional data. Skipping {0}".format(features_pair))
        return
        
    scatter_subsamples = config['plotting']['scatter_subsamples']
    line_width = config['plotting']['line_width']
    eps = config['plotting']['eps']
    font_size = config['plotting']['font_size']
    dpi = config['plotting']['dpi']
        
    #subsample data if needed
    p1, p2 = data[:,0], data[:,1]
    sub_inds = np.arange(data.shape[0])
    if data.shape[0] > scatter_subsamples:
        sub_inds = np.choose(scatter_subsamples, np.arange(data.shape[0]))
    p1_sub = p1[sub_inds]
    p2_sub = p2[sub_inds]
    labels_sub = labels[sub_inds]
        
    #compute best fit line
    A = np.c_[data, np.ones(data.shape[0])]
    b = labels
    w, _, _, _ = np.linalg.lstsq(A, b)
    to_corr = np.c_[data, labels]
    rho = np.corrcoef(to_corr, rowvar=0) 
        
    min_p1, max_p1 = np.min(p1), np.max(p1)
    min_p2, max_p2 = np.min(p2), np.max(p2)
    x_vals = [min_p1, max_p1]
    y_vals = [w[2] + w[0] * min_p1 + w[1] * min_p2, w[2] + w[0] * max_p1 + w[1] * max_p2]
    
    #plot
    fig = plt.figure()
    plt.scatter(p1_sub, labels_sub, c='b', s=50)
    plt.scatter(p2_sub, labels_sub, c='g', s=50)
    plt.plot(x_vals, y_vals, c='r', linewidth=line_width)
    plt.xlim(min(np.percentile(p1_sub, 1), np.percentile(p2_sub, 1)), max(np.percentile(p1_sub, 99), np.percentile(p2_sub, 99)))
    plt.ylim(min(labels_sub) - eps, max(labels_sub) + eps)
    plt.title('Metric {0}\nvs {1} and {2}'.format(wrap_text.wrap(label_name), features_pair[0], features_pair[1]), fontsize=font_size)
    plt.xlabel("{0} {1}".format(features_pair[0], features_pair[1]), fontsize=font_size)
    plt.ylabel(wrap_text.wrap(label_name), fontsize=font_size)
    leg = plt.legend(('Best Fit Line (rho={:.3g})'.format(rho[2,0]), features_pair[0], features_pair[1]), loc='best', fontsize=12)
    leg.get_frame().set_alpha(0.7)
    plt.tight_layout()
    
    name = 'metric_{0}{1}{2}'.format(label_name, features_pair[0], features_pair[1])
    figname = '{0}scatter.pdf'.format(name)
    logging.info("Saving {0}".format(figname))
    plt.savefig(os.path.join(output_path, figname), dpi=dpi)
    plt.close()
    
    name1  = 'metric_{0}{1}'.format(label_name, features_pair[0])
    name2  = 'metric_{0}{1}'.format(label_name, features_pair[1])
    ss.append_data(name1, to_corr[:,[0,2]])
    ss.append_data(name2, to_corr[:,[1,2]])
    
def _plot_save_scatter(config, feature_data, split_by, all_labels, feature_name, label_name, output_path, ss):
    if feature_data.reshape(feature_data.shape[0], -1).shape[1] != 1:
        logging.warn("Scatter plot can only accept one-dimensional data. Skipping {0}".format(feature_name))
        return
        
    scatter_subsamples = config['plotting']['scatter_subsamples']
    line_width = config['plotting']['line_width']
    eps = config['plotting']['eps']
    font_size = config['plotting']['font_size']
    dpi = config['plotting']['dpi']
    
    splitted_data = {} #maps split name to splitted data
    for split_by_name, split_by_data in split_by.items():
        if split_by_data is None:
            splitted_data['']  = (feature_data, all_labels)
        else:
            split_data_ind_0 = np.where(split_by_data == 0)[0]
            split_data_ind_1 = np.where(split_by_data == 1)[0]
            
            splitted_data['{0}1_'.format(split_by_name)] = (feature_data[split_data_ind_1], all_labels[split_data_ind_1])
            splitted_data['{0}0_'.format(split_by_name)] = (feature_data[split_data_ind_0], all_labels[split_data_ind_0])

    for split_name, data_labels in splitted_data.items():
        data, labels = data_labels[0], data_labels[1]
        
        #subsample data if needed
        sub_inds = np.arange(data.shape[0])
        if data.shape[0] > scatter_subsamples:
            sub_inds = np.choose(scatter_subsamples, np.arange(data.shape[0]))
        data_sub = data[sub_inds]
        labels_sub = labels[sub_inds]

        #compute best fit line
        A = np.c_[data, np.ones(data.shape[0])] 
        b = labels
        
        w, _, _, _ = np.linalg.lstsq(A, b)
        to_corr = np.c_[data, labels]
        rho = np.corrcoef(to_corr, rowvar=0)[1,0]
            
        #scatter the PFC vs friction cone angle
        min_alpha = np.min(data)
        max_alpha = np.max(data)
        x_vals = [min_alpha, max_alpha]
        y_vals = [w[1] + w[0] * min_alpha, w[1] + w[0] * max_alpha]

        full_feature_name = feature_name
        if split_name != '':
            full_feature_name = '{0}split_{1}'.format(feature_name, split_name)
        
        #plot
        fig = plt.figure()
        plt.scatter(data_sub, labels_sub, c='b', s=50)
        plt.plot(x_vals, y_vals, c='r', linewidth=line_width)
        plt.xlim(x_vals[0]  - eps, x_vals[1] + eps)
        plt.ylim(min(labels) - eps, max(labels) + eps)
        plt.title('Metric {0}\n{1}'.format(wrap_text.wrap(label_name), full_feature_name), fontsize=font_size)
        plt.xlabel(full_feature_name, fontsize=font_size)
        plt.ylabel(wrap_text.wrap(label_name), fontsize=font_size)
        plt.legend(('Best Fit Line (rho={:.2g})'.format(rho), 'Datapoints'), loc='best')
        plt.tight_layout()

        name = 'metric_{0}{1}'.format(label_name, full_feature_name)
        figname = '{0}scatter.pdf'.format(name)
        logging.info("Saving {0}".format(figname))
        plt.savefig(os.path.join(output_path, figname), dpi=dpi)
        plt.close()

        ss.append_data(name, to_corr)
    
def analyze_patch_features(config, input_path, output_path):
    #load data
    features_set_hist_bin = PDL.get_include_set_from_dict(config['feature_prefixes_hist_bin'])
    
    features_set_scatter_pair = []
    pair_configs = config['feature_prefixes_scatter_pair']
    for pair_config in pair_configs:
        if pair_config['use']:
            features_set_scatter_pair.append(set(pair_config['pair']))
            
    features_set_scatter = set()
    features_scatter_splits = {}
    split_by_set = set()
    for feature, feature_config in config['feature_prefixes_scatter'].items():
        if feature_config['use']:
            features_set_scatter.add(feature)
            features_scatter_splits[feature] = feature_config['split_by']
            for split_name in feature_config['split_by']:
                if split_name != '':
                    split_by_set.add(split_name)
    
    labels_set = PDL.get_include_set_from_dict(config['label_prefixes'])

    features_set = [features_set_hist_bin, features_set_scatter, split_by_set]
    features_set.extend(features_set_scatter_pair)
    features_set = set.union(*features_set)

    pdl = PDL(0.25, input_path, eval(config['file_nums']), features_set, set(), labels_set, split_by_objs=False)
    
    csv_hist_filename = 'histograms_statistics.csv'
    hs = CSVStatistics(os.path.join(output_path, csv_hist_filename), CSVStatistics.HIST_STATS)
    
    csv_scatter_filename = 'scatter_statistics.csv'
    ss = CSVStatistics(os.path.join(output_path, csv_scatter_filename), CSVStatistics.SCATTER_STATS)
    
    for label_name in labels_set:
        labels = pdl._raw_data[label_name]
        
        for feature_name in features_set_hist_bin:
            _plot_save_hist_bin(config, pdl._raw_data[feature_name], labels, feature_name, label_name, output_path, hs)
        
        for feature_name in features_set_scatter:
            split_by = {}
            split_by_names = features_scatter_splits[feature_name]
            for split_by_name in split_by_names:
                split_by_data = None
                if split_by_name != '':
                    split_by_data = pdl._raw_data[split_by_name]
                split_by[split_by_name] = split_by_data               
            _plot_save_scatter(config, pdl._raw_data[feature_name], split_by, labels, feature_name, label_name, output_path, ss)
        
        for features_pair in features_set_scatter_pair:
            features_pair = list(features_pair)
            features_pair.sort()
            _plot_save_scatter_pair(config, pdl.get_partial_raw_data(tuple(features_pair)), labels, features_pair, label_name, output_path, ss)
            
    logging.info("Saving {0}".format(csv_hist_filename))
    hs.save()
    logging.info("Saving {0}".format(csv_scatter_filename))
    ss.save()
            
if __name__ == '__main__':
    #read args
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(logging.INFO)
    
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    _ensure_dir_exists(args.output_path)
    
    analyze_patch_features(config, args.input_path, args.output_path)