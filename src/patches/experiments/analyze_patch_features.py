import numpy as np
import argparse
import yaml
import os
import matplotlib
matplotlib.use('Agg')#prevents using X server backend for matplotlib
import matplotlib.pyplot as plt
from patches_data_loader import PatchesDataLoader as PDL
import sys
_grasp_selection_path = os.path.join(os.path.dirname(__file__), '..', '..', 'grasp_selection')
sys.path.append(_grasp_selection_path)
import plotting

def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def _plot_save_hist(config, data, labels, feature_name, label_name, output_path):
    #read plot config
    font_size = config['plotting']['font_size']
    num_bins = config['plotting']['num_bins']
    dpi = config['plotting']['dpi']
    
    #get data and compute statistics
    positive_metrics = np.take(labels, np.argwhere(data == 1).flatten())
    negative_metrics = np.take(labels, np.argwhere(data == 0).flatten()) 
    
    min_range = min(np.min(positive_metrics), np.min(negative_metrics))
    max_range = max(np.max(positive_metrics), np.max(negative_metrics))
    
    pos_mean = np.mean(positive_metrics)
    pos_median = np.median(positive_metrics)
    pos_std = np.std(positive_metrics)

    neg_mean = np.mean(negative_metrics)
    neg_median = np.median(negative_metrics)
    neg_std = np.std(negative_metrics)

    msg_template = "mean:{:.3g}\nmedian:{:.3g}\nstd:{:.3g}"
    pos_msg = msg_template.format(pos_mean, pos_median, pos_std)
    neg_msg = msg_template.format(neg_mean, neg_median, neg_std)
    
    #plotting
    textbox_props = {'boxstyle':'square', 'facecolor':'white'}
    
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle('Metric {0} Density'.format(label_name), fontsize=font_size)
    
    ax = plt.subplot("121")
    ax.set_title("{0}=1".format(feature_name), fontsize=font_size)
    plt.ylabel('Normalized Density', fontsize=font_size)
    plt.xlabel(label_name, fontsize=font_size)
    plotting.plot_histogram(positive_metrics, min_range=min_range, max_range=max_range, 
                                        num_bins=num_bins, normalize=True)
    ax.text(0.05, 0.95, pos_msg, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=textbox_props, alpha=0.7)

    ax = plt.subplot("122")
    ax.set_title("{0}=0".format(feature_name), fontsize=font_size)
    plt.ylabel('Normalized Density', fontsize=font_size)
    plt.xlabel(label_name, fontsize=font_size)
    plotting.plot_histogram(negative_metrics, min_range=min_range, max_range=max_range, 
                                        num_bins=num_bins, normalize=True)
    ax.text(0.05, 0.95, neg_msg, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=textbox_props, alpha=0.7)
    ax.set_position((0, -0.2, 1, 1.2))
    plt.subplots_adjust(top=0.8)
    
    figname = 'metric_{0}{1}histogram.pdf'.format(label_name, feature_name)
    plt.savefig(os.path.join(output_path, figname), dpi=dpi)
    plt.close()

def _plot_save_scatter(config, data, labels, feature_name, label_name, output_path):
    return
    
def analyze_patch_features(config, input_path, output_path):
    #load data
    features_set_hist = PDL.get_include_set_from_dict(config['feature_prefixes_hist'])
    features_set_scatter = PDL.get_include_set_from_dict(config['feature_prefixes_scatter'])
    
    labels_set = PDL.get_include_set_from_dict(config['label_prefixes'])
    
    pdl = PDL(0.25, input_path, eval(config['file_nums']), features_set_hist.union(features_set_scatter), 
                    set(), labels_set, split_by_objs=False)
    
    for label_name in labels_set:
        labels = pdl._raw_data[label_name]
        
        for feature_name in features_set_hist:
            _plot_save_hist(config, pdl._raw_data[feature_name], labels, feature_name, label_name, output_path)
            
        for feature_name in features_set_scatter:
            pass
            _plot_save_scatter(config, pdl._raw_data[feature_name], labels, feature_name, label_name, output_path)
    
if __name__ == '__main__':
    #read args
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    
    with open(args.config) as config_file:
        config = yaml.safe_load(config_file)
    
    _ensure_dir_exists(args.output_path)
    
    analyze_patch_features(config, args.input_path, args.output_path)