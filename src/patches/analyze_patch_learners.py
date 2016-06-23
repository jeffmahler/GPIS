'''
Performs learning and corresponding analyses on patches data.
Author: Jacky Liang
'''

import numpy as np
import IPython
import logging
import yaml
import argparse
import os
import shutil
import joblib
import csv

#learners
from sklearn.svm import SVC
from sklearn.qda import QDA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from tensorflow.contrib.learn import TensorFlowDNNClassifier, TensorFlowDNNRegressor

#metrics
from sklearn.metrics import zero_one_loss, log_loss, r2_score

#misc
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from patches_data_loader import PatchesDataLoader as PDL
from sk_learner import SKLearner

import matplotlib
matplotlib.use('Agg')#prevents using X server backend for matplotlib
import matplotlib.pyplot as plt

import sys
_grasp_selection_path = os.path.join(os.path.dirname(__file__), '..', 'grasp_selection')
_data_analysis_path = os.path.join(os.path.dirname(__file__), '..', 'data_analysis')
sys.path.append(_grasp_selection_path)
sys.path.append(_data_analysis_path)
import wrap_text
import plotting
from csv_statistics import CSVStatistics
from error_statistics import ContinuousErrorStats
from confusion_matrix import BinaryConfusionMatrix

LEARNERS_MAP = {
    "classifiers":{#preferrably classifiers that implement predict_proba
        "SVC": SVC,
        "QDA": QDA,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "TensorFlowDNNClassifier": lambda:TensorFlowDNNClassifier(hidden_units=[100], n_classes=2)
    },
    "regressors":{
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "AdaBoostRegressor": AdaBoostRegressor,
        "TensorFlowDNNRegressor": lambda:TensorFlowDNNRegressor(hidden_units=[100], n_classes=2)
    }
}

_CLASSIFIER_TYPE = 'classifiers'
_REGRESSOR_TYPE = 'regressors'

def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class _Results:

    def __init__(self, output_path):
        self._output_path = output_path
        self._results = []
        self._result_names = []
        
    def append_result(self, learner_name, label_name, results={}):
        self._results.append((learner_name, label_name, results))
        for key in results.keys():
            if key not in self._result_names:
                self._result_names.append(key)
                
    def save(self, name):
        output_filename = "{0}.csv".format(name)
        output_full_path = os.path.join(self._output_path, output_filename)
        logging.info("Saving {0}".format(output_filename))
        
        with open(output_full_path, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            
            title_row = ["Learner Name", "Label Name"] + self._result_names
            csvwriter.writerow(title_row)
            
            for result in self._results:
                row = list(result[:2])
                for result_name in self._result_names:
                    if result_name in result[2]:
                        row.append(result[2][result_name])
                    else:
                        row.append('')
                csvwriter.writerow(row)
        
def _classification_post_process(pdl, estimator, preds, learner_name, label_name, output_path, config):
    def filter_measure(type, pred_val, true_val):
        return lambda i: preds[type][i] == pred_val and pdl.labels[label_name][type][i] == true_val
    
    tr_indices = [i for i in range(len(preds['tr']))]
    t_indices = [i for i in range(len(preds['t']))]
    
    #find false positives - where preds = 1 and actual = 0
    fp_tr = np.array(filter(filter_measure('tr', 1, 0), tr_indices))
    fp_t = np.array(filter(filter_measure('t', 1, 0), t_indices))
    
    #find false negatives - where preds = 0 and actual = 1
    fn_tr = np.array(filter(filter_measure('tr', 0, 1), tr_indices))
    fn_t = np.array(filter(filter_measure('t', 0, 1), t_indices))
    
    n_to_save = 10
    
    #use predict_proba to find the worst fp and fn if possible. else randomly sample
    if hasattr(estimator, 'predict_proba'):
        tr_proba = estimator.predict_proba(pdl.tr)
        t_proba = estimator.predict_proba(pdl.t)
        
        #get corresponding probabilities and
        #get most erroneous predictions based on probabilities
        if len(fp_tr) > 0:
            fp_tr_proba = np.take(tr_proba, fp_tr)
            fp_tr = fp_tr[np.argsort(fp_tr_proba)][-n_to_save:]
        if len(fp_t) > 0:
            fp_t_proba = np.take(t_proba, fp_t)
            fp_t = fp_t[np.argsort(fp_t_proba)][-n_to_save:]
        if len(fn_tr) > 0:
            fn_tr_proba = np.take(tr_proba, fn_tr)
            fn_tr = fn_tr[np.argsort(fn_tr_proba)][:n_to_save]
        if len(fn_t) > 0:
            fn_t_proba = np.take(t_proba, fn_t)
            fn_t = fn_t[np.argsort(fn_t_proba)][:n_to_save]
        
    else:
        logging.warn("{0} does not implement predict_proba, false positives and negatives will be randomly sampled.")
        fp_tr = np.random.choice(fp_tr, min(n_to_save, len(fp_tr)))
        fp_t = np.random.choice(fp_t, min(n_to_save, len(fp_t)))
        fn_tr = np.random.choice(fn_tr, min(n_to_save, len(fn_tr)))
        fn_t = np.random.choice(fn_t, min(n_to_save, len(fn_t)))
        
    metas = {
        'fp_tr': pdl.get_all_meta('tr', fp_tr),
        'fp_t': pdl.get_all_meta('t', fp_t),
        'fn_tr': pdl.get_all_meta('tr', fn_tr),
        'fn_t': pdl.get_all_meta('t', fn_t)
    }
    
    #save fn/fp to a file
    output_path = os.path.join(output_path, 'false_positives_negatives')
    _ensure_dir_exists(output_path)
    output_filename_template = "fnp_{0}_{1}_{2}.csv".format(learner_name, label_name, '{0}')
    
    for name, data in metas.items():
        output_filename = output_filename_template.format(name)
        logging.info("Saving {0}".format(output_filename))
        
        with open(os.path.join(output_path, output_filename), 'wb') as csvfile:
            writer = csv.writer(csvfile)
            header = list(pdl._metadata_set)
            
            writer.writerow(header)
            for i in range(data.shape[0]):
                writer.writerow(data[i,:].tolist())
    
    #TODO Visualize patches
    #TODO use confusion matrix to generate and save confusion matrix stats, as well as visualization

def _regression_post_process_gen(err_stats, tr_stats, t_stats):
    #plotting params
    font_size = config['plotting']['font_size']
    num_bins = config['plotting']['num_bins']
    dpi = config['plotting']['dpi']
    normalize_hist = config['plotting']['normalize_hist']
    
    def _regression_post_process(pdl, estimator, preds, learner_name, full_label_name, output_path, config):
        #generate and save histograms of errors of all metrics        
        label_name, variant = full_label_name[0], full_label_name[1]
        title = "{0}_{1}{2}".format(learner_name, label_name, variant)
        
        #plot errors
        error_stats_tr = ContinuousErrorStats(pdl.get_labels_variants(label_name, variant)['tr'], preds['tr'], title)
        error_stats_t = ContinuousErrorStats(pdl.get_labels_variants(label_name, variant)['t'], preds['t'], title)
        
        output_path_err = os.path.join(output_path, 'regression_error_histograms')
        _ensure_dir_exists(output_path_err)
        
        error_stats_tr.plot_error_histograms(output_dir=output_path_err, normalize=normalize_hist, show_stats=True, csvstats=err_stats)
        error_stats_t.plot_error_histograms(output_dir=output_path_err, normalize=normalize_hist, show_stats=True, csvstats=err_stats)
        
        #plot predictions
        title_tr = "{0}_train".format(title)
        title_t = "{0}_test".format(title)
        
        output_path_preds = os.path.join(output_path, 'regression_preds_histograms')
        _ensure_dir_exists(output_path_preds)
        
        plt.figure()
        plotting.plot_histogram(preds['tr'], num_bins=num_bins, normalize=True, show_stats=True)
        plt.title(wrap_text.wrap(title_tr) + '\nHistogram')
        plt.ylabel('Normalized Density', fontsize=font_size)
        plt.xlabel(wrap_text.wrap(title_tr), fontsize=font_size)
        plt.tight_layout()
        figname = 'preds_{0}_histogram.pdf'.format(title_tr)
        logging.info("Saving {0}".format(figname))
        plt.savefig(os.path.join(output_path_preds, figname), dpi=dpi)
        plt.close()
        
        plt.figure()
        plotting.plot_histogram(preds['t'], num_bins=num_bins, normalize=True, show_stats=True)
        plt.title(wrap_text.wrap(title_t) + '\nHistogram')
        plt.ylabel('Normalized Density', fontsize=font_size)
        plt.xlabel(wrap_text.wrap(title_t), fontsize=font_size)
        plt.tight_layout()
        figname = 'preds_{0}_histogram.pdf'.format(title_t)
        logging.info("Saving {0}".format(figname))
        plt.savefig(os.path.join(output_path_preds, figname), dpi=dpi)
        plt.close()
        
        #saving stats to csv
        tr_stats.append_data(title_tr, preds['tr'])
        t_stats.append_data(title_t, preds['t'])
        
    return _regression_post_process
        
def eval_learn(config, input_path, output_path):
    #plotting params
    font_size = config['plotting']['font_size']
    num_bins = config['plotting']['num_bins']
    dpi = config['plotting']['dpi']

    #read config about which files to include
    features_set = PDL.get_include_set_from_dict(config["feature_prefixes"])
    metadata_set = PDL.get_include_set_from_dict(config["metadata_prefixes"])
    regression_labels_set = PDL.get_include_set_from_dict(config["regression_label_prefixes"])
    classification_labels_set = PDL.get_include_set_from_dict(config["classification_label_prefixes"])
    
    regression_label_variants_set = PDL.get_include_set_from_dict(config['regression_label_variants'])
    
    labels_set_map = {
        _CLASSIFIER_TYPE: classification_labels_set,
        _REGRESSOR_TYPE: regression_labels_set
    }
                    
    #load data
    pdl = PDL(config['test_ratio'], input_path, eval(config['file_nums']), features_set, metadata_set,
                   classification_labels_set.union(regression_labels_set), config['split_by_objs'])
                                        
    #make folder of saved learners if needed
    learners_output_path = os.path.join(output_path, "learners")
    if config['save_learners']:
        _ensure_dir_exists(learners_output_path)

    #save histograms of regressor labels
    regression_labels_filename = 'regression_labels_stats.csv'
    regression_labels_hists_path = os.path.join(output_path, 'regression_labels_histograms')
    _ensure_dir_exists(regression_labels_hists_path)
    labels_hs = CSVStatistics(os.path.join(output_path, regression_labels_filename), CSVStatistics.HIST_STATS)
    for label_name in labels_set_map['regressors']:
        data = pdl._raw_data[label_name]
        plt.figure()
        plotting.plot_histogram(data, num_bins=num_bins, normalize=True, show_stats=True)
        plt.ylabel('Normalized Density', fontsize=font_size)
        plt.xlabel(wrap_text.wrap(label_name), fontsize=font_size)
        plt.title(wrap_text.wrap(label_name) + '\n Histogram')
        plt.tight_layout()        
        figname = 'metric_{0}_histogram.pdf'.format(label_name)
        logging.info("Saving {0}".format(figname))
        plt.savefig(os.path.join(regression_labels_hists_path, figname), dpi=dpi)
        plt.close()
        
        labels_hs.append_data(label_name, data)
        
    logging.info("Saving {0}".format(regression_labels_filename))
    labels_hs.save()
        
    def do_learn(learner_type, post_process):
        #records of all training and test results for all learners
        all_results = _Results(output_path)
        metrics = config['metrics']
        
        for learner_name, learner in LEARNERS_MAP[learner_type].items():
            if learner_name not in config['learners']:
                logging.warn("Specified learner not found in config. Skipping: {0}".format(name))
                continue
            
            if config['learners'][learner_name]['use']:
                #check for params for given learner
                all_params = {'':None}
                if 'params' in config['learners'][learner_name]:
                    all_params = config['learners'][learner_name]['params']

                    #if only one set of params is provided
                    if not config['learners'][learner_name]['multi_params']:
                        all_params = {'': all_params}
                    
                for label_name in labels_set_map[learner_type]:
                    for suffix, params in all_params.items():
                        full_learner_name = "{0}_{1}".format(learner_name, suffix)
                        
                        #get list of label variants
                        if learner_type == _REGRESSOR_TYPE:
                            all_labels = []
                            for variant in regression_label_variants_set:
                                all_labels.append({
                                    'name': variant,
                                    'tr': pdl.get_labels_variants(label_name, variant)['tr'],
                                    't': pdl.get_labels_variants(label_name, variant)['t']
                                })
                        else:
                            all_labels = [{
                                            'name': '',
                                            'tr': pdl.labels[label_name]['tr'],
                                            't': pdl.labels[label_name]['t']
                                        }]
                                        
                        for label in all_labels:
                            full_label_name = '{0}{1}'.format(label_name, label['name'])
                            
                            #training and score evaluations
                            estimator, preds, results = SKLearner.train(learner, pdl.tr, label['tr'], pdl.t, label['t'], full_learner_name,
                                                                                         metrics=metrics[learner_type], params=params)
                            #save learner if needed
                            if config['save_learners']:
                                output_filename = "{0}{1}".format(full_label_name, full_learner_name)
                                output_full_path = os.path.join(learners_output_path, output_filename)
                                if hasattr(estimator, "save"):
                                    estimator.save(output_full_path)
                                else:
                                    joblib.dump(estimator, output_full_path+".jbb", compress=3)
                            
                            all_results.append_result(full_learner_name, full_label_name, results)

                            post_process(pdl, estimator, preds, full_learner_name, (label_name, label['name']), output_path, config)
                            
                            del estimator
        
        all_results.save("{0}_results".format(learner_type))
        
    do_learn(_CLASSIFIER_TYPE, _classification_post_process)
    
    #saving stats about regression results
    regression_results_stats_output_path = os.path.join(output_path, 'regression_results_stats')
    regression_err_csv_filename = 'regression_err_stats.csv'
    regression_tr_csv_filename = 'regression_train_stats.csv'
    regression_t_csv_filename = 'regression_test_stats.csv'
    err_stats = CSVStatistics(os.path.join(regression_results_stats_output_path, regression_err_csv_filename), CSVStatistics.HIST_STATS)
    tr_stats = CSVStatistics(os.path.join(regression_results_stats_output_path, regression_tr_csv_filename), CSVStatistics.HIST_STATS)
    t_stats = CSVStatistics(os.path.join(regression_results_stats_output_path, regression_t_csv_filename), CSVStatistics.HIST_STATS)
    do_learn(_REGRESSOR_TYPE, _regression_post_process_gen(err_stats, tr_stats, t_stats))
    logging.info('Saving {0}'.format(regression_err_csv_filename))
    err_stats.save()
    logging.info('Saving {0}'.format(regression_tr_csv_filename))
    tr_stats.save()
    logging.info('Saving {0}'.format(regression_t_csv_filename))
    t_stats.save()

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
        
    #save config
    shutil.copy(args.config, os.path.join(args.output_path, os.path.basename(args.config)))
     
    #load data, perform learning, and record results
    eval_learn(config, args.input_path, args.output_path)