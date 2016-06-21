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

import sys
_grasp_selection_path = os.path.join(os.path.dirname(__file__), '..', 'grasp_selection')
_data_analysis_path = os.path.join(os.path.dirname(__file__), '..', 'data_analysis')
sys.path.append(_grasp_selection_path)
sys.path.append(_data_analysis_path)
import wrap_text
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

def _regression_post_process_gen(hs):
    
    def _regression_post_process(pdl, estimator, preds, learner_name, label_name, output_path, config):
        #generate and save histograms of errors of all metrics
        title = "{0}_{1}".format(learner_name, label_name)
        
        error_stats_tr = ContinuousErrorStats(pdl.labels[label_name]['tr'], preds['tr'], title)
        error_stats_t = ContinuousErrorStats(pdl.labels[label_name]['t'], preds['t'], title)
        
        output_path = os.path.join(output_path, 'regression_error_histograms')
        _ensure_dir_exists(output_path)
        
        error_stats_tr.plot_error_histograms(output_dir=output_path, normalize=config['normalize'], show_stats=True, csvstats=hs)
        error_stats_t.plot_error_histograms(output_dir=output_path, normalize=config['normalize'], show_stats=True, csvstats=hs)
        
    return _regression_post_process
        
def eval_learn(config, input_path, output_path):
    #read config about which files to include
    features_set = PDL.get_include_set_from_dict(config["feature_prefixes"])
    metadata_set = PDL.get_include_set_from_dict(config["metadata_prefixes"])
    regression_labels_set = PDL.get_include_set_from_dict(config["regression_label_prefixes"])
    classification_labels_set = PDL.get_include_set_from_dict(config["classification_label_prefixes"])
    
    labels_set_map = {
        'classifiers':classification_labels_set,
        'regressors':regression_labels_set
    }
                    
    #load data
    pdl = PDL(config['test_ratio'], input_path, eval(config['file_nums']), features_set, metadata_set,
                   classification_labels_set.union(regression_labels_set), config['split_by_objs'])
                                        
    #make folder of saved learners if needed
    learners_output_path = os.path.join(output_path, "learners")
    if config['save_learners']:
        _ensure_dir_exists(learners_output_path)

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
                params = None
                if 'params' in config['learners'][learner_name]:
                    params = config['learners'][learner_name]['params']

                for label_name in labels_set_map[learner_type]:
                    #training and score evaluations
                    estimator, preds, results = SKLearner.train(learner, pdl.tr, pdl.labels[label_name]['tr'], 
                                                                                    pdl.t, pdl.labels[label_name]['t'], learner_name,
                                                                                    metrics=metrics[learner_type], params=params)
                    #save learner if needed
                    if config['save_learners']:
                        output_filename = "{0}{1}".format(label_name, learner_name)
                        output_full_path = os.path.join(learners_output_path, output_filename)
                        if hasattr(estimator, "save"):
                            estimator.save(output_full_path)
                        else:
                            joblib.dump(estimator, output_full_path+".jbb", compress=3)
                    
                    all_results.append_result(learner_name, label_name, results)
                    
                    post_process(pdl, estimator, preds, learner_name, label_name, output_path, config)
                    
                    del estimator
        
        all_results.save("{0}_results".format(learner_type))
        
    do_learn("classifiers", _classification_post_process)
    
    regression_csv_filename = 'regression_stats.csv'
    hs = CSVStatistics(os.path.join(output_path, regression_csv_filename), CSVStatistics.HIST_STATS)
    do_learn("regressors", _regression_post_process_gen(hs))
    
    logging.info('Saving {0}'.format(regression_csv_filename))
    hs.save()

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