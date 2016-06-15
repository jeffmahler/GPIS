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

#learners
from sklearn.qda import QDA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from tensorflow.contrib.learn import TensorFlowDNNClassifier, TensorFlowDNNRegressor

#metrics
from sklearn.metrics import zero_one_loss, log_loss, r2_score

#misc
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from patches_data_loader import PatchesDataLoader
from sk_learner import SKLearner
from loop_time_forecaster import LoopTimeForecaster

LEARNERS_MAP = {
    "classifiers":{#preferrably classifiers that implement predict_proba
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
        for key in results:
            if key not in self._result_names:
                self._result_names.append(key)
                
    #TODO
    def save(self):
        return
        
def _classification_post_process(pdl, estimator, preds, learner_name, label_name, output_path):
    def filter_measure(type, pred_val, true_val):
        return lambda i: preds[type][i] == pred_val and pdl.labels[label_name][type][i] == true_val
    
    tr_indices = [i for i in range(len(preds['tr'])]
    t_indices = [i for i in range(len(preds['t'])]
    
    #find false positives - where preds = 1 and actual = 0
    fp_tr = filter(filter_measure('tr', 1, 0), tr_indices)
    fp_t = filter(filter_measure('t', 1, 0), t_indices)
    
    #find false negatives - where preds = 0 and actual = 1
    fn_tr = filter(filter_measure('tr', 0, 1), tr_indices)
    fn_t = filter(filter_measure('t', 0, 1), t_indices)
    
    n_to_save = 10
    
    if hasattr(estimator, 'predict_proba'):
        tr_proba = estimator.predict_proba(pdl.tr)
        t_proba = estimator.predict_proba(pdl.t)
        
        #get corresponding probabilities
        fp_tr_proba = np.take(tr_proba, fp_tr)
        fp_t_proba = np.take(t_proba, fp_t)
        fn_tr_proba = np.take(tr_proba, fn_tr)
        fn_t_proba = np.take(t_proba, fn_t)
        
        #get most erroneous 10 predictions based on probabilities
        fp_tr = fp_tr[np.argsort(fp_tr_proba)][-10:]
        fp_t = fp_t[np.argsort(fp_t_proba)][-10:]
        fn_tr = fn_tr[np.argsort(fn_tr_proba)][:10]
        fn_t = fn_t[np.argsort(fn_t_proba)][:10]
    else:
        logging.warn("{0} does not implement predict_proba, false positives and negatives will be randomly sampled.")
        fp_tr = np.random.choice(fp_tr, min(n_to_save, len(fp_tr)))
        fp_t = np.random.choice(fp_t, min(n_to_save, len(fp_t)))
        fn_tr = np.random.choice(fn_tr, min(n_to_save, len(fn_tr)))
        fn_t = np.random.choice(fn_t, min(n_to_save, len(fn_t)))
        
    #TODO implement interface in pdl
    metas = {
        'fp_tr': pdl.get_all_meta('tr', fp_tr),
        'fp_t': pdl.get_all_meta('t', fp_t),
        'fn_tr': pdl.get_all_meta('tr', fn_tr),
        'fn_t': pdl.get_all_meta('t', fn_t_meta)
    }
    
    #save fn/fp to a file
    output_path = os.path.join(output_path, 'false_positives_negatives')
    _ensure_dir_exists(output_full_path)
    output_filename = "fnp_{0}_{1}.jbb".format(learner_name, label_name)
    
    logging.info("Saving {0}".format(output_filename))
    joblib.dump(metas, os.path.join(output_path, output_filename)
    
    #TODO Visualize patches
    
def _regression_post_process(pdl, estimator, preds, learner_name, label_name, output_path):
    #generate and save histograms of errors of all metrics
        
def eval_learn(config, input_path, output_path):
    #read config about which files to include
    features_set = set()
    metadata_set = set()
    regression_labels_set = set()
    classification_labels_set = set()
    
    def build_include_set(tag, target):
        for name, use in config[tag].items():
            if use:
                target.add(name)

    build_include_set("feature_prefixes", features_set)
    build_include_set("metadata_prefixes", metadata_set)
    build_include_set("classification_label_prefixes", classification_labels_set)
    build_include_set("regression_label_prefixes", regression_labels_set)
                    
    #load data
    #TODO
    pdl = PatchesDataLoader(config['test_ratio'], input_path, file_nums=eval(config['file_nums']), 
                                        features_set=features_set, metadata_set=metadata_set,
                                        labels_set=classification_labels_set.union(regression_labels_set), 
                                        split_by_objs=config['split_by_objs'])
                                        
    #make folder of saved learners if needed
    learners_output_path = os.path.join(output_path, "learners")
    if config['save_learner']:
        _ensure_dir_exists(learners_output_path)

    def do_learn(learner_type, post_process):
        #records of all training and test results for all learners
        all_results = _Results(os.path.join(output_path, 'results_{0}.csv'.format(learner_type)))
        
        for learner_name, learner in LEARNERS_MAP[learner_type].items():
            if learner_name not in config['learners']:
                logging.warn("Specified learner not found in config. Skipping: {0}".format(name))
                continue
            
            if config['learners'][learner_name]['use']:
                #check for params for given learner
                params = None
                if 'params' in config['learners'][learner_name]:
                    params = config['learners'][learner_name]['params']
                    for key, val in params:
                        params[key] = eval(str(val))

                for label_name in classification_labels_set:
                    #training and score evaluations
                    estimator, preds, results = SKLearner.train(learner, pdl.tr, pdl.labels[label_name]['tr'], 
                                                                                    pdl.t, pdl.labels[label_name]['t'], learner_name
                                                                                    evals=['score'], params=params)
                    #save learner if needed
                    if config['save_learner']:
                        output_filename = "{0}_{1}.jbb".format(label_name, learner_name)
                        output_full_path = os.path.join(learners_output_path, output_filename)
                        if hasattr(estimator, "save"):
                            estimator.save(output_full_path)
                        else:
                            joblib.dump(estimator, output_full_path)
                    
                    all_results.append_result(learner_name, label_name, results)
                    
                    post_process(pdl, estimator, preds, learner_name, label_name, output_path)
                    
                    del estimator
        
        all_results.save()
        
    do_learn("classifiers", _classification_post_process)
    do_learn("regressors", _regression_post_process)

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
    IPython.embed()
    exit(0)
    #save config
    shutil.copy(args.config, os.path.join(args.output_path, os.path.basename(args.config)))
     
    #load data, perform learning, and record results
    eval_learn(config, args.input_path, args.output_path)