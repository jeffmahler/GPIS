'''
Perform Grid Search CV on learners that implement sklearn interfaces on patches data
Author: Jacky
'''

import argparse
import os
import IPython
import yaml
import logging
import csv
import shutil
import joblib

import numpy as np
from sklearn.grid_search import GridSearchCV
from patches_data_loader import PatchesDataLoader as PDL

import sys
_data_analysis_path = os.path.join(os.path.dirname(__file__), '..', 'data_analysis')
sys.path.append(_data_analysis_path)
from tflearn_gridsearchcv import TFLearnGridSearchCV

from sklearn.svm import SVC
from sklearn.qda import QDA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from tensorflow.contrib.learn import TensorFlowDNNClassifier, TensorFlowDNNRegressor

LEARNERS_MAP = {
        "SVC": SVC,
        "QDA": QDA,
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "TensorFlowDNNClassifier": lambda:TensorFlowDNNClassifier(hidden_units=[100], n_classes=2),
        "LinearRegression": LinearRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "AdaBoostRegressor": AdaBoostRegressor,
        "TensorFlowDNNRegressor": lambda:TensorFlowDNNRegressor(hidden_units=[100], n_classes=2)
}

def _ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def grid_search(config, input_path, output_path):
    #load data
    features_set = PDL.get_include_set_from_dict(config['feature_prefixes'])
    label_name = config['label_prefix']
    
    pdl = PDL(0.25, input_path, eval(config['file_nums']), features_set, set(), set([label_name]), split_by_objs=False)
    data = pdl.all_tr_data
    label = pdl._raw_data[label_name]
    
    learners = []
    for learner in config['learners']:
        if learner['use']:
            learners.append(learner)
    
    if config['save_best_learner']:
        learners_output_path = os.path.join(output_path, 'best_learners')
        _ensure_dir_exists(learners_output_path)
    
    for learner in learners:
        name = learner['name']
        params = learner['params']
    
        if name not in LEARNERS_MAP:
            logging.warn("Learner {0} unknown! Skipping.".format(name))
    
        logging.info("Performing grid search on {0}".format(name))
    
        #perform grid search. either use tf learn or sk learn
        learner_instantiator = LEARNERS_MAP[name]
        if 'TensorFlow' in name:
            cv = TFLearnGridSearchCV(learner_instantiator, params)
        else:
            cv = GridSearchCV(learner_instantiator(), [params], cv=3)
        cv.fit(data, label)
        
        logging.info("Best params found for {0} with score {1} is \n {2}".format(name, cv.best_score_, cv.best_params_))
        
        #save best estimator
        if config['save_best_learner']:
            estimator = cv.best_estimator_
            learner_filename = "best_learner_{0}_on_{1}".format(name, label_name)
            output_full_path = os.path.join(learners_output_path, learner_filename)
            if hasattr(learner, "save"):
                estimator.save(output_full_path)
            else:
                joblib.dump(estimator, output_full_path+".jbb", compress=3)
        
        #save best params to a file
        best_learner_params_name = 'best_learner_{0}_on_{1}params'.format(name, label_name)
        best_learner_config = '{0}={1}'.format(best_learner_params_name, repr(cv.best_params_))
        best_learner_params_filename = '{0}.py'.format(best_learner_params_name)
        logging.info("Saving {0}".format(best_learner_params_filename))
        with open(os.path.join(output_path, best_learner_params_filename), 'wb') as file:
            file.write(best_learner_config)
        
        #save all params + scores to csv
        #columns: params, validation score
        learner_scores_filename = 'learner_{0}_on_{1}results.csv'.format(name, label_name)
        logging.info("Saving {0}".format(learner_scores_filename))
        with open(os.path.join(output_path, learner_scores_filename), 'wb') as file:
            writer = csv.writer(file)
            writer.writerow(["params", "validation_score"])
            for grid_score in cv.grid_scores_:
                writer.writerow([grid_score.parameters, grid_score.mean_validation_score])

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
     
    #load data, perform cv, and record results
    grid_search(config, args.input_path, args.output_path)