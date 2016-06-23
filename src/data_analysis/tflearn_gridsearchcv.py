'''
Custom Grid Search wrapper for TFLearn learners
Author: Jacky
'''

from sklearn.cross_validation import train_test_split
from collections import namedtuple
import itertools

import sys
_util_path = os.path.join(os.path.dirname(__file__), '..', 'util')

sys.path.append(_util_path)
from loop_time_forecaster import LoopTimeForecaster

class TFLearnGridSearchCV:

    _GRID_SCORE = namedtuple('_GridScore', 'parameters mean_validation_score')
    
    def __init__(self, estimator_instantiator, params, cv=0.25, random_state=None):
        self._estimator_instantiator = estimator_instantiator
        self._cv = cv
        self._random_state = random_state
        self._best_score = None
        self._best_params = None
        self._best_estimator = None
        self._grid_scores = None
        self._params = params
        self._params_grid = None
        self._generate_params_grid()
        
    def _generate_params_grid(self):
        self._params_grid = []
        keys = self._params.keys()
        options = self._params.values()
        options_combos = itertools.product(*options)

        for options_combo in options_combos:
            param = {}
            for i in range(len(keys)):
                param[keys[i]] = options_combo[i]

            self._params_grid.append(param)

    def fit(self, X, y):
        self._grid_scores = []
        self._best_score = float('-inf')
        self._best_params = None
        self._best_estimator = None
        
        X_tr, X_v, y_tr, y_v = train_test_split(X, y, test_size=self._cv)
        
        ltf = LoopTimeForecaster(len(self._params_grid), period=100)
        for params in self._params_grid:
            ltf.record_loop_start()
            estimator = self._estimator_instantiator().set_params(**params)
            estimator.fit(X_tr, y_tr)
            score = estimator.score(X_v, y_v)
            self._grid_scores.append(TFLearnGridSearchCV._GRID_SCORE(params, score))
            
            if score > self._best_score:
                self._best_score = score
                self._best_params = params
                self._estimator = estimator
            else:
                del estimator
            ltf.record_loop_end()
            ltf.report_forecast()
                
    @staticmethod
    def get_hidden_units_params(layer_nums, layer_sizes):
        all_hidden_unit_combos = []
        for layer_num in layer_nums:
            hidden_unit_combos = [combo for combo in itertools.permutations(layer_sizes, layer_num)]
            all_hidden_unit_combos.extend(hidden_unit_combos)
        return all_hidden_unit_combos
                
    @property
    def best_params_(self):
        return self._best_params
    
    @property
    def best_score_(self):
        return self._best_score
    
    @property
    def grid_scores_(self):
        return self._grid_scores
    
    @property
    def best_estimator_(self):
        return self._best_estimator
    
    def predict(self, X, y):
        return self._best_estimator.predict(X, y)
    
    def score(self, X, y):
        return self._best_estimator.score(X, y)