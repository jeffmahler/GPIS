'''
Helper to perform learning and evaluate results through sklearn's interface
Author: Jacky 
'''

from sklearn.metrics import mean_squared_error, mean_absolute_error, zero_one_loss, r2_score
import logging
import IPython

class SKLearner:
        
    @staticmethod
    def metric_score(title, metric_name, tr_preds, tr_labels, t_preds, t_labels):
        metric_method = _METRICS_MAP[metric_name]
        
        tr_score = metric_method(tr_labels, tr_preds)
        t_score = metric_method(t_labels, t_preds)
        
        logging.info("{0} train {1}: {2}, test {1}: {3}".format(title, metric_name, tr_score, t_score))
        
        return {
            "{0}_train".format(metric_name): tr_score,
            "{0}_test".format(metric_name): t_score
        }
        
    @staticmethod
    def train(estimator_instantiator, tr_data, tr_labels, t_data, t_labels, title, metrics=[], params=None):
        estimator = estimator_instantiator()
        if params is not None:
            estimator.set_params(**params)
            
            
        estimator.fit(tr_data, tr_labels)
        preds = {}
        preds['tr'] = estimator.predict(tr_data)
        preds['t'] = estimator.predict(t_data)
        
        results = {}
        for metric in metrics:
            if metric not in _METRICS_MAP:
                logging.error("Given metric does not exist: {0}".format(metric))
                continue
            else:
                results.update(SKLearner.metric_score(title, metric, preds['tr'], tr_labels, preds['t'], t_labels))
            
        return estimator, preds, results
        
_METRICS_MAP = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "accuracy": lambda true, pred: 1 - zero_one_loss(true, pred, normalize=True),
    'r2_score': r2_score
}