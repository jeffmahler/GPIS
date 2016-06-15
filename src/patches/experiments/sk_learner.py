'''
Helper to perform learning and evaluate results through sklearn's interface
Author: Jacky 
'''

from sklearn.metrics import mean_squared_error, mean_absolute_error, zero_one_loss
import logging

class SKLearner:
        
    @staticmethod
    def eval_score(title, eval_name, tr_preds, tr_labels, t_preds, t_labels):
        evaler = _EVAL_MAP[eval_name]
        
        tr_score = evaler(tr_labels, tr_preds)
        t_score = evaler(t_labels, t_preds)
        
        logging.info("{0} train {1}: {2}, test {1}: {3}".format(title, eval_name, tr_score, t_score))
        
        return {
            "{0}_train".format(eval_name): tr_score,
            "{0}_test".format(eval_name): t_score
        }
        
    @staticmethod
    def train(estimator_instantiator, tr_data, tr_labels, t_data, t_labels, title, evals=[], params=None):
        estimator = estimator_instantiator()
        if params is not None:
            estimator.set_params(*params)
            
        estimator.fit(tr_data, tr_labels)
        preds = {}
        preds{'tr'} = estimator.predict(tr_data)
        preds{'t'} = estimator.predict(t_data)
        
        results = []
        for eval in evals:
            if eval not in _EVAL_MAP:
                logging.error("Given evaluation method does not exist: {0}".format(eval))
                continue
            else:
                results.append(PatchesSKLearner.eval_score(title, eval, preds['tr'], tr_labels, preds['t'], t_labels))
            
        return estimator, preds, results
        
_EVAL_MAP = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "accuracy": lambda true, pred: 1 - zero_one_loss(true, pred, normalize=True),
}