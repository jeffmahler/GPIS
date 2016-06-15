from sklearn.metrics import mean_squared_error

class PatchesSKLearner:

    @staticmethod
    def print_mse(estimator, tr_data, tr_labels, t_data, t_labels, title):
        tr_pred = estimator.predict(tr_data)
        t_pred = estimator.predict(t_data)
        
        tr_mse = mean_squared_error(tr_pred, tr_labels)
        t_mse = mean_squared_error(t_pred, t_labels)
        
        print "{0} train mse {1}, test mse {2}".format(title, tr_mse, t_mse)
        
    @staticmethod
    def print_accuracy(estimator, tr_data, tr_labels, t_data, t_labels, title):
        tr_accu = estimator.score(tr_data, tr_labels)
        t_accu = estimator.score(t_data, t_labels)
        
        print "{0} train accuracy {1}, test accuracy {2}".format(title, tr_accu, t_accu)
        
    @staticmethod
    def train(estimator_instantiator, tr_data, tr_labels, t_data, t_labels, title, print_mse = False, params=None):
        estimator = estimator_instantiator()
        if params is not None:
            estimator.set_params(*params)
            
        estimator.fit(tr_data, tr_labels)
        PatchesSKLearner.print_accuracy(estimator, tr_data, tr_labels, t_data, t_labels, title)
        if print_mse:
            PatchesSKLearner.print_mse(estimator, tr_data, tr_labels, t_data, t_labels, title)
            
        return estimator