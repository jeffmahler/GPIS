import os
import numpy as np
from sklearn.metrics import mean_squared_error

FEATURES = (
    ('moment_arms', 'moment_arms'),
    ('patch_ori', 'patch_orientation'),
    ('w1_proj', 'w1_projection_window'),
    ('w2_proj', 'w2_projection_window'),
    ('w1_normal', 'w1_projection_window_approx_normals'),
    ('w2_normal', 'w2_projection_window_approx_normals')
)

LABELS = (
    ('pfc_20', 'pfc_f_0.200000_tg_0.010000_rg_0.200000_to_0.010000_ro_0.200000'),
    ('pfc_10', 'pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000'),
    ('pfc_5', 'pfc_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000'),
    ('ferrari', 'ferrari_canny_L1'),
    ('fc', 'force_closure')
)

class PatchesDataLoader:

    def __init__(self, p_test, data_path, file_nums=[0]):
        self._file_nums = file_nums
        self._data_path = data_path
        self._p_test = p_test
        
        self.partial_data = {}
        
        self.load()
        self.concat()
        self.tr, self.t, self.indices = PatchesDataLoader.split_train_test(self.all_tr_data, self._p_test)
        self.labels = {}
        for label in LABELS:
            label_type = label[0]
            self.labels[label_type] = {}
            self.labels[label_type]['tr'], self.labels[label_type]['t'], _ = PatchesDataLoader.split_train_test(self._raw_data[label_type], self._p_test, self.indices)
    
    def get_partial_train_data(self, feature_set):
        if feature_set not in self.partial_data:
            data = self._raw_data[feature_set[0]]
            tr_data = data.reshape(data.shape[0], -1)
            for feature in feature_set[1:]:
                data = self._raw_data[feature]
                tr_data = np.c_[tr_data, data.reshape(data.shape[0], -1)]
            
            tr, t, _ = PatchesDataLoader.split_train_test(tr_data, self._p_test, self.indices)
            self.partial_data[feature_set] = {"tr":tr, "t":t}
        
        return self.partial_data[feature_set]
    
    def _ensure_double_digits(self, num):
        if num < 10:
            return"0{0}".format(num)
        return num            
    
    def _construct_file_path(self, name, num):
        return os.path.join(self._data_path, "{0}_{1}.npz".format(name, self._ensure_double_digits(num)))
    
    def load(self):
        file_num = self._file_nums[0]
        self._raw_data = {}
        
        for feature in FEATURES:
            self._raw_data[feature[0]] = np.load(self._construct_file_path(feature[1], file_num))['arr_0']
        for label in LABELS:
            self._raw_data[label[0]] = np.load(self._construct_file_path(label[1], file_num))['arr_0']

        for file_num in self._file_nums[1:]:
            for feature in FEATURES:
                self._raw_data[feature[0]] = np.r_[self._raw_data[feature[0]], np.load(self._construct_file_path(feature[1], file_num))['arr_0']]
            for label in LABELS:
                self._raw_data[label[0]] = np.r_[self._raw_data[label[0]], np.load(self._construct_file_path(label[1], file_num))['arr_0']]

    def concat(self):
        first_data = self._raw_data[FEATURES[0][0]]
        self.all_tr_data = first_data.reshape(first_data.shape[0], -1)
        for feature in FEATURES[1:]:
            data = self._raw_data[feature[0]]
            self.all_tr_data = np.c_[self.all_tr_data, data.reshape(data.shape[0], -1)]
        
    @staticmethod
    def split_train_test(X, p_test, indices=None):
        n = X.shape[0]
        if indices is None:
            indices = [i for i in range(n)]
            np.random.shuffle(indices)
        
        split = int(n*p_test)
        return np.take(X, indices[split:], axis=0), np.take(X, indices[:split], axis=0), indices

class PatchesSKLearner:

    @staticmethod
    def print_mse(predictor, tr_data, tr_labels, t_data, t_labels, title):
        tr_pred = predictor.predict(tr_data)
        t_pred = predictor.predict(t_data)
        
        tr_mse = mean_squared_error(tr_pred, tr_labels)
        t_mse = mean_squared_error(t_pred, t_labels)
        
        print "{0} train mse {1}, test mse {2}".format(title, tr_mse, t_mse)
        
    @staticmethod
    def print_accuracy(predictor, tr_data, tr_labels, t_data, t_labels, title):
        tr_accu = predictor.score(tr_data, tr_labels)
        t_accu = predictor.score(t_data, t_labels)
        
        print "{0} train accuracy {1}, test accuracy {2}".format(title, tr_accu, t_accu)
        
    @staticmethod
    def train(PREDICTOR, tr_data, tr_labels, t_data, t_labels, title, print_mse = False):
        predictor = PREDICTOR()
        predictor.fit(tr_data, tr_labels)
        PatchesSKLearner.print_accuracy(predictor, tr_data, tr_labels, t_data, t_labels, title)
        if print_mse:
            PatchesSKLearner.print_mse(predictor, tr_data, tr_labels, t_data, t_labels, title)
            
        return predictor