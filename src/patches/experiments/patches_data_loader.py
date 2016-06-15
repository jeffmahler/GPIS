import os
import numpy as np

IDS = (
    ('ids','obj_ids'),
)

FEATURES = (
    ('moment_arms', 'moment_arms'),
    ('patch_ori', 'patch_orientation'),
    ('w1_proj', 'w1_projection_window'),
    ('w2_proj', 'w2_projection_window'),
    ('w1_normal', 'w1_approx_normals'),
    ('w2_normal', 'w2_approx_normals')
)

LABELS = (
#    ('pfc_20', 'pfc_f_0.200000_tg_0.010000_rg_0.200000_to_0.010000_ro_0.200000'),
#    ('pfc_10', 'pfc_f_0.100000_tg_0.005000_rg_0.100000_to_0.005000_ro_0.100000'),
#    ('pfc_5', 'pfc_f_0.050000_tg_0.002500_rg_0.050000_to_0.002500_ro_0.050000'),
    ('ferrari', 'ferrari_canny_L1'),
    ('fc', 'force_closure')
)

class PatchesDataLoader:

    def __init__(self, p_test, data_path, file_nums=[0], by_objs=True):
        self._file_nums = file_nums
        self._data_path = data_path
        self._p_test = p_test
        self._by_objs = by_objs
        
        self.partial_data = {}
        
        self.load()
        self.concat()
        self.tr, self.t, self.indices = self.split_train_test(self.all_tr_data, self._p_test)
        self.labels = {}
        for label in LABELS:
            label_type = label[0]
            self.labels[label_type] = {}
            self.labels[label_type]['tr'], self.labels[label_type]['t'], _ = self.split_train_test(self._raw_data[label_type], self._p_test, self.indices)
    
    def get_partial_train_data(self, feature_set):
        if feature_set not in self.partial_data:
            data = self._raw_data[feature_set[0]]
            tr_data = data.reshape(data.shape[0], -1)
            for feature in feature_set[1:]:
                data = self._raw_data[feature]
                tr_data = np.c_[tr_data, data.reshape(data.shape[0], -1)]
            
            tr, t, _ = self.split_train_test(tr_data, self._p_test, self.indices)
            self.partial_data[feature_set] = {"tr":tr, "t":t}
        
        return self.partial_data[feature_set]

    def load(self):
        file_num = self._file_nums[0]
        self._raw_data = {}
        
        all_files = [file for file in os.listdir(self._data_path) if file.endswith('.npz')]
        
        def find_name(file):
            for names in IDS + FEATURES + LABELS:
                if file.startswith(names[1]):
                    return names[0]
            return None
        
        for file in all_files:
            name = find_name(file)
            loaded = np.load(os.path.join(self._data_path, file))['arr_0']
            if name is not None:
                if name in self._raw_data:
                    self._raw_data[name] = np.r_[self._raw_data[name], loaded]
                else:   
                    self._raw_data[name] = loaded
                    
    def concat(self):
        first_data = self._raw_data[FEATURES[0][0]]
        self.all_tr_data = first_data.reshape(first_data.shape[0], -1)
        for feature in FEATURES[1:]:
            data = self._raw_data[feature[0]]
            self.all_tr_data = np.c_[self.all_tr_data, data.reshape(data.shape[0], -1)]
        
    def split_train_test(self ,X, p_test, indices=None):
        n = X.shape[0]
        if indices is None:
            if self._by_objs:
                obj_ids = np.unique(self._raw_data['ids'])
                np.random.shuffle(obj_ids)
                obj_split = int(len(obj_ids) * p_test)
                t_ids = set(obj_ids[:obj_split])
                
                tr_ind = []
                t_ind = []
                
                for i in range(len(self._raw_data['ids'])):
                    if self._raw_data['ids'][i] in t_ids:
                        t_ind.append(i)
                    else:
                        tr_ind.append(i)
                        
                np.random.shuffle(tr_ind)
                np.random.shuffle(t_ind)
                
                indices = {
                    'tr': tr_ind, 
                    't': t_ind
                }
                
            else:
                indices = [i for i in range(n)]
                np.random.shuffle(indices)
                split = int(n*p_test)
                indices = {
                    'tr': indices[split:],
                    't': indices[:split]
                }
                
        return np.take(X, indices['tr'], axis=0), np.take(X, indices['t'], axis=0), indices