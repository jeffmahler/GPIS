'''
Helper to handle loading and partitioning patches data
Author: Jacky Liang
'''

import os
import IPython
import logging
import numpy as np

class PatchesDataLoader:

    def __init__(self, p_test, data_path, file_nums, features_set, metadata_set, labels_set, split_by_objs=True):
        self._p_test = p_test
        self._data_path = data_path
        self._file_nums = file_nums
        self._features_set = features_set
        self._metadata_set = metadata_set
        self._labels_set = labels_set
        
        has_obj_ids = 'obj_ids_' in metadata_set
        if split_by_objs and not has_obj_ids:
                logging.warning("Set to split train/test data by objs but obj_ids files are not given. Will not be splitting by objs.")
        self._split_by_objs = split_by_objs and has_obj_ids
        
        self._raw_data = {}
        self._partial_train_data = {}
        self._partial_raw_data = {}
        self.all_tr_data = None
        self.all_meta_data = None

        #loads all files
        self.load()
        
        #concat relevant matrices (for features and metas) into single matrices
        self.concat()
        
        #split train test on both features and labels
        self.tr, self.t, self.indices = self.split_train_test(self.all_tr_data, self._p_test)
        self.labels = {}
        for label_name in self._labels_set:
            self.labels[label_name] = {}
            self.labels[label_name]['tr'], self.labels[label_name]['t'], _ = self.split_train_test(self._raw_data[label_name], 
                                                                                                                                    self._p_test, self.indices)
    
    def get_partial_raw_data(self, feature_set):
        if feature_set not in self._partial_raw_data:
            data = self._raw_data[feature_set[0]]
            data = data.reshape(data.shape[0], -1)
            
            for feature in feature_set[1:]:
                a_data = self._raw_data[feature]
                data = np.c_[data, a_data.reshape(a_data.shape[0], -1)]
            
            self._partial_raw_data[feature_set] = self._concat_raw_data(feature_set)
        
        return self._partial_raw_data[feature_set]
    
    def get_partial_train_data(self, feature_set):
        if feature_set not in self._partial_train_data:
            partial_raw_data = self.get_partial_raw_data(feature_set)            
            tr, t, _ = self.split_train_test(partial_raw_data, self._p_test, self.indices)
            self.partial_data[feature_set] = {"tr":tr, "t":t}
        
        return self.partial_data[feature_set]

    def get_all_meta(self, type, post_indices):
        if len(post_indices) == 0:
            return np.array([])
        else:
            return np.take(self.all_meta_data, self.indices[type][post_indices])
        
    def load(self):
        all_files_prefixes = self._features_set.union(self._metadata_set.union(self._labels_set))
        all_files_num_pairs = PatchesDataLoader.get_patch_files_and_nums(self._data_path, all_files_prefixes, include_prefix=True)
        
        for filename, num, prefix in all_files_num_pairs:
            if int(num) in self._file_nums:
                loaded = np.load(os.path.join(self._data_path, filename))['arr_0']
                if prefix in self._raw_data:
                    self._raw_data[prefix] = np.r_[self._raw_data[name], loaded]
                else:   
                    self._raw_data[prefix] = loaded
                    
    def _concat_raw_data(self, prefixes):
        if len(prefixes) == 0:
            return np.array([])
        prefixes = list(prefixes)
        first_data = self._raw_data[prefixes[0]]
        all_data = first_data.reshape(first_data.shape[0], -1)
        for prefix in prefixes[1:]:
            data = self._raw_data[prefix]
            all_data = np.c_[all_data, data.reshape(data.shape[0], -1)]
        return all_data
    
    def concat(self):
        self.all_tr_data = self._concat_raw_data(self._features_set)
        self.all_meta_data = self._concat_raw_data(self._metadata_set)
        
    def split_train_test(self ,X, p_test, indices=None):
        n = X.shape[0]
        if indices is None:                
            if self._split_by_objs:
                obj_ids = np.unique(self._raw_data['obj_ids_'])
                np.random.shuffle(obj_ids)
                obj_split = int(len(obj_ids) * p_test)
                t_ids = set(obj_ids[:obj_split])
                
                tr_ind = []
                t_ind = []
                
                for i in range(len(self._raw_data['obj_ids_'])):
                    if self._raw_data['obj_ids_'][i] in t_ids:
                        t_ind.append(i)
                    else:
                        tr_ind.append(i)
                        
                np.random.shuffle(tr_ind)
                np.random.shuffle(t_ind)
                
                indices = {
                    'tr': np.array(tr_ind), 
                    't': np.array(t_ind)
                }
            else:
                indices = [i for i in range(n)]
                np.random.shuffle(indices)
                split = int(n*p_test)
                indices = {
                    'tr': np.array(indices[split:]),
                    't': np.array(indices[:split])
                }
                
        return np.take(X, indices['tr'], axis=0), np.take(X, indices['t'], axis=0), indices
        
    @staticmethod
    def get_patch_files_and_nums(input_path, prefixes, include_prefix=False):
        file_num_pairs = []
        
        for filename in os.listdir(input_path):
            found_prefix = None
            for prefix in prefixes:
                if filename.startswith(prefix):
                    found_prefix = prefix
                    break
                    
            if found_prefix is not None:
                _, ext = os.path.splitext(filename)
                num = filename[len(prefix):-len(ext)]
                
                if include_prefix:
                    file_num_pairs.append((filename, num, found_prefix))
                else:
                    file_num_pairs.append((filename, num))
                    
        return file_num_pairs
        
    @staticmethod
    def get_include_set_from_dict(dct):
        target_set = set()
        for name, use in dct.items():
            if use:
                target_set.add(name)
        return target_set