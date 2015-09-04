"""
Add symlinks from the database directories to the grasps in the extracted tars, since copying is unreasonable slow.

Author: Jeff Mahler
"""
import os
import shutil
import sys
import time

big_data_dir = '/home/brian/data'
#experiment_names = ['szrvyjkidb']
experiment_names = ['axlneljhga', 'dqqsbdpqgf', 'jeufzcrent', 'szrvyjkidb']
datasets = ['dataset_all_train', 'dataset_all_val', 'ModelNet40', 'NTU3D', 'YCB']
feature_subdir= 'features'
json_matches = ['*.json']
any_matches = ['*.json', '*[0-3]', '*[4-7]', '*[8-9]']

debug = False

def link_subdirs(source_data_dir, dest_data_dir, matches, debug=False):
    for match in matches:
        ln_cmd = 'ln -fs %s/%s %s/' %(source_data_dir, match, dest_data_dir)
        print ln_cmd
        os.system(ln_cmd)
    return

if __name__ == '__main__':
    k = 0
    big_data_subdirs = os.listdir(big_data_dir)

    for subdir in big_data_subdirs:
        # check if this is an experiment directory
        has_grasps = False
        for experiment_name in experiment_names:
            if subdir.find(experiment_name) != -1:
                has_grasps = True
            
        # add symlinks for all grasps in the experiment
        if has_grasps:
            print 'Linking experiment', subdir
            experiment_dir = os.path.join(big_data_dir, subdir, 'cm_out')
            experiment_subdirs = os.listdir(experiment_dir)
            dataset = ''

            # find the dataset for this experiment
            for s in experiment_subdirs:
                for d in datasets:
                    if s == d:
                        dataset = d

            # get the dataset dir
            experiment_data_dir = os.path.join(experiment_dir, dataset)
            dataset_dir = os.path.join(big_data_dir, dataset)
            if debug:
                print 'Dataset', dataset_dir
                print 'Experiment data dir', experiment_data_dir

            start_time = time.time()

            # link the subdirectories
            link_subdirs(experiment_data_dir, dataset_dir, json_matches, debug)

            # link other datasets
            other_datasets = []
            if dataset == 'dataset_all_train' or dataset == 'dataset_all_val':
                other_datasets = ['dataset_10_val', 'dataset_100_val', 'dataset_1000_val']
                other_datasets.extend(['dataset_10_train', 'dataset_100_train', 'dataset_1000_train'])

            # link all against the other datasets (yes there will be too many grasps in the small ones but fuck it)
            for od in other_datasets:
                other_dataset_dir = os.path.join(big_data_dir, od) 
                link_subdirs(experiment_data_dir, other_dataset_dir, json_matches, debug)

            experiment_features_dir = os.path.join(experiment_data_dir, feature_subdir)
            dataset_features_dir = os.path.join(dataset_dir, feature_subdir)

            if debug:
                print 'Experiment features', experiment_features_dir
                print 'Dataset features', dataset_features_dir

            # create features subdir if not yet in existence
            if os.path.exists(dataset_features_dir) and not os.path.isdir(dataset_features_dir): # remove old links
                os.remove(dataset_features_dir) # REMOVE ME!
            if not os.path.exists(dataset_features_dir):
                os.makedirs(dataset_features_dir)

            # link the subdirectories
            link_subdirs(experiment_features_dir, dataset_features_dir, any_matches, debug)
            
            # link other datasets (TODO: this is swapped due to a nesting bug)
            other_datasets = []
            if dataset == 'dataset_all_train' or dataset == 'dataset_all_val':
                other_datasets = ['dataset_10_val', 'dataset_100_val', 'dataset_1000_val']
                other_datasets.extend(['dataset_10_train', 'dataset_100_train', 'dataset_1000_train'])

            for od in other_datasets:
                other_dataset_features_dir = os.path.join(big_data_dir, od, feature_subdir) 
                if debug:
                    print 'Other feature dir', other_dataset_features_dir

                # create features subdir if not yet in existence
                if os.path.exists(other_dataset_features_dir) and not os.path.isdir(other_dataset_features_dir):
                    os.remove(other_dataset_features_dir) # REMOVE ME!
                if not os.path.exists(other_dataset_features_dir):
                    os.makedirs(other_dataset_features_dir)

                link_subdirs(experiment_features_dir, other_dataset_features_dir, json_matches, debug)
                    
            end_time = time.time()
            print 'Linking took %f sec' %(end_time - start_time)

#            time.sleep(2)
            k = k+1

#            exit(0)
