import argparse
import IPython
import logging
import os
import shutil
import sys
import time

def link_file(source_file, dest_file):
    ln_cmd = 'ln -fs %s %s' %(source_file, dest_file)
    os.system(ln_cmd)

datasets = ['amazon_picking_challenge', 'BigBIRD', 'Cat50_ModelDatabase', 'KIT', 'ModelNet40', 'SHREC14LSGTB', 'YCB']
formatted_datasets = ['amazon_picking_challenge', 'BigBIRD', 'Cat50_ModelDatabase', 'KIT', 'SHREC14LSGTB']
query_keys = ['BigBIRD_detergent', 'KIT_MelforBottle_800_tex', 'Cat50_ModelDatabase_bunker_boots']
formatted_dataset_source = 'dataset_1000_train' # source from this because it is linked to all of the files from the formatted sources
query_dataset_source = 'PriorsAll' # source from this because it is linked to all of the files from the formatted sources
files_to_ignore = ['all_keys.db'] # files to skip over creating a db for
features_folder = 'features'
extensions_to_link = ['.obj', '.sdf', '.ftr', '.json']
feature_extensions_to_link = ['.json']

logging.getLogger().setLevel(logging.INFO)

# parse args
argc = len(sys.argv)
if argc < 3:
    logging.error('Must supply a source directory and the master db directory to create the datasets')
    exit(0)
dataset_db_file_dir = sys.argv[1]
database_root_dir = sys.argv[2]

# loop through all candidate db files
dataset_files = os.listdir(dataset_db_file_dir)
for dataset_file in dataset_files:
    logging.info('Checking file %s' %(dataset_file))
    start_time = time.time()

    # check the valididty of the the candidate file
    dataset_file_root, file_ext = os.path.splitext(dataset_file)
    if file_ext != '.db' or dataset_file in files_to_ignore:
        continue

    # open the index file
    index_filename = os.path.join(dataset_db_file_dir, dataset_file)
    index_file = open(index_filename, 'r')

    # make a new dataset on the disk if it doesn't exist
    new_dataset_folder = os.path.join(database_root_dir, dataset_file_root)
    new_index_filename = os.path.join(new_dataset_folder, 'index.db')
    new_features_folder = os.path.join(new_dataset_folder, features_folder)
    if not os.path.exists(new_dataset_folder):
        os.makedirs(new_dataset_folder)
    if not os.path.exists(new_features_folder):
        os.makedirs(new_features_folder)
    shutil.copyfile(index_filename, new_index_filename)
    logging.debug('Copied index to: %s' %(new_index_filename))

    # link all keys
    logging.debug('Opened %s' %(index_filename))
    k = 0
    for line in index_file:
        # grab the keys
        tokens = line.split()
        full_key = tokens[0]

        # get the dataset and key, based on the source dataset
        dataset = ''
        source_key = full_key
        for ds in datasets:
            if full_key.find(ds) != -1:
                dataset = ds
        orig_key = full_key[len(dataset)+1:]
        if dataset not in formatted_datasets:
            source_key = orig_key
            
        logging.debug('Dataset: %s' %(dataset))
        logging.debug('Key: %s' %(source_key))

        # set up source dataset
        source_dataset = dataset
        grasp_source_dataset = dataset
        if dataset in formatted_datasets and source_key not in query_keys:
            grasp_source_dataset = formatted_dataset_source
        elif dataset in formatted_datasets and source_key in query_keys:
            grasp_source_dataset = query_dataset_source 
        logging.debug('Source dataset: %s' %(source_dataset))

        # link to source files
        for ext in extensions_to_link:
            source_filename = os.path.join(database_root_dir, source_dataset, orig_key + ext)
            if ext == '.json':
                source_filename = os.path.join(database_root_dir, grasp_source_dataset, source_key + ext)

            dest_filename = os.path.join(new_dataset_folder, full_key + ext)
            logging.debug('Source filename: %s' %(source_filename))
            logging.debug('Dest filename: %s' %(dest_filename))

            if not os.path.exists(dest_filename):
                os.symlink(source_filename, dest_filename)

        # link to features jsons only (the json tells it to look in the original spot)
        for ext in feature_extensions_to_link:
            source_filename = os.path.join(database_root_dir, source_dataset, features_folder, orig_key + ext)
            if ext == '.json':
                source_filename = os.path.join(database_root_dir, grasp_source_dataset, features_folder, source_key + ext)

            dest_filename = os.path.join(new_features_folder, full_key + ext)
            logging.debug('Source filename: %s' %(source_filename))
            logging.debug('Dest filename: %s' %(dest_filename))

            if not os.path.exists(dest_filename):
                os.symlink(source_filename, dest_filename)

        k=k+1

    end_time = time.time()
    logging.info('Linking dataset %s took %f' %(dataset_file, end_time - start_time))
    
