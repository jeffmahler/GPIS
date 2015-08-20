"""
Script to generate a new database of mesh and SDF models for grasping research.
Cleans the meshes, rescales, converts to obj, converts to sdf, extracts features, etc
TODO: Use stable poses and extract mesh categories

Author: Jeff Mahler
"""
import IPython
import logging
import os
import shutil
import sys
import stable_poses

import experiment_config as ec
from mesh_file import MeshFile
from mesh_cleaner import MeshCleaner

# Global variables for conversion
class DatasetConfig:
    """ Struct to store relevant info about conversion for a particular dataset """
    def __init__(self, name, extension, name_filter = '', fix_names=False, scale=1.0, synthetic=True):
        self.name = name               # name of dataset
        self.extension = extension     # default model extension for dataset
        self.name_filter = name_filter # template to search for a substring of the name
        self.fix_names = fix_names     # whether or not to rename the mesh
        self.scale = scale             # how much to scale the mesh to get to meters (because some objects are in mm instead of m)
        self.synthetic = synthetic     # whether or not the meshes are synthetic
        self.count = 0                 # counter to determine how many models per dataset

    def set_count(self, count):
        """ Sets a new model count"""
        self.count = count

# Global array of all datasets and params
DATASETS = [
    DatasetConfig(name='amazon_picking_challenge', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False),
#    DatasetConfig(name='Archive3D', extension='.3DS'),
    DatasetConfig(name='BigBIRD', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False),
    DatasetConfig(name='Cat50_ModelDatabase', extension='.obj'),
    DatasetConfig(name='KIT', extension='.obj', name_filter='800_tex', scale=1e-3, synthetic=False),
    DatasetConfig(name='ModelNet40', extension='.off'),
    DatasetConfig(name='NTU3D', extension='.obj'),
    DatasetConfig(name='PrincetonShapeBenchmark', extension='.off'),
    DatasetConfig(name='SHREC14LSGTB', extension='.off'),
    DatasetConfig(name='YCB', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False)
]

# Invalid categories
INVALID_CATEGORIES = ['mnt', 'terastation', 'shape_data', 'textured_meshes', 'processed', '']
INVALID_CATEGORIES.extend([d.name for d in DATASETS])

def get_category_from_directory(file_dir):
    """ Grabs the category from the filename """
    head, tail = os.path.split(file_dir)
    while head != '/' and tail in INVALID_CATEGORIES:
        head, tail = os.path.split(head)
    return tail

# read in params
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/generate_database.yaml')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)

    # filesystem params
    dataset = config['dataset']
    shape_db_root_folder = config['shape_data_root_folder']
    dest_root_folder = config['destination_root_folder']

    # numeric params
    min_dim = config['min_dim']
    dim = config['sdf_dim']
    padding = config['sdf_padding']
    density = config['density']
    gripper_size = config['gripper_size']

    dataset_start = 0
    # get indices of dataset configurations
    dataset_names = [d.name for d in DATASETS]
    if dataset != 'all' and dataset in dataset_names:
        # get the indices of the dataset
        dataset_inds = [dataset_names.index(dataset)]
    # get all indices
    elif dataset == 'all':
        dataset_inds = range(len(DATASETS))
        dataset_start = config['dataset_start']
        if dataset_start == None:
            dataset_start = 0
    else:
        raise Exception('Invalid dataset!')

    # create dest folder if doesn't exist
    if not os.path.exists(dest_root_folder):
        os.mkdir(dest_root_folder)

    # create list to store all exceptions
    exceptions = []

    # loop through datasets and cleanup
    for j in range(dataset_start, len(dataset_inds)):
        # clear filenames list
        all_filenames = []

        # get next dataset
        i = dataset_inds[j]
        dataset = DATASETS[i]
        data_folder = os.path.join(shape_db_root_folder, dataset.name)
        data_list = os.listdir(data_folder)
        logging.info('Processing dataset %s' %(dataset.name))

        # file extensions
        logging.info('Dataset scale=%f' %(dataset.scale))

        # create target dir name
        target_dir = os.path.join(dest_root_folder, dataset.name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            os.system('chmod a+rwx %s' %(target_dir) )

        # walk through the shape directory structure
        for root, sub_folders, files in os.walk(data_folder):
            category = get_category_from_directory(root)

            # for each file in the subdir, look for valid models
            for f in files:
                # create file name
                filename = os.path.join(root, f)
                file_root, file_ext = os.path.splitext(f)
                fullpath_file_root = os.path.join(root, file_root)

                # grab the models with a given format and filtered by the given filter (fresh clean for each file)
                if file_ext == dataset.extension and filename.find(dataset.name_filter) > -1 and filename.find('clean') == -1:
                    # optionally rename by category
                    if dataset.fix_names:
                        target_filename = os.path.join(target_dir, category)
                    else:
                        target_filename = os.path.join(target_dir, file_root)

                    try:
                        # convert to obj
                        obj_filename = '%s.obj' %(fullpath_file_root)
                        smoothing_script = ''
                        m = MeshFile.extract_mesh(filename, obj_filename, smoothing_script)

                        # clean up mesh triangles
                        mesh_cleaner = MeshCleaner(m)
                        mesh_cleaner.remove_bad_tris()
                        mesh_cleaner.remove_unreferenced_vertices()
                        mesh_cleaner.standardize_pose()
                    
                        # scale mesh to meters
                        mesh_cleaner.rescale_vertices(dataset.scale, rescaling_type=MeshCleaner.RescalingTypeAbsolute)

                        # rescale synthetic meshes to fit within the gripper
                        if dataset.synthetic:
                            mesh_cleaner.rescale_vertices(gripper_size, rescaling_type=MeshCleaner.RescalingTypeMin)

                        # set metadata (mass + category)
                        m.density = density
                        m.category = category
                   
                        # save obj, sdf, etc.
                        MeshFile.write_obj(m, target_filename)
                        MeshFile.write_sdf(m, target_filename, dim, padding)
                        MeshFile.write_stp(m, target_filename)
                        MeshFile.write_json(m, target_filename)
                        MeshFile.write_shot(m, target_filename)

                        # add filename to master list
                        if dataset.fix_names:
                            all_filenames.append(category)
                        else:
                            all_filenames.append(file_root)
                    except Exception as e:
                        exceptions.append('Dataset: %s,  Model: %s, Exception: %s' % (dataset.name, filename, str(e))) 
                
        # save filenames to master list (index.db)
        master_list_filename = os.path.join(target_dir, 'index.db')
        out_f = open(master_list_filename, 'w')
        for filename in all_filenames:
            out_f.write('%s\n' %(filename))
        out_f.close()
        os.system('chmod a+rwx %s' %(master_list_filename) )

    # print all exceptions
    exceptions_filename = os.path.join(dest_root_folder, 'exceptions.txt')
    out_exceptions = open(exceptions_filename, 'w')
    for exception in exceptions:
        out_exceptions.write('%s\n' %(exception))
    out_exceptions.close()
