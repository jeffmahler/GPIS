"""
Script to generate a new database of mesh and SDF models for grasping research.
Cleans the meshes, rescales, converts to obj, converts to sdf, extracts features, etc
TODO: Use stable poses and extract mesh categories

Author: Jeff Mahler
"""
import IPython
import logging
import matplotlib.pyplot as plt
import os
import shutil
import sys
sys.path.append('src/grasp_selection/feature_vectors')

import database as db
import experiment_config as ec
from mesh_file import MeshFile
from mesh_cleaner import MeshCleaner
import mesh_database as mdb
import mesh_processor as mp
import obj_file
import stable_poses

# Global variables for conversion
class DatasetConfig:
    """ Struct to store relevant info about conversion for a particular dataset """
    def __init__(self, name, extension, name_filter = '', fix_names=False, scale=1.0, synthetic=True, cat_db=None):
        self.name = name               # name of dataset
        self.extension = extension     # default model extension for dataset
        self.name_filter = name_filter # template to search for a substring of the name
        self.fix_names = fix_names     # whether or not to rename the mesh
        self.scale = scale             # how much to scale the mesh to get to meters (because some objects are in mm instead of m)
        self.synthetic = synthetic     # whether or not the meshes are synthetic
        self.cat_db_ = cat_db          # dictionary of object categories
        self.count = 0                 # counter to determine how many models per dataset

    def set_count(self, count):
        """ Sets a new model count"""
        self.count = count

    def category(self, key):
        """ Returns the category for a given key """
        if self.cat_db_:
            return self.cat_db_.object_category_for_key(key)
        return ''

# Global array of all datasets and params
DATASETS = [
    DatasetConfig(name='amazon_picking_challenge', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=mdb.BerkeleyObjectDatabase()),
    DatasetConfig(name='autodesk', extension='.off', synthetic=True),
#    DatasetConfig(name='Archive3D', extension='.3DS'),
    DatasetConfig(name='BigBIRD', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=mdb.BerkeleyObjectDatabase()),
    DatasetConfig(name='Cat50_ModelDatabase', extension='.obj', cat_db=mdb.Cat50ObjectDatabase('/mnt/terastation/shape_data/Cat50_ModelDatabase')),
    DatasetConfig(name='KIT', extension='.obj', name_filter='800_tex', scale=1e-3, synthetic=False),
    DatasetConfig(name='ModelNet40', extension='.off', cat_db=mdb.ModelNet40ObjectDatabase('/mnt/terastation/shape_data/MASTER_DB_v2/ModelNet40/index.db')),
    DatasetConfig(name='NTU3D', extension='.obj'),
    DatasetConfig(name='PrincetonShapeBenchmark', extension='.off'),
    DatasetConfig(name='SHREC14LSGTB', extension='.off', cat_db=mdb.SHRECObjectDatabase('/mnt/terastation/shape_data/SHREC14LSGTB/SHREC14LSGTB.cla')),
    DatasetConfig(name='YCB', extension='.obj', name_filter='poisson_texture_mapped', fix_names=True, synthetic=False, cat_db=mdb.BerkeleyObjectDatabase())
]

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
    dataset = config['gen_dataset']
    shape_db_root_folder = config['shape_data_dir']
    dest_root_folder = config['database_dir']

    # numeric params
    min_dim = config['min_dim']
    dim = config['sdf_dim']
    padding = config['sdf_padding']
    min_prob = config['stp_min_prob']
    density = config['density']
    gripper_size = config['gripper_size']

    dataset_start = 0
    # get indices of dataset configurations
    dataset_names = [d.name for d in DATASETS]
    print dataset_names
    if dataset != 'all' and dataset in dataset_names:
        # get the indices of the dataset
        dataset_inds = [dataset_names.index(dataset)]
    # get all indices
    elif dataset == 'all':
        dataset_inds = range(len(DATASETS))
        dataset_start = config['gen_dataset_start']
        if dataset_start == None:
            dataset_start = 0
    else:
        raise Exception('Invalid dataset!')

    # create dest folder if doesn't exist
    if not os.path.exists(dest_root_folder):
        os.mkdir(dest_root_folder)

    # create list to store all exceptions
    exceptions = []

    # open up the database
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config, access_level=db.READ_WRITE_ACCESS)
    
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
        logging.info('Dataset scale=%f' %(dataset.scale))

        # create tmp dir for storing output of converter binaries
        target_dir = os.path.join(dest_root_folder, dataset.name)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
            os.system('chmod a+rwx %s' %(target_dir) )

        # create dest
        database.create_dataset(dataset.name)
        dataset_handle = database.dataset(dataset.name)

        # walk through the shape directory structure
        for root, sub_folders, files in os.walk(data_folder):
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
                        category = dataset.category(root)
                    else:
                        category = dataset.category(file_root)

                    if dataset.fix_names:
                        target_filename = os.path.join(target_dir, category)
                    else:
                        target_filename = os.path.join(target_dir, file_root)

                    try:
                        # convert to obj
                        obj_filename = '%s.obj' %(fullpath_file_root)
                        decimation_script = os.path.join(config['root_dir'], 'scripts/decimation2.mlx')
                        mesh = MeshFile.extract_mesh(filename, obj_filename, decimation_script)

                        # clean up mesh triangles
                        mesh_processor = mp.MeshProcessor(mesh, obj_filename)
                        mesh_processor.clean()
                    
                        # scale mesh to meters
                        mesh_processor.rescale_vertices(dataset.scale, rescaling_type=MeshCleaner.RescalingTypeAbsolute)

                        # rescale synthetic meshes to fit within the gripper
                        if dataset.synthetic:
                            mesh_processor.rescale_vertices(gripper_size, rescaling_type=MeshCleaner.RescalingTypeDiag)

                        # get convex pieces (NOT WORKING WELL...)                        
                        #convex_pieces = mesh_processor.convex_pieces(config['cvx_decomp'])

                        # set metadata (mass + category)
                        mesh = mesh_processor.mesh
                        mass = mesh.mass
                        if mass < config['mass_thresh']:
                            mass = config['default_mass']

                        # get the extra info
                        sdf = mesh_processor.convert_sdf(dim, padding)
                        stable_poses = mesh_processor.stable_poses(min_prob)
                        shot_features = mesh_processor.extract_shot()

                        # write to database
                        if dataset.fix_names:
                            key = category
                        else:
                            key = file_root
                        dataset_handle.create_graspable(key, mesh, sdf, shot_features, stable_poses, category=category, mass=mass)

                        # TODO: remove, for testing purposes only
                        #g = dataset_handle.read_datum(key)
                        #g.sdf.scatter()
                        #plt.show()

                    except Exception as e:
                        exceptions.append('Dataset: %s,  Model: %s, Exception: %s' % (dataset.name, filename, str(e))) 
                
    # print all exceptions
    exceptions_filename = os.path.join(dest_root_folder, 'exceptions.txt')
    out_exceptions = open(exceptions_filename, 'w')
    for exception in exceptions:
        out_exceptions.write('%s\n' %(exception))
    out_exceptions.close()

    IPython.embed()
