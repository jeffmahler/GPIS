from abc import ABCMeta, abstractmethod

import datetime as dt
import h5py
import logging
import numbers
import numpy as np
import os
import sys
import time

import experiment_config as ec
import hdf5_factory as hfact
import json_serialization as jsons
import grasp
import graspable_object as go
import obj_file
import sdf_file
import feature_file
import stp_file

import IPython

INDEX_FILE = 'index.db'
HDF5_EXT = '.hdf5'
OBJ_EXT = '.obj'
STL_EXT = '.stl'
SDF_EXT = '.sdf'

READ_ONLY_ACCESS = 'READ_ONLY'
READ_WRITE_ACCESS = 'READ_WRITE'
WRITE_ACCESS = 'WRITE'

# Keys for easy lookups in HDF5 databases
OBJECTS_KEY = 'objects'
MESH_KEY = 'mesh'
SDF_KEY = 'sdf'
GRASPS_KEY = 'grasps'
GRIPPERS_KEY = 'grippers'
NUM_GRASPS_KEY = 'num_grasps'
LOCAL_FEATURES_KEY = 'local_features'
GLOBAL_FEATURES_KEY = 'global_features'
SHOT_FEATURES_KEY = 'shot'
RENDERED_IMAGES_KEY = 'rendered_images'
SENSOR_DATA_KEY = 'sensor_data'
STP_KEY = 'stable_poses'
CATEGORY_KEY = 'category'
MASS_KEY = 'mass'

CREATION_KEY = 'time_created'
DATASETS_KEY = 'datasets'

def generate_metric_tag(root, config):
    sig_mu = config['sigma_mu']
    sig_t_g = config['sigma_trans_grasp']
    sig_r_g = config['sigma_rot_grasp']
    sig_t_o = config['sigma_trans_obj']
    sig_r_o = config['sigma_rot_obj']

    if isinstance(sig_t_g, np.ndarray):
        sig_t_g = sig_t_g[0,0]

    if isinstance(sig_r_g, np.ndarray):
        sig_r_g = sig_r_g[0,0]

    if isinstance(sig_t_o, np.ndarray):
        sig_t_o = sig_t_o[0,0]

    if isinstance(sig_r_o, np.ndarray):
        sig_r_o = sig_r_o[0,0]

    tag = '%s_f_%f_tg_%f_rg_%f_to_%f_ro_%f' %(root, sig_mu, sig_t_g, sig_r_g,
                                              sig_t_o, sig_r_o)
    return tag

class Database(object):
    """ Abstract class for databases. Main purpose is to wrap individual datasets """
    __metaclass__ = ABCMeta

    def __init__(self, config, access_level=READ_ONLY_ACCESS):
        self.access_level_ = access_level
        self._read_config(config)

    def _read_config(self, config):
        """ Parse common items from the configuation file """
        self.database_root_dir_ = config['database_dir']
        self.dataset_names_ = config['datasets']
        if self.dataset_names_ is None:
            self.dataset_names_ = []

    @property
    def access_level(self):
        return self.access_level_

    @property
    def dataset_names(self):
        return self.dataset_names_

class Hdf5Database(Database):
    def __init__(self, database_filename, config, access_level=READ_ONLY_ACCESS):
        Database.__init__(self, config, access_level)
        self.database_filename_ = database_filename
        if not self.database_filename_.endswith(HDF5_EXT):
            raise ValueError('Must provide HDF5 database')

        self.config_ = config
        self.dataset_names_ = [] # override default behavior
        self._parse_config(config)

        self._load_database()
        self._load_datasets(config)

    def _parse_config(self, config):
        """ Parse the configuation file """
        self.database_cache_dir_ = config['database_cache_dir']
        self.dataset_names_ = []
        if config[DATASETS_KEY]:
            self.dataset_names_ = config[DATASETS_KEY].keys()

    def _create_new_db(self):
        """ Creates a new database """
        self.data_ = h5py.File(self.database_filename_, 'w')

        dt_now = dt.datetime.now()
        creation_stamp = '%s-%s-%s-%sh-%sm-%ss' %(dt_now.month, dt_now.day, dt_now.year, dt_now.hour, dt_now.minute, dt_now.second) 
        self.data_.attrs[CREATION_KEY] = creation_stamp
        self.data_.create_group(DATASETS_KEY)

    def _load_database(self):
        """ Loads in the HDF5 file """
        if self.access_level == READ_ONLY_ACCESS:
            self.data_ = h5py.File(self.database_filename_, 'r')
        elif self.access_level == READ_WRITE_ACCESS:
            if os.path.exists(self.database_filename_):
                self.data_ = h5py.File(self.database_filename_, 'r+')
            else:
                self._create_new_db()
        elif self.access_level == WRITE_ACCESS:
            self._create_new_db()

    def _load_datasets(self, config):
        """ Load in the datasets """
        self.datasets_ = []
        for dataset_name in self.dataset_names_:
            if dataset_name not in self.data_[DATASETS_KEY].keys():
                logging.warning('Dataset %s not in database' %(dataset_name))
            else:
                dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
                self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name], config,
                                                  cache_dir=dataset_cache_dir))

    @property
    def datasets(self):
        return self.datasets_

    def dataset(self, dataset_name):
        """ Returns handles to individual datasets """
        if self.datasets is None or dataset_name not in self.dataset_names_:
            return None
        for dataset in self.datasets_:
            if dataset.name == dataset_name:
                return dataset

    def close(self):
        """ Close the HDF5 file """
        self.data_.close()

    def __getitem__(self, dataset_name):
        """ Dataset name indexing """
        return self.dataset(dataset_name)
        
    # New dataset creation / modification functions
    def create_dataset(self, dataset_name, obj_keys=[]):
        """ Create dataset with obj keys"""
        if dataset_name in self.data_[DATASETS_KEY].keys():
            logging.warning('Dataset %s already exists. Cannot overwrite' %(dataset_name))
            return self.datasets_[self.data_[DATASETS_KEY].keys().index(dataset_name)]
        self.data_[DATASETS_KEY].create_group(dataset_name)
        self.data_[DATASETS_KEY][dataset_name].create_group(OBJECTS_KEY)
        for obj_key in obj_keys:
            self.data_[DATASETS_KEY][dataset_name][OBJECTS_KEY].create_group(obj_key)

        dataset_cache_dir = os.path.join(self.database_cache_dir_, dataset_name)
        self.dataset_names_.append(dataset_name)
        self.datasets_.append(Hdf5Dataset(dataset_name, self.data_[DATASETS_KEY][dataset_name], self.config_,
                                          cache_dir=dataset_cache_dir))
        return self.datasets_[-1] # return the dataset
        
    def create_linked_dataset(self, dataset_name, graspable_list, nearest_neighbors):
        """ Creates a new dataset that links to objects physically stored as part of another dataset """
        raise NotImplementedError()

class Dataset(object):
    pass

class Hdf5Dataset(Dataset):
    def __init__(self, dataset_name, data, config, cache_dir=''):
        self.dataset_name_ = dataset_name
        self.data_ = data
        self.object_keys_ = None
        self.start_index_ = 0
        self.end_index_ = len(self.object_keys)
        self.cache_dir_ = cache_dir
        if not os.path.exists(self.cache_dir_):
            os.mkdir(self.cache_dir_)

        self._parse_config(config)

    def _parse_config(self, config):
        if config['datasets'] and self.dataset_name_ in config['datasets']:
            self.start_index_ = config['datasets'][self.dataset_name_]['start_index']
            self.end_index_ = config['datasets'][self.dataset_name_]['end_index']

    @property
    def name(self):
        return self.dataset_name_

    @property
    def dataset_root_dir(self):
        return self.dataset_root_dir_

    @property
    def objects(self):
        return self.data_[OBJECTS_KEY]

    @property
    def object_keys(self):
        if not self.object_keys_:
            self.object_keys_ = self.objects.keys()
        return self.object_keys_

    # easy data accessors
    def object(self, key):
        return self.objects[key]

    def sdf_data(self, key):
        return self.objects[key][SDF_KEY]

    def mesh_data(self, key):
        return self.objects[key][MESH_KEY]

    def grasp_data(self, key, gripper=None):
        if gripper:
            return self.objects[key][GRASPS_KEY][gripper]
        return self.objects[key][GRASPS_KEY]

    def local_feature_data(self, key):
        return self.objects[key][LOCAL_FEATURES_KEY]

    def shot_feature_data(self, key):
        return self.local_feature_data(key)[SHOT_FEATURES_KEY]

    def stable_pose_data(self, key, stable_pose_id=None):
        if stable_pose_id is not None:
            self.objects[key][STP_KEY][stable_pose_id]
        return self.objects[key][STP_KEY]

    def category(self, key):
        return self.objects[key].attrs[CATEGORY_KEY]

    def rendered_image_data(self, key, stable_pose_id=None, image_type=None):
        if stable_pose_id is not None and image_type is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY][image_type]
        elif stable_pose_id is not None:
            return self.stable_pose_data(key)[stable_pose_id][RENDERED_IMAGES_KEY]
        elif image_type is not None:
            return self.object(key)[RENDERED_IMAGES_KEY][image_type]
        return self.object(key)[RENDERED_IMAGES_KEY]

    # iterators
    def __getitem__(self, index):
        """ Index a particular object in the dataset """
        if isinstance(index, numbers.Number):
            if index < 0 or index >= len(self.object_keys):
                raise ValueError('Index out of bounds. Dataset contains %d objects' %(len(self.object_keys)))
            obj = self.graspable(self.object_keys[index])
            return obj
        elif isinstance(index, (str, unicode)):
            obj = self.graspable(index)
            return obj

    def __iter__(self):
        """ Generate iterator """
        self.iter_count_ = self.start_index_ # NOT THREAD SAFE!
        return self
    
    def next(self):
        """ Read the next object file in the list """
        if self.iter_count_ >= len(self.object_keys) or self.iter_count_ >= self.end_index_:
            raise StopIteration
        else:
            logging.info('Returning datum %s' %(self.object_keys[self.iter_count_]))
            if True:#try:
                 obj = self.graspable(self.object_keys[self.iter_count_])    
            #except:
            #    logging.warning('Error reading %s. Skipping' %(self.object_keys[self.iter_count_]))
            #    self.iter_count_ = self.iter_count_ + 1
            #    return self.next()

            self.iter_count_ = self.iter_count_ + 1
            return obj

    # direct reading / writing
    def graspable(self, key):
        """Read in the GraspableObject3D corresponding to given key."""
        if key not in self.object_keys:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))

        # read in data (need new interfaces for this....
        sdf = hfact.Hdf5ObjectFactory.sdf_3d(self.sdf_data(key))
        mesh = hfact.Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        features = None
        if SHOT_FEATURES_KEY in self.local_feature_data(key).keys():
            features = hfact.Hdf5ObjectFactory.local_features(self.shot_feature_data(key))
        return go.GraspableObject3D(sdf, mesh=mesh, features=features, key=key, model_name='')

    def create_graspable(self, key, mesh=None, sdf=None, shot_features=None, stable_poses=None, category='', mass=1.0):
        """ Creates a graspable object """
        # create object tree
        self.objects.create_group(key)
        self.object(key).create_group(MESH_KEY)
        self.object(key).create_group(SDF_KEY)
        self.object(key).create_group(STP_KEY)
        self.object(key).create_group(LOCAL_FEATURES_KEY)
        self.object(key).create_group(GLOBAL_FEATURES_KEY)
        self.object(key).create_group(RENDERED_IMAGES_KEY)
        self.object(key).create_group(SENSOR_DATA_KEY)
        self.object(key).create_group(GRASPS_KEY)

        # add the different pieces if provided
        if sdf:
            hfact.Hdf5ObjectFactory.write_sdf_3d(sdf, self.sdf_data(key))
        if mesh:
            hfact.Hdf5ObjectFactory.write_mesh_3d(mesh, self.mesh_data(key))
        if shot_features:
            hfact.Hdf5ObjectFactory.write_shot_features(shot_features, self.local_feature_data(key))
        if stable_poses:
            hfact.Hdf5ObjectFactory.write_stable_poses(stable_poses, self.stable_pose_data(key))

        # add the attributes
        self.object(key).attrs.create(CATEGORY_KEY, category)
        self.object(key).attrs.create(MASS_KEY, mass)
        self.object_keys_.append(key)

    def update_mesh(self, key, mesh):
        """ Updates the mesh for the given key """
        hfact.Hdf5ObjectFactory.write_mesh_3d(mesh, self.mesh_data(key), force_overwrite=True)

    def obj_mesh_filename(self, key, scale=1.0):
        """ Writes an obj file in the database "cache"  directory and returns the path to the file """
        mesh = hfact.Hdf5ObjectFactory.mesh_3d(self.mesh_data(key))
        mesh.rescale(scale)
        obj_filename = os.path.join(self.cache_dir_, key + OBJ_EXT)
        of = obj_file.ObjFile(obj_filename)
        of.write(mesh)
        return obj_filename

    def stl_mesh_filename(self, key, scale=1.0):
        """ Writes an stl file in the database "cache"  directory and returns the path to the file """
        obj_filename = self.obj_mesh_filename(key, scale=scale)
        stl_filename = os.path.join(self.cache_dir_, key + STL_EXT)
        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(obj_filename, stl_filename)
        os.system(meshlabserver_cmd)
        return stl_filename

    # grasp data
    # TODO: implement handling of stable poses and tasks
    def grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable, optionally associated with the given stable pose """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return []
        return hfact.Hdf5ObjectFactory.grasps(self.grasp_data(key, gripper))

    def sorted_grasps(self, key, metric, gripper='pr2', stable_pose_id=None):
        """ Returns the list of grasps for the given graspable sorted by decreasing quality according to the given metric """
        grasps = self.grasps(key, gripper=gripper, stable_pose_id=stable_pose_id)
        if len(grasps) == 0:
            return []
        
        grasp_metrics = self.grasp_metrics(key, grasps, gripper=gripper, stable_pose_id=stable_pose_id)
        if metric not in grasp_metrics[grasp_metrics.keys()[0]].keys():
            raise ValueError('Metric %s not recognized' %(metric))

        grasps_and_metrics = [(g, grasp_metrics[g.grasp_id][metric]) for g in grasps]
        grasps_and_metrics.sort(key=lambda x: x[1], reverse=True)
        sorted_grasps = [g[0] for g in grasps_and_metrics]
        sorted_metrics = [g[1] for g in grasps_and_metrics]
        return sorted_grasps, sorted_metrics

    def delete_grasps(self, key, gripper='pr2', stable_pose_id=None):
        """ Deletes a set of grasps associated with the given gripper """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Nothing to delete' %(gripper))
            return False
        del self.grasp_data(key)[gripper]
        return True

    def store_grasps(self, key, grasps, gripper='pr2', stable_pose_id=None, force_overwrite=False):
        """ Associates grasps in list |grasps| with the given object. Optionally associates the grasps with a single stable pose """
        # create group for gripper if necessary
        if gripper not in self.grasp_data(key).keys():
            self.grasp_data(key).create_group(gripper)
            self.grasp_data(key, gripper).attrs.create(NUM_GRASPS_KEY, 0)

        # store each grasp in the database
        return hfact.Hdf5ObjectFactory.write_grasps(grasps, self.grasp_data(key, gripper), force_overwrite)

    def grasp_metrics(self, key, grasps, gripper='pr2', stable_pose_id=None, task_id=None):
        """ Returns a list of grasp metric dictionaries fot the list grasps provided to the database """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return {}
        return hfact.Hdf5ObjectFactory.grasp_metrics(grasps, self.grasp_data(key, gripper))

    def store_grasp_metrics(self, key, grasp_metric_dict, gripper='pr2', stable_pose_id=None, task_id=None, force_overwrite=False):
        """ Add grasp metrics in list |metrics| to the data associated with |grasps| """
        return hfact.Hdf5ObjectFactory.write_grasp_metrics(grasp_metric_dict, self.grasp_data(key, gripper), force_overwrite)

    def grasp_features(self, key, grasps, gripper='pr2', stable_pose_id=None, task_id=None, feature_names=None):
        """ Returns the list of grasps for the given graspable, optionally associated with the given stable pose """
        if gripper not in self.grasp_data(key).keys():
            logging.warning('Gripper type %s not found. Returning empty list' %(gripper))
            return {}
        return hfact.Hdf5ObjectFactory.grasp_features(grasps, self.grasp_data(key, gripper), feature_names)

    def store_grasp_features(self, key, grasp_feature_dict, gripper='pr2', stable_pose_id=None, task_id=None, force_overwrite=False):
        """ Add grasp metrics in list |metrics| to the data associated with |grasps| """
        return hfact.Hdf5ObjectFactory.write_grasp_features(grasp_feature_dict, self.grasp_data(key, gripper), force_overwrite)

    # stable pose data
    def stable_poses(self, key, min_p=0.0):
        """ Stable poses for object key """
        stps = hfact.Hdf5ObjectFactory.stable_poses(self.stable_pose_data(key))

        # prune low probability stable poses
        stp_list = []
        for stp in stps:
            if stp.p > min_p:
                stp_list.append(stp)
        return stp_list

    def stable_pose(self, key, stable_pose_id):
        """ Stable pose of stable pose id for object key """
        if stable_pose_id not in self.stable_pose_data(key).keys():
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        return hfact.Hdf5ObjectFactory.stable_pose(self.stable_pose_data(key), stable_pose_id)

    def store_stable_poses(self, key, stable_poses):
        """ Store stable pose data """
        if STP_KEY not in self.object(key).keys():
            self.object(key).create_group(STP_KEY)
        hfact.Hdf5ObjectFactory.write_stable_poses(stable_poses, self.stable_pose_data(key))

    def delete_stable_poses(self, key):
        """ Delete stable pose data for object """
        if STP_KEY in self.object(key).keys():
            del self.object(key)[STP_KEY]

    # rendered image data
    def rendered_images(self, key, stable_pose_id=None, image_type="depth"):
        if stable_pose_id is not None and stable_pose_id not in self.stable_pose_data(key).keys():
            logging.warning('Stable pose id %s unknown' %(stable_pose_id))
            return[]
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in self.stable_pose_data(key)[stable_pose_id].keys():
            logging.warning('No rendered images for stable pose %s' %(stable_pose_id))
            return []
        if stable_pose_id is not None and image_type not in self.rendered_image_data(key, stable_pose_id).keys():
            logging.warning('No rendered images of type %s for stable pose %s' %(image_type, stable_pose_id))
            return []
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in self.object(key).keys():
            logging.warning('No rendered images for object')
            return []
        if stable_pose_id is None and image_type not in self.rendered_image_data(key).keys():
            logging.warning('No rendered images of type %s for object' %(image_type))
            return []

        rendered_images = hfact.Hdf5ObjectFactory.rendered_images(self.rendered_image_data(key, stable_pose_id, image_type))
        for rendered_image in rendered_images:
            rendered_image.obj_key = key
        if stable_pose_id is not None:
            stable_pose = self.stable_pose(key, stable_pose_id)
            for rendered_image in rendered_images:
                rendered_image.stable_pose = stable_pose
        return rendered_images

    def store_rendered_images(self, key, rendered_images, stable_pose_id=None, image_type="depth", force_overwrite=False):
        """ Store rendered images of the object for a given stable pose """
        if stable_pose_id is not None and stable_pose_id not in self.stable_pose_data(key).keys():
            raise ValueError('Stable pose id %s unknown' %(stable_pose_id))
        if stable_pose_id is not None and RENDERED_IMAGES_KEY not in self.stable_pose_data(key)[stable_pose_id].keys():
            self.stable_pose_data(key)[stable_pose_id].create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is not None and image_type not in self.rendered_image_data(key, stable_pose_id).keys():
            self.rendered_image_data(key, stable_pose_id).create_group(image_type)
        if stable_pose_id is None and RENDERED_IMAGES_KEY not in self.object(key).keys():
            self.object(key).create_group(RENDERED_IMAGES_KEY)
        if stable_pose_id is None and image_type not in self.rendered_image_data(key).keys():
            self.rendered_image_data(key).create_group(image_type)

        return hfact.Hdf5ObjectFactory.write_rendered_images(rendered_images, self.rendered_image_data(key, stable_pose_id, image_type),
                                                             force_overwrite)

""" Deprecated dataset for use with filesystems """
class FilesystemDataset(object):
    def __init__(self, dataset_name, config):
        self._parse_config(config)

        self.dataset_name_ = dataset_name
        self.dataset_root_dir_ = os.path.join(self.database_root_dir_, self.dataset_name_)
        self.iter_count_ = 0

        # read in filenames
        self._read_data_keys()

    def _parse_config(self, config):
        self.database_root_dir_ = config['database_dir']

    def _read_data_keys(self, start=0, end=None):
        """Read in all the data keys from start to end in the index."""
        index_filename = os.path.join(self.dataset_root_dir_, INDEX_FILE)
        if not os.path.exists(index_filename):
            raise IOError('Index file does not exist! Invalid dataset: ' + self.dataset_root_dir_)

        self.data_keys_ = []
        self.data_categories_ = {}
        index_file_lines = open(index_filename, 'r').readlines()
        if end is None:
            end = len(index_file_lines)
        for i, line in enumerate(index_file_lines):
            if not (start <= i < end):
                continue

            tokens = line.split()
            if not tokens: # ignore empty lines
                continue

            self.data_keys_.append(tokens[0])
            if len(tokens) > 1:
                self.data_categories_[tokens[0]] = tokens[1]
            else:
                self.data_categories_[tokens[0]] = ''

    @property
    def name(self):
        return self.dataset_name_

    @property
    def data_keys(self):
        return self.data_keys_

    @property
    def dataset_root_dir(self):
        return self.dataset_root_dir_

    @staticmethod
    def sdf_filename(file_root):
        return file_root + '.sdf'

    @staticmethod
    def obj_filename(file_root):
        return file_root + '.obj'

    @staticmethod
    def json_filename(file_root):
        return file_root + '.json'

    @staticmethod
    def stp_filename(file_root):
        return file_root + '.stp'

    @staticmethod
    def features_filename(file_root):
        return file_root + '.ftr'

    def read_datum(self, key):
        """Read in the GraspableObject3D corresponding to given key."""
        if key not in self.data_keys_:
            raise ValueError('Key %s not found in dataset %s' % (key, self.name))

        file_root = os.path.join(self.dataset_root_dir_, key)
        sdf_filename = FilesystemDataset.sdf_filename(file_root)
        obj_filename = FilesystemDataset.obj_filename(file_root)
        features_filename = FilesystemDataset.features_filename(file_root)

        # read in data
        sf = sdf_file.SdfFile(sdf_filename)
        sdf = sf.read()
        
        of = obj_file.ObjFile(obj_filename)
        mesh = of.read()

        if os.path.exists(features_filename):
            ff = feature_file.LocalFeatureFile(features_filename)
            features = ff.read()
        else:
            features = None

        return go.GraspableObject3D(sdf, mesh=mesh, features=features, key=key, model_name=obj_filename, category=self.data_categories_[key])

    def load_grasps(self, key, grasp_dir=None):
        """Loads a list of grasps from a file (grasp_dir/key.json).
        Params:
            key - string name of a graspable
            grasp_dir - string path to the grasp.json directory; defaults to
              self.dataset_root_dir_
        """
        if grasp_dir is None:
            grasp_dir = self.dataset_root_dir_
        path = os.path.join(grasp_dir, FilesystemDataset.json_filename(key))
        try:
            with open(path) as f:
                grasps = jsons.load(f)
        except:
            logging.warning('No grasp file found for key %s' %(key))
            return []
        return [grasp.ParallelJawPtGrasp3D.from_json(g) for g in grasps]

    def save_grasps(self, graspable, grasps):
        """Saves a list of grasps in the database.
        Params:
            graspable - the GraspableObject for the grasps
            grasps - a list of Grasps or a single Grasp to be saved
        """
        if not isinstance(grasps, list): # only one grasp
            grasps = [grasps]
        graspable_dict = {
            'key': graspable.key,
            'category': graspable.category,
            'grasps': [g.to_json() for g in grasps]
        }

        file_root = os.path.join(self.dataset_root_dir_, graspable.key)
        grasp_filename = Dataset.json_filename(file_root)
        # TODO: what should happen if grasp_filename already exists?
        with open(grasp_filename, 'w') as f:
            jsons.dump(grasps, f)

    def load_stable_poses(self, graspable):
        file_root = os.path.join(self.dataset_root_dir_, graspable.key)
        stp = stp_file.StablePoseFile()
        stp_filename = Dataset.stp_filename(file_root)
        stable_poses = stp.read(stp_filename)
        return stable_poses

    def __getitem__(self, index):
        """ Index a particular object in the dataset """
        if isinstance(index, numbers.Number):
            if index < 0 or index >= len(self.data_keys_):
                raise ValueError('Index out of bounds. Dataset contains %d objects' %(len(self.data_keys_)))
            obj = self.read_datum(self.data_keys_[index])
            return obj
        elif isinstance(index, (str, unicode)):
            obj = self.read_datum(index)
            return obj

    def __iter__(self):
        """ Generate iterator """
        self.iter_count_ = 0 # NOT THREAD SAFE!
        return self
    
    def next(self):
        """ Read the next object file in the list """
        if self.iter_count_ >= len(self.data_keys_):
            raise StopIteration
        else:
            logging.info('Returning datum %s' %(self.data_keys_[self.iter_count_]))
            try:
                obj = self.read_datum(self.data_keys_[self.iter_count_])    
            except:
                logging.warning('Error reading %s. Skipping' %(self.data_keys_[self.iter_count_]))
                self.iter_count_ = self.iter_count_ + 1
                return self.next()

            self.iter_count_ = self.iter_count_ + 1
            return obj

    # Hmmm... need to access things by stable pose as well

    # New accessor functions
    def grasps(self, graspable, stable_pose=None, gripper='pr2'):
        """
        Returns grasps for a graspable object

        Params:
           graspable (GO3D) - the object to index
           stable_pose (string) - the tag for the stable pose to use
           gripper (string) - which gripper to use
        """
        pass

    def grasp_features(self, graspable, grasp, stable_pose=None, gripper='pr2'):
        """
        Returns grasp features for a graspable object and single grasp

        Params:
           graspable (GO3D) - the object to index
           grasp (abstract grasp type) - the grasp to get features for
           stable_pose (string) - the tag for the stable pose to use
           gripper (string) - which gripper to use
        """
        pass

    def grasps_and_features(self, graspable, stable_pose=None, gripper='pr2'):
        """
        Returns a list of grasps and grasp features for a graspable object

        Params:
           graspable (GO3D) - the object to index
           stable_pose (string) - the tag for the stable pose to use
           gripper (string) - which gripper to use
        """
        pass

    def rendered_images(self, graspable):
        """
        Returns a list of the rendered images for a graspable object
        """
        pass

    def images(self, graspable):
        """
        Returns the list of "real" color images for the graspable object
        """
        pass

    def point_clouds(self, graspable):
        """
        Returns the list of point clouds for the graspable object
        """
        pass

    def object_nearest_neighbors(self, graspable, k=5):
        """
        Returns the list of nearest neighbors for the graspable object
        """
        pass

    def object_within_radius(self, graspable, eps=1e-3):
        """
        Returns the list of graspable objectS within a given feature distance (hard to decide the distance threshold, I know)
        """
        pass

    # New db writing functions
    def save_grasps2(self, graspsable, grasps, stable_pose=None, gripper='pr2'):
        """
        Saves a list of grasps of the given gripper type for a graspable object in the given stable pose
        """
        pass

    def update_grasp_quality(self, graspsable, grasp, quality_tag, quality, stable_pose=None, gripper='pr2'):
        """
        Updates the quality of a grasp of the given gripper type for a graspable object in the given stable pose
        """
        pass

    def save_grasp_features(self, graspable, grasp, grasp_features, stable_pose=None, gripper='pr2'):
        """
        Saves a list of grasp features for grasp of a given gripper type on a graspable object
        """
        pass

    def save_grasps_with_features(self, graspable, grasps, grasp_features):
        """
        Saves a list of grasps with corresponding features
        """
        pass

    def save_rendered_images(self, graspable, images):
        """
        Saves a set of rendered images for a graspable object
        """
        pass

    def save_image(self, graspable, image):
        """
        Saves a "real" image of a graspable object
        """
        pass

    def save_point_cloud(self, graspable, point_cloud):
        """
        Saves a point cloud of a graspable object
        """
        pass

class FilesystemChunk(FilesystemDataset):
    def __init__(self, config):
        self._parse_config(config)

        self.dataset_root_dir_ = os.path.join(self.database_root_dir_, self.dataset_name_)
        self.iter_count_ = 0

        # read in filenames
        self._read_data_keys(self.start, self.end)

    def _parse_config(self, config):
        super(FilesystemChunk, self)._parse_config(config)
        self.dataset_name_ = config['dataset']
        self.start = config['datasets'][self.dataset_name_]['start_index']
        self.end = config['datasets'][self.dataset_name_]['end_index']

def test_dataset():
    logging.getLogger().setLevel(logging.INFO)
    config_filename = 'cfg/basic_labelling.yaml'
    config = ec.ExperimentConfig(config_filename)

    db = Database(config)
    keys = []
    logging.info('Reading datset %s' %(db.datasets[0].name))
    for obj in db.datasets[0]:
        keys.append(obj.key)

    assert(len(keys) == 26)

def test_load_grasps():
    logging.getLogger().setLevel(logging.INFO)
    config_filename = 'cfg/basic_labelling.yaml'
    config = ec.ExperimentConfig(config_filename)

    key = 'feline_greenies_dental_treats'
    db = Database(config)
    apc = db.datasets[0]
    grasps = apc.load_grasps(key, 'results/gce_grasps/amazon_picking_challenge')
    graspable = apc[key]

if __name__ == '__main__':
    test_dataset()
