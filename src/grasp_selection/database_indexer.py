from abc import ABCMeta, abstractmethod

import IPython
from PIL import Image
import os
import sys

import matplotlib.pyplot as plt

import cnn_feature_extractor as cfex
import database as db
import experiment_config as ec
import graspable_object as go
import kernels
import numpy as np
import rendered_image as ri

class Hdf5DatabaseIndexer:
    """
    Abstract class for database indexing. Main purpose is to wrap individual datasets.
    Basically wraps the kernel nearest neighbor classes to automatically use HDF5 data and specific featurizations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, matcher):
        if not isinstance(matcher, kernels.NearestNeighbor):
            raise ValueError('Must provide a nearest neighbor object for indexing')
        self.matcher_ = matcher # nearest neighbor object
        self._create_table()

    @abstractmethod
    def _retrieve_objects(self):
        """ Private method to retrieve objects from an HDF5 database """
        pass

    def _create_table(self):
        """ Creates the master indexing table """
        object_list = self._retrieve_objects()
        featurized_objects = self._featurize(object_list)
        self.matcher_.train(featurized_objects)

    def _featurize(self, datapoints):
        """ Featurizes the datapoints """
        return datapoints

    def nearest(self, query, return_indices=False):
        """ Featurizes a datapoint x from the database """
        return self.k_nearest(query, 1, return_indices)

    def k_nearest(self, query, k, return_indices=False):
        """ Featurizes a datapoint x from the database """
        featurized_query = self._featurize([query])[0]
        return self.matcher_.nearest_neighbors(featurized_query, k, return_indices)

    def within_distance(self, query, dist=0.5, return_indices=False):
        """ Featurizes a datapoint x from the database """
        featurized_query = self._featurize([query])[0]
        return self.matcher_.within_distance(featurized_query, dist, return_indices)

class CNN_Hdf5DatabaseIndexer(Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, config):
        self.feature_extractor_ = cfex.CNNBatchFeatureExtractor(config)
        matcher = kernels.KDTree(phi = lambda x: x.descriptor)
        self._parse_config(config)
        Hdf5DatabaseIndexer.__init__(self, matcher)

    def _parse_config(self, config):
        """ Parse the config to read in key parameters """
        self.use_stable_poses_ = config['use_stable_poses']
        self.image_type_ = config['image_type']

    def _featurize(self, datapoints):
        """ Converts an image x to a CNN feature vector """
        images = [x.image for x in datapoints]
        descriptors = self.feature_extractor_.extract(images)
        for x, descriptor in zip(datapoints, descriptors):
            x.descriptor = descriptor
        return datapoints

class CNN_Hdf5DatasetIndexer(CNN_Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, dataset, config):
        if not isinstance(dataset, db.Hdf5Dataset):
            raise ValueError('Must provide an Hdf5 dataset object to index')
        self.dataset_ = dataset # handle to hdf5 data
        CNN_Hdf5DatabaseIndexer.__init__(self, config)

    def _retrieve_objects(self):
        """ Retrieves objects from the provided dataset. """
        rendered_image_pool = []
        for obj_key in dataset.object_keys:
            if self.use_stable_poses_:
                stable_poses = self.dataset_.stable_poses(obj_key)
                for stable_pose in stable_poses:
                    rendered_image_pool.extend(self.dataset_.rendered_images(obj_key, stable_pose_id=stable_pose.id, image_type=self.image_type_))
            else:
                rendered_image_pool.extend(self.dataset_.rendered_images(obj_key, image_type=self.image_type_))
        return rendered_image_pool

class CNN_Hdf5ObjectIndexer(CNN_Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, obj_key, dataset, config):
        if not isinstance(dataset, db.Hdf5Dataset):
            raise ValueError('Must provide an Hdf5 dataset object to index')
        if obj_key not in dataset.object_keys:
            raise ValueError('Object key %s not in datset' %(obj_key))            
        self.obj_key_ = obj_key
        self.dataset_ = dataset # handle to hdf5 data
        CNN_Hdf5DatabaseIndexer.__init__(self, config)

    def _retrieve_objects(self):
        """ Retrieves objects from the provided dataset. """
        rendered_image_pool = []
        if self.use_stable_poses_:
            stable_poses = self.dataset_.stable_poses(self.obj_key_)
            for stable_pose in stable_poses:
                rendered_image_pool = self.dataset_.rendered_images(self.obj_key_, stable_pose_id=stable_pose.id, image_type=self.image_type_)
        else:
            rendered_image_pool = self.dataset_.rendered_images(self.obj_key_, image_type=self.image_type_)
        return rendered_image_pool

# TODO: Implement below when needed
class RawHdf5DatabaseIndexer(Hdf5DatabaseIndexer):
    """ Indexes data using the raw distance between objects """
    pass

class MVCNN_Hdf5DatabaseIndexer(Hdf5DatabaseIndexer):
    """ Indexes data using the distance between MV-CNN representations of objects """
    pass

if __name__ == '__main__':
    config_filename = 'cfg/test_cnn_database_indexer.yaml'
    config = ec.ExperimentConfig(config_filename)

    test_image_filename = 'data/test/database_indexing/spray_binary.jpg'
    test_image = np.array(Image.open(test_image_filename).convert('RGB'))
    IPython.embed()
    test_image = ri.RenderedImage(test_image, np.zeros(3), np.zeros(3), np.zeros(3))

    database_name = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_name, config)
    dataset = database.dataset(config['datasets'].keys()[0])

    cnn_indexer = CNN_Hdf5ObjectIndexer('spray', dataset, config)
    nearest_neighbors = cnn_indexer.k_nearest(test_image, k=5)

    nearest_images = nearest_neighbors[0]
    nearest_distances = nearest_neighbors[1]
    font_size = 15
    j = 0

    plt.figure()
    plt.subplot(2, 5, 3)
    plt.imshow(test_image.image, cmap=plt.cm.Greys_r, interpolation='none')
    plt.title('QUERY IMAGE', fontsize=font_size)
    plt.axis('off')

    for image, distance in zip(nearest_images, nearest_distances):
        plt.subplot(2, 5, j+6)
        plt.imshow(image.image, cmap=plt.cm.Greys_r, interpolation='none')
        plt.title('NEIGHBOR %d, DISTANCE = %f' %(j, distance), fontsize=font_size)
        plt.axis('off')
        j = j + 1

    plt.show()

