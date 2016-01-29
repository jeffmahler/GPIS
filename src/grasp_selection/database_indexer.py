from abc import ABCMeta, abstractmethod

import cnn_feature_extractor as cfex
import kernels

class Hdf5DatabaseIndexer:
    """
    Abstract class for database indexing. Main purpose is to wrap individual datasets.
    Basically wraps the kernel nearest neighbor classes to automatically use HDF5 data and specific featurizations.
    """
    __metaclass__ = ABCMeta

    def __init__(self, hdf5_data, matcher):
        if not isinstance(matcher, Kernels.NearestNeighbor):
            raise ValueError('Must provide a nearest neighbor object for indexing')
        self.hdf5_data = hdf5_data # handle to hdf5 data
        self.matcher_ = matcher # nearest neighbor object

    @abstractmethod
    def _retrieve_objects(self):
        """ Private method to retrieve objects from an HDF5 database """
        pass

    def _create_table(self):
        """ Creates the master indexing table """
        object_list = self._retrieve_objects()
        featurized_objects = self.featurize(object_list)
        self.matcher_.train(featurized_objects)

    def _featurize(self, datapoints):
        """ Featurizes the datapoints """
        return datapoints

    def nearest(self, query, return_indices=False):
        """ Featurizes a datapoint x from the database """
        # something here to ask nearest neighbor structure...
        return self.k_nearest(query, 1, return_indices)

    def k_nearest(self, query, k, return_indices=False):
        """ Featurizes a datapoint x from the database """
        # something here to ask nearest neighbor structure...
        return self.matcher_.nearest_neighbors(query, k, return_indices)

    def within_distance(self, query, dist=0.5, return_indices=False):
        """ Featurizes a datapoint x from the database """
        # something here to ask nearest neighbor structure...
        return self.matcher_.within_distance(query, dist, return_indices)

class CNN_Hdf5DatabaseIndexer(Hdf5DatabaseIndexer):
    """ Indexes data using the distance between CNN representations of images """
    def __init__(self, hdf5_data, config):
        self.feature_extractor_ = cfex.CNNBatchFeatureExtractor(config)
        matcher = kernels.KDTree(phi = lambda x: x[1])
        Hdf5DatabaseIndexer.__init__(self, hdf5_data, matcher)

    def _retrieve_objects(self):
        """ Retrieves objects from the database """   
        pass

    def _featurize(self, datapoints):
        """ Converts an image x to a CNN feature vector """
        images = [x.image for x in datapoints]
        descriptors = self.feature_extractor_.extract(images)
        for x, descriptor in zip(datapoints, descriptors):
            x.descriptor = descriptor
        return datapoints

# TODO: Implement below when needed
class RawHdf5DatabaseIndexer(HDF5DatabaseIndexer):
    """ Indexes data using the raw distance between objects """
    pass

class MVCNN_Hdf5DatabaseIndexer(HDF5DatabaseIndexer):
    """ Indexes data using the distance between MV-CNN representations of objects """
    pass
