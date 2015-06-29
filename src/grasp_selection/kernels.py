"""
Classes for similarity functions and nearest neighbors.

Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod
import time

import numpy as np
from sklearn import neighbors

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

DISTANCE_FNS = {
    'euclidean': euclidean_distance,
}

class Kernel:
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, x, y):
        """Evaluates the kernel function with two objects x and y."""
        pass

    @abstractmethod
    def error_radius(self, tolerance):
        """Returns the radius of the ball that produces an error less than
        tolerance by solving tolerance = k(x, y)."""
        pass

    @abstractmethod
    def gradient(self, x):
        pass

    def __call__(self, x, y):
        return self.evaluate(x, y)

class SquaredExponentialKernel(Kernel):
    """ k(x, y) = \sigma^2 exp(-||x - y||^2 / 2l^2) """
    def __init__(self, sigma=1.0, l=1.0, dist='euclidean', phi=None):
        # kernel parameters
        self.sigma_ = sigma
        self.l_ = l

        # look up distance function name, default to Euclidean
        self.dist_ = DISTANCE_FNS.get(dist, euclidean_distance)
        self.phi_ = phi

    def error_radius(self, tolerance):
        assert 0 < tolerance <= 1, 'Tolerance must be between 0 and 1.'
        # TODO: should this depend on the distance function?
        return np.sqrt(2.0 * self.l_**2 * np.log(self.sigma_**2 / tolerance))

    def evaluate(self, x, y):
        if self.phi_ is not None:
            x, y = self.phi_(x), self.phi_(y)
        return self.sigma_**2 * np.exp(-self.dist_(x, y)**2 / 2 * self.l_**2)

    def gradient(self, x):
        todo = 1

class NearestNeighbor:
    __metaclass__ = ABCMeta

    def __init__(self, dist='euclidean', phi=lambda x: x):
        self.data_ = None
        self.featurized_ = None
        self.dist_metric_ = neighbors.DistanceMetric.get_metric(dist)
        self.phi_ = phi # feature fn to extract an np.array from each object

    @abstractmethod
    def train(self, data):
        """Creates an internal data structure for nearest neighbor lookup. Also
        sets the data_ and featurized_ attributes.
        Params:
            data - list of objects to insert into the data structure
        """

    @abstractmethod
    def within_distance(self, x, dist=0.5, return_indices=False):
        """Returns the objects that are close to x and within a distance of dist
        according to self.dist_metric_.
        Params:
            x - object to find neighbors for
            dist - (float) cutoff for how far neighbors can be
            return_indices - (bool) True if returning indices rather than objects
        Returns:
            list of objects, list of distances (when return_indices is False)
            list of indices, list of distances (when return_indices is True)
        """

    @abstractmethod
    def nearest_neighbors(self, x, k, return_indices=False):
        """Returns the k nearest neighbors to x.
        Params:
            x - object to find neighbors for
            k - (int) number of neighbors to return
            return_indices - (bool) True if returning indices rather than objects
        Returns:
            list of objects, list of distances (when return_indices is False)
            list of indices, list of distances (when return_indices is True)
        """

class BinaryTree(NearestNeighbor):
    def train(self, data, tree_class=neighbors.KDTree):
        self.data_ = np.array(data)
        featurized = []
        for i, d in enumerate(data, 1):
            logging.info('Extracting features from object %d (of %d).' %(i, len(data)))
            featurize_start = time.clock()
            featurized.append(self.phi_(d))
            featurize_end = time.clock()
            logging.info('Took %f sec' %(featurize_end - featurize_start))
        self.featurized_ = np.array(featurized)

        logging.info('Constructing nearest neighbor data structure.')
        train_start = time.clock()
        self.tree_ = tree_class(self.featurized_, metric=self.dist_metric_)
        train_end = time.clock()
        logging.info('Took %f sec' %(train_end - train_start))

    def within_distance(self, x, dist=0.2, return_indices=False):
        indices, distances = self.tree_.query_radius(self.phi_(x), dist,
                                                     return_distance=True)
        indices = indices[0]
        if return_indices:
            return indices, distances
        else:
            return self.data_[indices], distances

    def nearest_neighbors(self, x, k, return_indices=False):
        distances, indices = self.tree_.query(self.phi_(x), k,
                                              return_distance=True)
        if return_indices:
            return indices, distances
        else:
            return self.data_[indices], distances

class KDTree(BinaryTree):
    def train(self, data):
        BinaryTree.train(self, data, tree_class=neighbors.KDTree)

class BallTree(BinaryTree):
    def train(self, data):
        BinaryTree.train(self, data, tree_class=neighbors.BallTree)

class LSHForest(NearestNeighbor):
    # Warning: all distances returned by LSHF are cosine distances!

    def train(self, data):
        self.data_ = np.array(data)
        featurized = []
        for i, d in enumerate(data, 1):
            logging.info('Extracting features from object %d (of %d).' %(i, len(data)))
            featurize_start = time.clock()
            featurized.append(self.phi_(d))
            featurize_end = time.clock()
            logging.info('Took %f sec' %(featurize_end - featurize_start))
        self.featurized_ = np.array(featurized)

        logging.info('Constructing nearest neighbor data structure.')
        train_start = time.clock()
        self.lshf_ = neighbors.LSHForest() # TODO -- set params
        self.lshf_.fit(data)
        train_end = time.clock()
        logging.info('Took %f sec' %(train_end - train_start))

    def within_distance(self, x, dist=0.2, return_indices=False):
        distances, indices = self.lshf_.radius_neighbors(self.phi_(x), dist,
                                                         return_distance=True)
        indices = indices[0]
        if return_indices:
            return indices, distances
        else:
            return self.data_[indices], distances

    def nearest_neighbors(self, x, k, return_indices=False):
        distances, indices = self.lshf_.kneighbors(self.phi_(x), k,
                                                   return_distance=True)
        if return_indices:
            return indices, distances
        else:
            return self.data_[indices], distances

def test(nn_class, distance, k):
    np.random.seed(0)
    data = np.random.rand(100, 100)
    datum = data[0]

    name = nn_class.__name__
    nn = nn_class()
    nn.train(data)

    obj, _ = nn.within_distance(datum, distance)
    print '%s: %d objects within %f' %(name, len(obj), distance)

    _, dist = nn.nearest_neighbors(datum, k)
    print '%s: %d-NN are within %f' %(name, k, np.max(dist))

def test_kdtree(distance=3.7, k=3):
    test(KDTree, distance, k)

def test_balltree(distance=3.7, k=3):
    test(BallTree, distance, k)

def test_lshf(distance=0.23, k=3):
    test(LSHForest, distance, k)

if __name__ == '__main__':
    test_kdtree()
    test_balltree()
    test_lshf()
