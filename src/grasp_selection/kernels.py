"""
Classes for similarity functions and nearest neighbors.

Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod

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
    def gradient(self, x):
        pass

    def __call__(self, x, y):
        return self.evaluate(x, y)

class SquaredExponentialKernel(Kernel):
    """ k(x, y) = \sigma^2 exp(-||x - y||^2 / 2l^2) """
    def __init__(self, sigma=1.0, l=1.0, dist='euclidean'):
        # kernel parameters
        self.sigma_ = sigma
        self.l_ = l

        # look up distance function name, default to Euclidean
        self.dist_ = DISTANCE_FNS.get(dist, euclidean_distance)

    def evaluate(self, x, y):
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
    def within_distance(self, x, dist=0.5):
        """Returns the objects that are close to x and within a distance of dist
        according to self.dist_metric_.
        Params:
            x - object to find neighbors for
            dist - (float) cutoff for how far neighbors can be
        Returns: list of objects, list of distances
        """

    @abstractmethod
    def nearest_neighbors(self, x, k):
        """Returns the k nearest neighbors to x.
        Params:
            x - object to find neighbors for
            k - (int) number of neighbors to return
        Returns: list of objects, list of distances
        """

class BinaryTree(NearestNeighbor):
    def train(self, data, tree_class=neighbors.KDTree):
        self.data_ = np.array(data)
        self.featurized_ = np.array([self.phi_(d) for d in data])
        self.tree_ = tree_class(self.featurized_, metric=self.dist_metric_)

    def within_distance(self, x, dist=0.2):
        indices, distances = self.tree_.query_radius(self.phi_(x), dist,
                                                     return_distance=True)
        return self.data_[indices[0]], distances

    def nearest_neighbors(self, x, k):
        distances, indices = self.tree_.query(self.phi_(x), k,
                                              return_distance=True)
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
        self.featurized_ = np.array([self.phi_(d) for d in data])
        self.lshf_ = neighbors.LSHForest() # TODO -- set params
        self.lshf_.fit(data)

    def within_distance(self, x, dist=0.2):
        distances, indices = self.lshf_.radius_neighbors(self.phi_(x), dist,
                                                         return_distance=True)
        return self.data_[indices[0]], distances

    def nearest_neighbors(self, x, k):
        distances, indices = self.lshf_.kneighbors(self.phi_(x), k,
                                                   return_distance=True)
        return self.data_[indices], distances

def test_kdtree():
    np.random.seed(0)
    data = np.random.rand(100, 100)
    datum = data[0]

    kdtree = KDTree()
    kdtree.train(data)
    print kdtree.within_distance(datum, 3.7)
    print kdtree.nearest_neighbors(datum, 3)

def test_balltree():
    np.random.seed(0)
    data = np.random.rand(100, 100)
    datum = data[0]

    ball_tree = BallTree()
    ball_tree.train(data)
    print ball_tree.within_distance(datum, 3.7)
    print ball_tree.nearest_neighbors(datum, 3)

def test_lshf():
    np.random.seed(0)
    data = np.random.rand(100, 100)
    datum = data[0]

    lshf = LSHForest()
    lshf.train(data)
    print lshf.within_distance(datum, 0.23)
    print lshf.nearest_neighbors(datum, 3)

if __name__ == '__main__':
    test_kdtree()
    test_balltree()
    test_lshf()
