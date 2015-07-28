"""
Classes for similarity functions and nearest neighbors.

Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod
import IPython
import logging
import time

import pickle
import numpy as np
import scipy.sparse as ss
from sklearn import neighbors

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.distances import Distance, EuclideanDistance
from nearpy.filters import NearestFilter

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

class KLDivergence(Distance):
    """ Symmetric version of KL divergence """

    def distance(self, x, y):
        """ Distance between two probability distributions """
        log_pq_ratio = np.log(x / y)
        return (x - y).dot(log_pq_ratio)

DISTANCE_FNS = {
    'euclidean': euclidean_distance,
    'nearpy_euclidean': EuclideanDistance,
    'kl_divergence': KLDivergence
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

    def matrix(self, data):
        """Computes the kernel matrix for a list of data."""
        num_data = len(data)
        mat = np.zeros((num_data, num_data))
        for i, x in enumerate(data):
            for j, y in enumerate(data):
                if i <= j:
                    mat[i, j] = mat[j, i] = self(x, y)
        return mat

    def __call__(self, x, y):
        return self.evaluate(x, y)

    def __mul__(self, other):
        assert isinstance(other, Kernel), 'Can only multiply Kernels!'
        return KernelProduct([self, other])

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
        return self.sigma_**2 * np.exp(-self.dist_(x, y)**2 / (2 * self.l_**2))

    def gradient(self, x):
        raise NotImplementedError

class SymmetricSquaredExponentialKernel(Kernel):
    def __init__(self, sigma=1.0, l=1.0, dist='euclidean',
                 phi=None, alternate_phi=None):
        assert phi is not None and alternate_phi is not None, 'Two phi needed'
        self.k1_ = SquaredExponentialKernel(sigma, l, dist, phi)
        self.k2_ = SquaredExponentialKernel(sigma, l, dist, alternate_phi)

    def error_radius(self, tolerance):
        return self.k1_.error_radius(tolerance) # should be same as k2

    def evaluate(self, x, y):
        k1, k2 = self.k1_.evaluate(x, y), self.k2_.evaluate(x, y)
        return np.mean([k1, k2])

    def gradient(self, x):
        raise NotImplementedError

class KernelProduct(Kernel):
    """ k(x, y) = k1(x, y) * k2(x, y) * ... """
    def __init__(self, kernels, phi=None):
        self.kernels_ = kernels
        self.phi_ = phi

    def evaluate(self, x, y):
        if self.phi_ is not None:
            x, y = self.phi_(x), self.phi_(y)
        return np.prod([k(x, y) for k in self.kernels_])

    def error_radius(self, tolerance):
        raise NotImplementedError

    def gradient(self, x):
        raise NotImplementedError

class NearestNeighbor:
    __metaclass__ = ABCMeta

    def __init__(self, dist=neighbors.DistanceMetric.get_metric('euclidean'), phi=lambda x: x):
        self.data_ = None
        self.featurized_ = None
        self.dist_metric_ = dist
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

    def featurize(self, data):
        """Computes features for each object in data.
        Params:
            data - list of objects to compute features for
        Returns:
            numpy array of feature values (num_data x num_features)
        """
        featurized = []
        for i, d in enumerate(data, 1):
            logging.info('Extracting features from object %d (of %d).' %(i, len(data)))
            featurize_start = time.clock()
            featurized.append(self.phi_(d))
            featurize_end = time.clock()
            logging.info('Took %f sec' %(featurize_end - featurize_start))
        return np.array(featurized)

class BinaryTree(NearestNeighbor):
    def train(self, data, tree_class=neighbors.KDTree):
        self.data_ = np.array(data)
        self.featurized_ = self.featurize(data)

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

class SymmetricKDTree(KDTree):
    def __init__(self, dist=neighbors.DistanceMetric.get_metric('euclidean'),
                 phi=lambda x: x, alternate_phi=lambda x: x):
        KDTree.__init__(self, dist, phi)
        self.alternate_phi_ = alternate_phi

    def within_distance(self, x, dist=0.2, return_indices=False):
        indices, distances = self.tree_.query_radius(self.phi_(x), dist,
                                                     return_distance=True)
        indices = indices[0]
        alt_indices_and_distances = self.tree_.query_radius(self.alternate_phi_(x), dist,
                                                             return_distance=True)
        alt_indices, alt_distances = [], []
        for alt_index, alt_dist in alt_indices_and_distances:
            if alt_index not in indices:
                alt_indices.append(alt_index)
                alt_distances.append(alt_dist)

        indices = np.concatenate(indices, alt_indices)
        distances = np.concatenate(distances, alt_distances)

        if return_indices:
            return indices, distances
        else:
            return self.data_[indices], distances

class LSHForest(NearestNeighbor):
    # Warning: all distances returned by LSHF are cosine distances!

    def train(self, data):
        self.data_ = np.array(data)
        self.featurized_ = self.featurize(data)

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

class NearPy(NearestNeighbor):
    def __init__(self, dist=EuclideanDistance(), phi=lambda x: x):
        NearestNeighbor.__init__(self, dist, phi)

    def _create_engine(self, k, lshashes=None):
        self.k_ = k
        self.engine_ = Engine(self.dimension_, lshashes,
                              distance=self.dist_metric_,
                              vector_filters=[NearestFilter(k)])

        for i, feature in enumerate(self.featurized_):
            if self.transpose_:
                self.engine_.store_vector(feature.T, i)
            else:
                self.engine_.store_vector(feature, i)

    def train(self, data, k=10):
        self.data_ = np.array(data)
        self.featurized_ = self.featurize(data)

        shape = featurized[0].shape
        assert len(shape) <= 2, 'Feature shape must be (1, N), (N, 1), or (N,)'
        if len(shape) == 1:
            self.transpose_ = False
            self.dimension_ = shape[0]
        else:
            assert 1 in shape, 'Feature shape must be (1, N) or (N, 1)'
            self.transpose_ = (shape[0] == 1)
            self.dimension_ = shape[1] if self.transpose_ else shape[0]

        logging.info('Constructing nearest neighbor data structure.')
        train_start = time.clock()
        self._create_engine(k)
        train_end = time.clock()
        logging.info('Took %f sec' %(train_end - train_start))

    def within_distance(x, dist=0.5, return_indices=False):
        raise NotImplementedError

    def nearest_neighbors(self, x, k, return_indices=False):
        # HACK: load all data back into new engine if k doesn't match
        if k != self.k_:
            self._create_engine(k)

        feature = self.phi_(x)
        if self.transpose_:
            query_result = self.engine_.neighbours(feature.T)
        else:
            query_result = self.engine_.neighbours(feature)

        if len(query_result) == 0:
            return [], []

        features, indices, distances = zip(*query_result)
        if return_indices:
            return list(indices), list(distances)
        else:
            indices = np.array(indices)
            return list(self.data_[indices]), list(distances)

def test(nn_class, distance, k, within_distance=True, nearest_neighbors=True):
    np.random.seed(0)
    data = np.random.rand(100, 100)
    datum = data[0]

    name = nn_class.__name__
    nn = nn_class()
    nn.train(data)

    if within_distance:
        obj, _ = nn.within_distance(datum, distance)
        print '%s: %d objects within %f' %(name, len(obj), distance)

    if nearest_neighbors:
        _, dist = nn.nearest_neighbors(datum, k)
        print '%s: %d-NN are within %f' %(name, k, np.max(dist))

def test_kdtree(distance=3.7, k=3, **kw):
    test(KDTree, distance, k, **kw)

def test_balltree(distance=3.7, k=3, **kw):
    test(BallTree, distance, k, **kw)

def test_lshf(distance=0.23, k=3, **kw):
    test(LSHForest, distance, k, **kw)

def test_nearpy(distance=None, k=3, **kw):
    test(NearPy, distance, k, **kw)

def test_sparse():
    dim = 500
    num_train = 1000
    num_test = 1
    train_data = ss.rand(dim, num_train)#pickle.load('/home/jmahler/Downloads/feature_objects.p')
    test_data = ss.rand(dim, num_test)

    rbp = RandomBinaryProjections('rbp', 10)
    engine = Engine(dim, lshashes=[rbp])

    for i in range(num_train):
        engine.store_vector(train_data.getcol(i))

    for j in range(num_test):
        N = engine.neighbours(test_data.getcol(j))
        print N

    IPython.embed()

def test_kernels():
    np.random.seed(0)
    x, y = np.random.rand(10), np.random.rand(10)

    k1 = SquaredExponentialKernel(0.5, 2.0)
    k2 = SquaredExponentialKernel()
    k111 = KernelProduct([k1, k1, k1])
    k12 = KernelProduct([k1, k2])
    k21 = KernelProduct([k2, k1])

    k1_result = k1(x, y)
    k2_result = k2(x, y)
    k111_result = k111(x, y)
    k12_result = k12(x, y)
    k21_result = k21(x, y)

    assert k111_result == k1_result**3
    assert k12_result == k21_result == k1_result * k2_result

    k12111 = KernelProduct([k12, k111])
    k12111_result = k12111(x, y)

    assert k12111_result == k12_result * k111_result

    k1221 = k12 * k21
    k1221_result = k1221(x, y)
    assert k1221_result == k12_result**2

if __name__ == '__main__':
    test_kdtree()
    test_balltree()
    test_lshf()
    test_sparse()
    test_kernels()
