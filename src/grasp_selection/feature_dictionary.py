from abc import ABCMeta, abstractmethod

import IPython
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import os
import sys

import sklearn.cluster as sc
import sklearn.decomposition as sd

import features as fs
import feature_file as ff
import kernels

class ZcaTransform():
    def __init__(self):
        self.C_ = None # rotation, identity, plus rotation tf
        self.mu_ = None # data centering

    def fit(self, X, eps = 1e-4, apply_tf=False):
        """ Fit a ZCA transform to n_samples x dim data matrix X, optionally applying it as well """
        # center data
        self.mu_ = np.mean(X, axis=0)
        X_centered = X - self.mu_

        # estimate covariance
        Sigma = X_centered.T.dot(X_centered)
        U, S, V = np.linalg.svd(Sigma)
        S_sqrt_inv = 1.0 / np.sqrt(S + eps)
        S_sqrt_inv[np.isinf(S_sqrt_inv)] = 0

        # zca is matrix "closest" to original data, U * S^(1/2) * V^T
        self.C_ = U.dot(np.diag(S_sqrt_inv)).dot(V)

        # apply tf to data
        if apply_tf:
            return self.apply(X)
        return None

    def fit_bootstrapped(self, mu, Sigma, eps=1e-4):
        """ Fit a ZCA transform to data with mean and Sigma estimated externally """
        # assign mean, decompose covariance
        self.mu_ = mu
        U, S, V = np.linalg.svd(Sigma)
        S_sqrt_inv = 1.0 / np.sqrt(S + eps)
        S_sqrt_inv[np.isinf(S_sqrt_inv)] = 0

        # zca is matrix "closest" to original data, U * S^(1/2) * V^T
        self.C_ = U.dot(np.diag(S_sqrt_inv)).dot(V)

    def transform(self, X):
        """ Apply ZCA transform to n_samples x dim data matrix X """
        return self.C_.dot((X - self.mu_).T).T        

    def __call__(self, X):
        """ Allows for parenthetical indexing """
        return self.transform(X)

class FeatureDictionary:
    __metaclass__ = ABCMeta
    def fit(self):
        """ Learn a feature dictionary """
        pass

    def transform(self, features):
        """ Represent data points by words in dictionary """
        pass

class KMeansFeatureDictionary(FeatureDictionary):
    def __init__(self, num_clusters, preprocess_fn = lambda x : x):
        self.kmeans_ = sc.KMeans(num_clusters)
        self.num_clusters_ = num_clusters
        self.phi_ = preprocess_fn # preprocessing transform

    def fit(self, features):
        self.kmeans_.fit(self.phi_(features.descriptors))

    def transform(self, features, eps=1e-2):
        """ Create a normalized histogram of dictionary "words" present in the object """
        labels = self.kmeans_.predict(self.phi_(features.descriptors))
        word_histogram, words = np.histogram(labels, bins=np.linspace(0, self.num_clusters_, self.num_clusters_+1))
        word_histogram = word_histogram + eps # smooth things out a tiny bit
        word_histogram = word_histogram.astype(np.float32) / np.sum(word_histogram)
        return word_histogram

def test_random_points():
    num_points = 100
    dim = 2
    X = np.random.rand(num_points, dim)

    z = ZcaTransform()
    X_t = z.fit(X, apply_tf=True)

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c=u'b')
    plt.scatter(X_t[:,0], X_t[:,1], c=u'g')
    plt.show()    

def zca_from_shot(feature_dir, num_samples_per_shape = 75, num_clusters = 25):
    num_shapes = 0
    feature_count = 0
    b = fs.BagOfFeatures()
    feat_filenames = []
    cat_filenames = []

    # walk through directory, adding files to feature rep
    for root, sub_folders, files in os.walk(feature_dir):
        for f in files:            
            file_name = os.path.join(root, f)
            file_root, file_ext = os.path.splitext(file_name)
            
            if file_ext == '.cat':
                cat_filenames.append(file_name)
                file_name = file_root+'_features.txt'
                feat_filenames.append(file_name)
                print 'Processing file %s (%d of %d)' %(feat_filenames[-1], num_shapes, len(files) / 2)

                # read features
                feat_file = ff.LocalFeatureFile(file_name)
                features = feat_file.read()

                # get a random subset of features
                num_features = features.descriptors.shape[0]
                num_samples = min(num_samples_per_shape, num_features)
                indices = np.random.choice(num_features, num_samples, replace=False)
                descriptor_subset = features.descriptors[indices,:]

                if num_shapes == 0:
                    feature_dim = features.descriptors.shape[1]
                    mu = np.zeros(feature_dim)
                    Sigma = np.zeros([feature_dim, feature_dim])

                # update data matrix, mean, covariance estimates
                new_feature_count = feature_count + num_samples
                old_weight = float(feature_count) / float(new_feature_count)
                new_weight = float(num_samples) / float(new_feature_count)
                b.extend(features.feature_subset(indices))
                mu = old_weight * mu + new_weight * np.sum(descriptor_subset, axis=0)
                Sigma = old_weight * Sigma + new_weight * descriptor_subset.T.dot(descriptor_subset)

                num_shapes += 1
                feature_count += num_samples
#                if num_shapes >= 200:
#                    break

    # preprocessing transform with zca whitening
    print 'Learning ZCA'
    z = ZcaTransform()
    z.fit_bootstrapped(mu, Sigma)

    # lear feature dict with kmeans
    print 'Learning dict'
    k = KMeansFeatureDictionary(num_clusters, preprocess_fn = z)
    k.fit(b)

    # loop through objects and transform
    shape_reps = np.zeros([len(feat_filenames),num_clusters])
    categories = []
    i = 0
    print 'Repping shapes'
    for feat_file_name, cat_file_name in zip(feat_filenames, cat_filenames):
        print 'Repping ', feat_file_name
        cat_file = open(cat_file_name, 'r')
        cat = cat_file.readline()

        feat_file = ff.LocalFeatureFile(feat_file_name)
        features = feat_file.read()    
        shape_rep = k.transform(features)
                          
        shape_reps[i,:] = shape_rep
        categories.append(cat)
        i += 1

    # transform everything for plotting
    cat_list = list(set(categories))
    cat_indices = [cat_list.index(c) for c in categories]

    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, len(cat_list)))
    pointwise_colors = colors[cat_indices, :]

    p = sd.PCA()
    shapes_proj = p.fit_transform(shape_reps)
    shapes_tf = shapes_proj[:,:2]

    # plot all points
    patches = []
    for i in range(len(cat_list)):
        patches.append(mpatches.Patch(color='red', label=cat_list[i]))

    cat_array = np.array(cat_indices)
    objs = []
    plt.figure()
    for i in range(len(cat_list)):
        cat_ind = np.where(cat_array == i)
        cat_ind = cat_ind[0]
        o = plt.scatter(shapes_tf[cat_ind,0], shapes_tf[cat_ind,1], c=colors[i])
        objs.append(o)
    plt.legend(objs, cat_list)
    plt.show()

    f = open('shot_feature_dict.pkl', 'w')
    pkl.dump(k, f)

    # nearest neighbors queries
    train_pct = 0.75
    num_pts = len(categories)
    all_indices = np.linspace(0, num_pts-1, num_pts)
    train_indices = np.random.choice(num_pts, np.floor(train_pct * num_pts), replace=False)
    test_indices = np.setdiff1d(all_indices, train_indices)
    train_indices = train_indices.astype(np.int16)
    test_indices = test_indices.astype(np.int16)
    
    train_categories = []
    for i in range(train_indices.shape[0]):
        train_categories.append(categories[train_indices[i]])

    test_categories = []
    for i in range(test_indices.shape[0]):
        test_categories.append(categories[test_indices[i]])

    # nearest neighbors
    num_nearest = 5
    nn = kernels.NearPy()
    nn.train(shape_reps[train_indices,:], k=num_nearest)

    # setup confusion
    confusion = {}
    confusion[UNKNOWN_TAG] = {}
    for query_cat in cat_list:
        confusion[query_cat] = {}
    for query_cat in confusion.keys():
        for pred_cat in cat_list:
            confusion[query_cat][pred_cat] = 0
    
    # get test confusion matrix
    for i in range(test_indices.shape[0]):
        true_category = categories[test_indices[i]]
        [indices, dists] = nn.nearest_neighbors(shape_reps[test_indices[i],:], k=num_nearest, return_indices=True)
        neighbor_cats = []
        for index in indices:
            neighbor_cats.append(categories[train_indices[index]])
        print 'Shape nearest neighbors', true_category, neighbor_cats

        if len(indices) > 0:
            confusion[true_category][neighbor_cats[0]] += 1
        else:
            confusion[true_category][UNKNOWN_TAG] += 1            

    # accumulate results
    # convert the dictionary to a numpy array
    row_names = confusion.keys()
    confusion_mat = np.zeros([len(row_names), len(row_names)])
    i = 0
    for query_cat in confusion.keys():
        j = 0
        for pred_cat in confusion.keys():
            confusion_mat[i,j] = confusion[query_cat][pred_cat]
            j += 1
        i += 1

    # get true positives, etc for each category
    num_preds = len(test_files)
    tp = np.diag(confusion_mat)
    fp = np.sum(confusion_mat, axis=0) - np.diag(confusion_mat)
    fn = np.sum(confusion_mat, axis=1) - np.diag(confusion_mat)
    tn = num_preds * np.ones(tp.shape) - tp - fp - fn

    # compute useful statistics
    recall = tp / (tp + fn)
    tnr = tn / (fp + tn)
    precision = tp / (tp + fp)
    npv = tn / (tn + fn)
    fpr = fp / (fp + tn)
    accuracy = np.sum(tp) / num_preds # correct predictions over entire dataset

    # remove nans
    recall[np.isnan(recall)] = 0
    tnr[np.isnan(tnr)] = 0
    precision[np.isnan(precision)] = 0
    npv[np.isnan(npv)] = 0
    fpr[np.isnan(fpr)] = 0

    IPython.embed()
    
    """
    X_t = z.transform(X)

    p = sd.PCA()
    p.fit(X)
    X_proj = p.transform(X)
    X_proj = X_proj[:,:2]
    X_t_proj = p.transform(X_t)
    X_t_proj = X_t_proj[:,:2]
    
    num_clusters = 10
    k = KMeansFeatureDictionary(num_clusters)
    k.fit(X_t)
    l_t = k.transform(X_t)

    colors = plt.get_cmap('jet')(np.linspace(0, 1.0, num_clusters))
    pointwise_colors = colors[l_t, :]

    plt.figure()
#    plt.scatter(X_proj[:,0], X_proj[:,1], c=u'b')
    plt.scatter(X_t_proj[:,0], X_t_proj[:,1], c=pointwise_colors)
    plt.show()
    """

def plot_cats_vs_all(shapes_tf, cat_array, cat_list, indices):
    objs = []
    labels = ['Others']
    plt.figure()

    valid_ind = np.ones(cat_array.shape).astype(np.bool)
    for index in indices:
        valid_ind = valid_ind & (cat_array != index)

    cat_ind = np.where(valid_ind)
    cat_ind = cat_ind[0]
    o = plt.scatter(shapes_tf[cat_ind,0], shapes_tf[cat_ind,1], s=50, c=u'b')
    objs.append(o)

    colors = plt.get_cmap('jet')(np.linspace(0.5, 1.0, len(indices)))
    i = 0
    for index in indices:
        cat_ind = np.where(cat_array == index)
        cat_ind = cat_ind[0]
        o = plt.scatter(shapes_tf[cat_ind,0], shapes_tf[cat_ind,1], s=50, c=colors[i])
        objs.append(o)
        labels.append(cat_list[index])
        i = i+1
    
    plt.legend(objs, labels)
    plt.show()    

if __name__ == '__main__':
    argc = len(sys.argv)
    feature_dir = sys.argv[1]
    zca_from_shot(feature_dir)

