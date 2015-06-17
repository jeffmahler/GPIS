"""
Sahaana Suri
"""

import numpy as np
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

import IPython

def string_to_array(string):
    """
    Takes in a string of space-separated values and returns an np-array of those values
    """
    return np.array([float(i) for i in string.split()])


class ShotFeatures:
    def __init__(self, shot_file_name, pts_file_name):
        """
        Takes in shot file name and obj file name, and loads:
            number of descriptors, length of each descriptor, length of rf, rfs, decriptors, points, normals
        Params:
            shot_file_name: string representing file with shot descriptors extracted with modified pcl code
            obj_file_name: string represinting original obj file
        """

        #working with the extracted shot features (extracted with c++ script and input as shot_file_name)
        HEADER_SIZE = 8 #size of obj file header
        shot_file = open(shot_file_name, 'r')

        self.num_descriptors = int(shot_file.readline())
        self.len_descriptors = int(shot_file.readline())
        self.len_rf = int(shot_file.readline())
       
        #initializing all matrices
        self.descriptors = np.zeros([self.num_descriptors, self.len_descriptors])#np.array(descriptor)
        self.points = np.zeros([self.num_descriptors, 3])#np.array(point)
        self.rfs = np.zeros([self.num_descriptors, self.len_rf])#np.array(rf)
        self.normals = np.zeros([self.num_descriptors, 3])#np.array(normal)

        #parsing through the lines of data from the shot file to fill out the matrices
        for i in range(self.num_descriptors):
            rf, descriptor, point, normal = [string_to_array(j) for j in shot_file.readline().split('\t')] 
            self.descriptors[i] = descriptor
            self.points[i] = point
            self.rfs[i] = rf
            self.normals[i] = normal
        shot_file.close()

        #determining indices of points with descriptors in original .obj/point list
        """
        opened_obj_file = open(obj_file_name, 'r')
        obj_file_lines = opened_obj_file.readlines()
        self.all_points = {} #mapping from point to index
        for i in range(HEADER_SIZE,len(obj_file_lines)):
            point_vector = obj_file_lines[i].split()[1:] #the obj was all vertices, so flag ignored
            point = tuple([float(coord) for coord in point_vector])
            self.all_points[point] = i - HEADER_SIZE
        
        opened_obj_file.close()
        """

        """
        all_points_arr = np.loadtxt(pts_file_name)
        self.all_points = {}
        for i in range(all_points_arr.shape[0]):
            self.all_points[tuple(all_points_arr[i,:])] = i 
        """
        self.all_points = np.loadtxt(pts_file_name)

    def get_index(self, point):
        inds = np.where(np.linalg.norm(point - self.all_points, axis=1) < 1e-4)
        if inds[0].shape[0] == 0:
            return -1
        return inds[0][0]

    def indices_with_descriptors(self):
        """
        Prints the sorted list of indices corresponding to the points that have descriptors
        """
        my_indices = []
        for i in self.points:
            my_indices.append(self.all_points[tuple(i)])
        for i in sorted(my_indices):
            print i

    def calc_closest_descriptors(self, other_model):
        """
        Brute force 1-NN of shot descriptors using cdist. Input self and another shot_features instance.
        Returns dictionary mapping of this model's points to the other_model's points, as well as matrix containing the points in 
        the other model closest to this model's keypoints.
        Params:
            other_model: shot_features object representing the object you want to match with this one

        Returns:
            matches: dictionary that maps the indices of this objects keypoints to the indices of the other_model's keypoints
            matched_points: numpy.array of the same dimensions as self.points (this model's keypoints), where row i contains the keypoint
                            in the other_model that is closest to keypoint i of this model
        """

        #calculate distance between this model's descriptors and each of the other_model's descriptors
        dists = spatial.distance.cdist(self.descriptors, other_model.descriptors) 
        
        #calculate the indices of the other_model that minimize the distance to the descriptors in this model
        my_closest_descriptors = dists.argmin(axis=1)
        other_closest_descriptors = dists.argmin(axis=0)
        matches = {}
        my_matched_points = np.zeros((0,3))
        other_matched_points = np.zeros((0,3))

        #calculate which points/indices the closest descriptors correspond to
        for i, j in enumerate(my_closest_descriptors):
            my_point_index = self.get_index(self.points[i])

            # for now, only keep correspondences that are a 2-way match
            if other_closest_descriptors[my_closest_descriptors[i]] == i:
                other_point_index = other_model.get_index(other_model.points[j])

                matches[my_point_index] = other_point_index
                my_matched_points = np.r_[my_matched_points, self.all_points[my_point_index:my_point_index+1, :]]
                other_matched_points = np.r_[other_matched_points, other_model.all_points[other_point_index:other_point_index+1, :]]
            else:
                matches[my_point_index] = -1

        return matches, my_matched_points, other_matched_points
    
    def calc_transformation(self, other_model, my_matched_points, other_matched_points):
        """
        Follows http://nghiaho.com/?page_id=671 to return the estimated transformation matrix between this model and the other_model, 
        given a matrix of keypoints from the other_model that correspond to those of this model.

        Params:
            other_model: shot_features object representing the object you want to match with this one
            matched_points: numpy.array of dimensions self.points.shape, and is the output from  calc_closest_descriptors(self, other_model).
                            Row i contains the keypoint in the other_model that is closest to keypoint i of this model

        Returns:
            tf: numpy.array containing the estimated transformation from this model to the other_model        
    
        """

        #calculate centroids
        my_centroid = np.mean(my_matched_points, axis=0)
        other_centroid = np.mean(other_matched_points, axis=0)
        
        #center the datasets
        N = my_matched_points.shape[0]
        my_centered_points = my_matched_points - np.tile(my_centroid, (N,1))
        other_centered_points = other_matched_points - np.tile(other_centroid, (N,1))

        #find the covariance matrix and finding the SVD
        H = np.dot(my_centered_points.T, other_centered_points)
        U, S, V = np.linalg.svd(H) #this decomposes H = USV, so V is "V.T"

        #calculate the rotation
        R = np.dot(V.T, U.T)
        
        #special case (reflection)
        if np.linalg.det(R) < 0:
                V[2,:] *= -1
                R = np.dot(V.T, U.T)
        
        #calculate the translation + concatenate the rotation and translation
        t = np.matrix(np.dot(-R, my_centroid) + other_centroid)
        tf = np.hstack([R, t.T])

        return tf
       

def test_feature_matching():
    a = ShotFeatures("data/test/features/pepper_orig_features.txt", "data/test/features/pepper_orig_pts.txt")
    b = ShotFeatures("data/test/features/pepper_tf_features.txt", "data/test/features/pepper_tf_pts.txt" )
    matches, my_matched_points, other_matched_points = a.calc_closest_descriptors(b)

    tf = a.calc_transformation(b, my_matched_points, other_matched_points)    
    tf_true = np.loadtxt("data/test/features/tf_true.txt", delimiter=" ")

    delta_tf = tf.dot(np.linalg.inv(tf_true))

    print 'Estimated TF'
    print tf

    print 'True TF'
    print tf_true

    print 'Delta TF'
    print delta_tf

if __name__ == '__main__':
    test_feature_matching()





            





            

