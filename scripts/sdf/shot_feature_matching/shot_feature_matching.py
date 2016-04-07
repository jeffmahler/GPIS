"""
Sahaana Suri
"""

import numpy as np
from scipy import spatial
from sklearn.neighbors import NearestNeighbors

RUN_EXAMPLE = True

def string_to_array(string):
    """
    Takes in a string of space-separated values and returns an np-array of those values
    """
    return np.array([float(i) for i in string.split()])


class shot_features:
    def __init__(self, shot_file_name, obj_file_name):
        """
        Takes in shot file name and obj file name, and loads:
            number of descriptors, length of each descriptor, length of rf, rfs, decriptors, points, normals
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
        opened_obj_file = open(obj_file_name, 'r')
        obj_file_lines = opened_obj_file.readlines()
        self.all_points = {} #mapping from point to index
        for i in range(HEADER_SIZE,len(obj_file_lines)):
            point_vector = obj_file_lines[i].split()[1:] #the obj was all vertices, so flag ignored
            point = tuple([float(coord) for coord in point_vector])
            self.all_points[point] = i - HEADER_SIZE
        
        opened_obj_file.close()

    def calc_closest_descriptors(self, other_model):
        """
        Brute force 1-NN calculation using cdist. Input self and another shot_features instance.
        Returns dictionary mapping of this model's points to the other_model's points, as well as matrix containing the points in 
        the other model closest to this model's keypoints.
        """
        #calculate distance between this model's descriptors and each of the other_model's descriptors
        dists = spatial.distance.cdist(self.descriptors, other_model.descriptors) 
        
        #calculate the indices of the other_model that minimize the distance to the descriptors in this model
        closest_descriptors = dists.argmin(1)
        matches = {}
        matched_points = np.zeros(self.points.shape)

        #calculate which points/indices the closest descriptors correspond to
        for i,j in enumerate(closest_descriptors):
            my_point_index = self.all_points[tuple(self.points[i])]
            other_point_index = other_model.all_points[tuple(other_model.points[j])]

            matches[my_point_index] = other_point_index
            matched_points[i] = other_model.points[j]
    
        return matches, matched_points

    def indices_with_descriptors(self):
        """
        Prints the sorted list of indices corresponding to the points that have descriptors
        """
        my_indices = []
        for i in self.points:
            my_indices.append(self.all_points[tuple(i)])
        for i in sorted(my_indices):
            print i

    
    def calc_transformation(self, other_model, matched_points):
        """
        Follows http://nghiaho.com/?page_id=671 to return the estimated transformation matrix between this model and the other_model, 
        given a matrix of keypoints from the other_model that correspond to those of this model.
        """

        #calculate centroids
        my_centroid = np.mean(self.points,0)
        other_centroid = np.mean(matched_points,0)
        
        #center the datasets
        N = self.points.shape[0]
        my_centered_points = self.points - np.tile(my_centroid, (N,1))
        other_centered_points = matched_points - np.tile(other_centroid, (N,1))

        #find the covariance matrix and finding the SVD
        H = np.dot(my_centered_points.T, other_centered_points)
        U, S, V = np.linalg.svd(H) #this decomposes H = USV, so V is "V.T"

        #calculate the rotation
        R = np.dot(V.T, U.T)
        
        #special case (reflection)
        if np.linalg.det(R) < 0:
                V[2,:] *= -1
                R = np.dot(V.T, U.T)
        
        #calculate the translation
        t = np.matrix(np.dot(-R,my_centroid) + other_centroid)

        #concatenate the rotation and translation
        return np.hstack([R, t.T])
       

def run_example():
    a = shot_features("extracted_shot_test2.txt", "pcl_test2.obj")
    b = shot_features("extracted_shot_test2_tf.txt", "pcl_test2_tf.obj" )
    matches, matched_points = a.calc_closest_descriptors(b)
    tf = a.calc_transformation(b, matched_points)
    print tf

if RUN_EXAMPLE == True:
    run_example()





            






"""
    def knn(self, other_model, k):
            model = NearestNeighbors(n_neighbors = k)
            model.fit(other_model.descriptors)#, [i for i in range(other_model.num_descriptors)])
            closest_indices_in_descriptors = model.kneighbors(self.descriptors, return_distance=False)
            closest_points = [ tuple(other_model.points[i]) for i in closest_indices_in_descriptors]
            for i,j in enumerate(closest_points):
                    my_point_index = self.all_points[tuple(self.points[i])]
                    other_point_indices = [other_model.all_points[tuple(k)] for k in j]
                    self.matches[my_point_index] = other_point_indices
            return self.matches
"""


            

