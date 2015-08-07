import time

from scipy.sparse import csc_matrix as sp_mat
from scipy.sparse import vstack
from sets import Set

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
import numpy as np
from numpy import argmin
from sklearn.decomposition import TruncatedSVD
from scipy.spatial import distance

from feature_database import FeatureDatabase
from feature_object import FeatureObject
from feature_object import FeatureVectorDatabase
from dataset_sorter import DatasetSorter


if __name__ == '__main__':
	feature_db = FeatureDatabase()

	distances_file = open('/Users/MelRod/Desktop/distances.txt', 'r')

	sorted_keys = feature_db.mesh_database().sorted_keys()
	length = len(sorted_keys)
	distance_mat = np.zeros((length, length))

	print 'CREATING MATRIX'
	lines = distances_file.readlines()
	index = 0
	for r in range(0, length):
		for c in range(0, length):
			distance_mat[r,c] = float(lines[index])
			index += 1
		if r%1000 == 0:
			print r

	print 'CREATING ACCURACY'
	num_correct = 0
	rows, cols = distance_mat.shape
	for r in range(0, rows):
		min_dist = 100000
		min_index = -1
		for c in range(0, cols):
			if r == c:
				continue
			dist = distance_mat[r, c]
			if dist < min_dist:
				min_dist = dist
				min_index = c
		r_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[r])
		nearest_cat = feature_db.mesh_database().object_category_for_key(sorted_keys[min_index])
		print sorted_keys[r]+'--'+r_cat+', '+sorted_keys[min_index]+'--'+nearest_cat+': '+str(min_dist)
		if r_cat == nearest_cat:
			num_correct += 1
		# print r

	print 'CORRECT: '+str(num_correct)
	print 'TOTAL: '+str(len(sorted_keys))
	print 'ACCURACY: '+str(num_correct/len(sorted_keys))
	import IPython; IPython.embed()
