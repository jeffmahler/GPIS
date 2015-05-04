"""
Grasp class that implements gripper endpoints and grasp functions
Author: Nikhil Sharma
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import IPython

import graspable_object as go
import sdf_file as sf
import sdf

class Grasp:
    __metaclass__ = ABCMeta

    @abstractmethod
    def find_contacts(self, obj):
        '''
        Finds the contact points on an object
        '''
        pass

    #@abstractmethod
    def to_json(self):
        '''
        Converts a grasp to json
        '''
        return None

class ParallelJawPtGrasp2D(Grasp):
	"""A grasp possesses gripper endpoints g1 and g2"""
	import numpy as np

	def __init__(self, g1, g2):
		"""
		Create a Grasp with gripper endpoints g1 and g2

		g1, g2 - represented as Python array with 3 elements
		representing x, y, z coordinates respectively

		If in 2D, third element of g1, g2 are simply 0
		"""
		self.g1 = g1
		self.g2 = g2

	def find_contacts(self, obj, num_samples=500):
		"""
		Steps along grasp directions to find the locations of contact

		obj - graspable object class
		num_samples - number of sample points between g1 and g2 to find contact points
		"""
                sdf = obj.sdf
		axis = self.grasp_axis()
		sample_points = [[self.g1[0] + axis[0]*t, self.g1[1] + axis[1]*t, self.g1[2] + axis2*t] for x in list(np.linspace(0, 1, num_samples))]
		contact_found = False
		for i in range(len(sample_points) - 1):
			sdf1 = sdf[sample_points[i][0], sample_points[i][1], sample_points[i][2]]
			sdf2 = sdf[sample_points[i+1][0], sample_points[i+1][1], sample_points[i+1][2]]
			if (sdf1 < 0 and sdf2 >= 0):
				contact_found = True
				contact_bounds.append([[i / num_samples, sdf1], [(i + 1) / num_samples, sdf2]])
				if (len(contact_bounds) == 2):
					break
		if (contact_found):
			contact_points = []
			for bound in contact_bounds:
				m = (bound[1][1] - bound[0][1]) * num_samples
				b = bound[0][1] - m*bound[0][0]
				contact_points.append(-b/m)
			return contact_points
		else:
			return "Contact point not found."

	def grasp_axis(self):
		"""Returns g2 - g1 (the direction the jaws travel)"""
		return [self.g2[x] - self.g1[x] for x in range(len(self.g1))]

	def grasp_center(self):
		"""Returns (g1 + g2) / 2 (the centroid of the grasp points)"""
		return [(self.g1[x] + self.g2[x]) / 2 for x in range(len(self.g1))]

if __name__ == '__main__':
	sdf_2d_file_name = 'data/test/sdf/brine_mini_soccer_ball_optimized_poisson_texture_mapped_mesh_clean_0.csv'
	sf2 = sf.SdfFile(sdf_2d_file_name)
	sdf_2d = sf2.read()
        obj_2d = go.GraspableObject2D(sdf_2d)
	grasp = ParallelJawPtGrasp2D([25, 0, 0], [25, 49, 0])
	grasp.find_contacts(obj_2d)

