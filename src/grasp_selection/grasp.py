"""
Grasp class that implements gripper endpoints and grasp functions
Author: Nikhil Sharma
"""
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import IPython

import graspable_object as go
import sdf_file as sf
import sdf

class Grasp:
    __metaclass__ = ABCMeta

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object """
        pass

    #@abstractmethod
    def to_json(self):
        """ Converts a grasp to json """
        return None

class ParallelJawPtGrasp3D(Grasp):
	"""A grasp possesses gripper endpoints g1 and g2"""
	import numpy as np

	def __init__(self, grasp_center, grasp_axis, grasp_width, jaw_width = 0):
		"""
		Create a point grasp for paralell jaws with given center and width
                Params: (Note: all in meters!)
		    grasp_center - numpy 3-array for center of grasp
		    grasp_axis - numpy 3-array for grasp direction (should be normalized)
                    grasp_width - how wide the jaws open in meters
                    jaw_width - the width of the jaws tangent to the axis (0 means classical point grasp)
		"""
                if not isinstance(grasp_center, np.ndarray) or grasp_center.shape[0] != 3:
                    raise ValueError('Center must be numpy ndarray of size 3')
                if not isinstance(grasp_axis, np.ndarray)  or grasp_axis.shape[0] != 3:
                    raise ValueError('Axis must be numpy ndarray of size 3')
                if jaw_width != 0:
                    raise ValueError('Nonzero jaw width not yet supported')

                self.center_ = grasp_center
                self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
                self.grasp_width_ = grasp_width
                self.jaw_width_ = jaw_width

        @property
        def center(self):
            return center_
        @property
        def axis(self):
            return axis_
        @property
        def grasp_width(self):
            return grasp_width_
        @property
        def jaw_width(self):
            return jaw_width_
        def grasp_endpoints(self):
            return self.center_ - (self.grasp_width_ / 2.0) * self.axis_, self.center_ + (self.grasp_width_ / 2.0) * self.axis_,

	def close_fingers(self, obj, vis = True):
		"""
		Steps along grasp axis to find the locations of contact
                Params:
		    obj - graspable object class
		    num_samples - number of sample points between g1 and g2 to find contact points
                    vis - (bool) whether or not to plot the shoe
                Returns:
                    c1 - the point of contact for jaw 1
                    c2 - the point of contact for jaw 2
                """
                # compute num samples to use based on sdf resolution
                grasp_width_grid = self.grasp_width_ / obj.sdf.resolution
                num_samples = int(2 * grasp_width_grid) # at least 2 samples per grid

                # get grasp endpoints in sdf frame
                g1_world, g2_world = self.grasp_endpoints()
                
                # get line2 of action
                line_of_action1 = [g1_world + t * self.grasp_width_ * self.axis_ for t in np.linspace(0, 1, num = num_samples)]
                line_of_action2 = [g2_world - t * self.grasp_width_ * self.axis_ for t in np.linspace(0, 1, num = num_samples)]

                # find contacts
                if vis:
                    obj.sdf.scatter()

                c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
                c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)
                if vis:
                    plt.show()
                
                contacts_found = c1_found and c2_found
                return contacts_found, c1, c2

        @staticmethod
        def find_contact(line_of_action, obj, vis = True):
		contact_found = False
                pt_zc = None
                num_pts = len(line_of_action)

                # step along line of action, get points on surface when possible
                i = 0
                while i < num_pts and not contact_found:
                    pt = line_of_action[i]
                    pt_grid = obj.sdf.transform_grid_basis(pt)

                    # visualize
                    if vis:
                        ax = plt.gca(projection = '3d')
                        ax.scatter(pt_grid[0], pt_grid[1], pt_grid[2], c=u'r')
                    
                    # check surface point
                    on_surface, sdf_here = obj.sdf.on_surface(pt_grid)
                    if on_surface:
                        contact_found = True

                        # linear approx if near endpoints of line of action
                        if i == 0:
                            pt_after = obj.sdf.transform_grid_basis(line_of_action[i+1])
                            sdf_after = obj.sdf[pt_after]
                            pt_zc = sdf.find_zero_crossing_linear(pt_grid, sdf_here, pt_after, sdf_after)

                        elif i == len(line_of_action) - 1:
                            pt_before = obj.sdf.transform_grid_basis(line_of_action[i-1])
                            sdf_before = obj.sdf[pt_before]
                            pt_zc = sdf.find_zero_crossing_linear(pt_grid, sdf_here, pt_after, sdf_after)

                        else:
                            # quadratic approximation to find actual zero crossing
                            pt_before = obj.sdf.transform_grid_basis(line_of_action[i-1])
                            sdf_before = obj.sdf[pt_before]
                            pt_after = obj.sdf.transform_grid_basis(line_of_action[i+1])
                            sdf_after = obj.sdf[pt_after]
                            pt_zc = sdf.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after, sdf_after)
            
                    i = i+1

                # visualization
                if vis and contact_found:
                    ax = plt.gca(projection = '3d')
                    ax.scatter(pt_zc[0], pt_zc[1], pt_zc[2], s=80, c=u'g')
                    ax.set_xlim3d(0, obj.sdf.dims_[0])
                    ax.set_ylim3d(0, obj.sdf.dims_[1])
                    ax.set_zlim3d(0, obj.sdf.dims_[2])

                return contact_found, pt_zc

def test_find_contacts():
    """ Should visually check for reasonable contacts (large green circles) """
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf3 = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = sf3.read()
        
    # create grasp
    test_grasp_center = np.zeros(3)
    test_grasp_axis = np.array([1, 0, 0])
    test_grasp_width = 1.0
    obj_3d = go.GraspableObject3D(sdf_3d)
    grasp = ParallelJawPtGrasp3D(test_grasp_center, test_grasp_axis, test_grasp_width)
    contact_found, c1, c2 = grasp.close_fingers(obj_3d, vis=True)
    
    assert(contact_found)

if __name__ == '__main__':
    test_find_contacts()
