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

class PointGrasp(Grasp):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(g, axis, width, objj, num_samples):
        """ Creates a line of action, or list of grid points, from a point g in world coordinates on an object """
        pass

class ParallelJawPtGrasp3D(PointGrasp):
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

	def close_fingers(self, obj, vis = False):
            """
            Steps along grasp axis to find the locations of contact
            Params:
                obj - graspable object class
                num_samples - number of sample points between g1 and g2 to find contact points
                vis - (bool) whether or not to plot the shoe
            Returns:
                c1 - the point of contact for jaw 1 in sdf grid coordinates
                c2 - the point of contact for jaw 2 in sdf grid coordinates
            """
            # compute num samples to use based on sdf resolution
            grasp_width_grid = obj.sdf.transform_pt_world_to_grid(self.grasp_width_)
            num_samples = int(2 * grasp_width_grid) # at least 2 samples per grid

            # get grasp endpoints in sdf frame
            g1_world, g2_world = self.grasp_endpoints()
                
            # get line2 of action
            line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.grasp_width_, obj, num_samples)
            line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.grasp_width_, obj, num_samples)

            # find contacts
            if vis:
                obj.sdf.scatter()

            c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
            c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)
            if vis:
                ax = plt.gca(projection = '3d')
                ax.set_xlim3d(0, obj.sdf.dims_[0])
                ax.set_ylim3d(0, obj.sdf.dims_[1])
                ax.set_zlim3d(0, obj.sdf.dims_[2])
                plt.show()
                
            contacts_found = c1_found and c2_found
            return contacts_found, c1, c2
        
        @staticmethod
        def create_line_of_action(g, axis, width, obj, num_samples, convert_grid=True):
            """
            Creates a straight line of action from a given point and direction in world or grid coords
            Params:
                g - numpy 3 array of the start point
                axis - normalized numpy 3 array of grasp direction
                width - the grasp width
                num_samples - # discrete points along the line of action
                convert_grid - whether or not the points are specified in world coords
            Returns:
                line_of_action - list of numpy 3-arrays in grid coords to check surface contacts
            """
            line_of_action = [g + t * width * axis for t in np.linspace(0, 1, num = num_samples)]
            if convert_grid:
                line_of_action = [obj.sdf.transform_pt_world_to_grid(g) for g in line_of_action]
            return line_of_action

        @staticmethod
        def find_contact(line_of_action, obj, vis = True):
            """
            Find the point at which a point travelling along a given line of action hits a surface
            Params:
                line_of_action - list of np 3-arrays, the points visited as the fingers close
                obj - GraspableObject3D to check contacts on
                vis - whether or not to display the contact check (for debugging)
            Returns:
                contact_found - whether or not the point contacts the object surface
                pt_zc - np 3-array of surface along line of action (None if contact not found)
            """
            contact_found = False
            pt_zc = None
            num_pts = len(line_of_action)

            # step along line of action, get points on surface when possible
            i = 0
            while i < num_pts and not contact_found:
                pt_grid = line_of_action[i]

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
                        pt_after = line_of_action[i+1]
                        sdf_after = obj.sdf[pt_after]
                        pt_zc = sdf.find_zero_crossing_linear(pt_grid, sdf_here, pt_after, sdf_after)
                        
                    elif i == len(line_of_action) - 1:
                        pt_before = line_of_action[i-1]
                        sdf_before = obj.sdf[pt_before]
                        pt_zc = sdf.find_zero_crossing_linear(pt_grid, sdf_here, pt_after, sdf_after)

                    else:
                        # quadratic approximation to find actual zero crossing
                        pt_before = line_of_action[i-1]
                        sdf_before = obj.sdf[pt_before]
                        pt_after = line_of_action[i+1]
                        sdf_after = obj.sdf[pt_after]
                        pt_zc = sdf.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after, sdf_after)
                i = i+1

            # visualization
            if vis and contact_found:
                ax = plt.gca(projection = '3d')
                ax.scatter(pt_zc[0], pt_zc[1], pt_zc[2], s=80, c=u'g')

            return contact_found, pt_zc

        @staticmethod
        def grasp_center_from_endpoints(g1, g2):
            """ Grasp center from endpoints as np 3-arrays """
            grasp_center = (g1 + g2) / 2
            return grasp_center

        @staticmethod
        def grasp_axis_from_endpoints(g1, g2):
            """ Normalized axis of grasp from endpoints as np 3-arrays """
            grasp_axis = g2 - g1
            return grasp_axis / np.linalg.norm(grasp_axis)

        @staticmethod
        def grasp_from_contact_and_axis_on_grid(obj, grasp_c1_grid, grasp_axis, grasp_width_world, jaw_width_world = 0, vis = False):
            """
            Creates a grasp from a single contact point in grid coordinates and direction in grid coordinates
            Params:
                obj - GraspableObject3D
                grasp_c1_grid - contact point 1 in grid coords
                grasp_axis - normalized direction of the grasp
                grasp_width_world - grasp_width in world coords
                jaw_width_world - width of jaws in world coords
                vis - whether or not to visualize the grasp
            Returns:
                ParallelJawGrasp3D object
            """
            grasp_width_grid = obj.sdf.transform_pt_world_to_grid(grasp_width_world)
            num_samples = int(2 * grasp_width_grid) # at least 2 samples per grid
            g2 = grasp_c1_grid + grasp_width_grid * grasp_axis

            # get line of action
            line_of_action = ParallelJawPtGrasp3D.create_line_of_action(g2, -grasp_axis, grasp_width_grid, obj, num_samples,
                                                                        convert_grid = False)
            if vis:
                obj.sdf.scatter()
                ax = plt.gca(projection = '3d')
                ax.scatter(grasp_c1_grid[0] - grasp_axis[0], grasp_c1_grid[1] - grasp_axis[1], grasp_c1_grid[2] - grasp_axis[2], c=u'r')
                ax.scatter(grasp_c1_grid[0], grasp_c1_grid[1], grasp_c1_grid[2], s=80, c=u'g')

            contact_found, grasp_c2_grid = ParallelJawPtGrasp3D.find_contact(line_of_action, obj, vis = vis)
            if vis:
                ax.set_xlim3d(0, obj.sdf.dims_[0])
                ax.set_ylim3d(0, obj.sdf.dims_[1])
                ax.set_zlim3d(0, obj.sdf.dims_[2])
                plt.show()
            
            # compute contacts in world frame and create grasp
            c1_world = obj.sdf.transform_pt_grid_to_world(grasp_c1_grid)
            c2_world = obj.sdf.transform_pt_grid_to_world(grasp_c2_grid)

            grasp_center = ParallelJawPtGrasp3D.grasp_center_from_endpoints(c1_world, c2_world)
            grasp_axis = ParallelJawPtGrasp3D.grasp_axis_from_endpoints(c1_world, c2_world)
            return ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width_world, jaw_width_world)


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

def test_grasp_from_contacts():
    """ Should visually check for reasonable contacts (large green circles) """
    np.random.seed(100)
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf3 = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = sf3.read()
        
    # create point on sdf surface
    obj_3d = go.GraspableObject3D(sdf_3d)
    surf_pts, surf_sdf = obj_3d.sdf.surface_points()
    rand_pt_ind = np.random.choice(surf_pts.shape[0])
    rand_surf_pt = surf_pts[rand_pt_ind, :]

    # get grasp direction
    sdf_grad = obj_3d.sdf.gradients
    axis = np.array([-sdf_grad[0][rand_surf_pt[0], rand_surf_pt[1], rand_surf_pt[2]],
                      -sdf_grad[1][rand_surf_pt[0], rand_surf_pt[1], rand_surf_pt[2]],
                      -sdf_grad[2][rand_surf_pt[0], rand_surf_pt[1], rand_surf_pt[2]]])
    axis = axis / np.linalg.norm(axis)

    test_grasp_width = 1.3                    
    g = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(obj_3d, rand_surf_pt, axis, test_grasp_width, vis = True) 

if __name__ == '__main__':
    test_find_contacts()
    test_grasp_from_contacts()
