"""
Grasp class that implements gripper endpoints and grasp functions
Author: Nikhil Sharma & Jeff Mahler
"""
from abc import ABCMeta, abstractmethod

import logging
import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np
import IPython
import time

import graspable_object as go
import sdf_file as sf
import sdf
import similarity_tf as stf
import tfx

PR2_GRASP_OFFSET = np.array([-0.0375, 0, 0])

class Grasp:
    __metaclass__ = ABCMeta

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object """
        pass

    @abstractmethod
    def to_json(self):
        """ Converts a grasp to json """
        pass

class PointGrasp(Grasp):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(g, axis, width, obj, num_samples):
        """ Creates a line of action, or list of grid points, from a point g in world coordinates on an object """
        pass

    #NOTE: close_fingers must return success, array of contacts (one per column)

class ParallelJawPtGrasp3D(PointGrasp):
    def __init__(self, grasp_center, grasp_axis, grasp_width, jaw_width = 0, grasp_angle = 0,
                 tf = stf.SimilarityTransform3D(tfx.identity_tf(from_frame = 'world'), 1.0)):
        """
        Create a point grasp for parallel jaws with given center and width (relative to object)
        Params: (Note: all in meters!)
            grasp_center - numpy 3-array for center of grasp
            grasp_axis - numpy 3-array for grasp direction (should be normalized)
            grasp_width - how wide the jaws open in meters
            jaw_width - the width of the jaws tangent to the axis (0 means classical point grasp)
            grasp_angle - the angle of approach for parallel-jaw grippers (the 6th DOF for point grasps)
                wrt the orthogonal dir to the axis in the XY plane
            tf - the similarity tf of the grasp wrt the world
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
        self.approach_angle_ = grasp_angle
        self.tf_ = tf
        self.surface_info_ = {}

    @property
    def center(self):
        return self.center_
    @property
    def axis(self):
        return self.axis_
    @property
    def grasp_width(self):
        return self.grasp_width_
    @property
    def jaw_width(self):
        return self.jaw_width_
    @property
    def approach_angle(self):
        return self.approach_angle_
    @property
    def tf(self):
        return self.tf_

    def endpoints(self):
        return self.center_ - (self.grasp_width_ / 2.0) * self.axis_, self.center_ + (self.grasp_width_ / 2.0) * self.axis_,

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

    def close_fingers(self, obj, vis = False):
        """
        Steps along grasp axis to find the locations of contact
        Params:
            obj - graspable object class
            num_samples - number of sample points between g1 and g2 to find contact points
            vis - (bool) whether or not to plot the shoe
        Returns:
            c1 - the point of contact for jaw 1 in obj frame
            c2 - the point of contact for jaw 2 in obj frame
        """
        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.grasp_width_)
        num_samples = int(2 * grasp_width_grid) # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints()

        # get line2 of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.grasp_width_, obj, num_samples)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.grasp_width_, obj, num_samples)

        # find contacts
        if vis:
            plt.clf()
            h = plt.gcf()
            plt.ion()
            obj.sdf.scatter()

        c1_found, c1_world = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        c2_found, c2_world = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)
        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found, np.array([c1_world, c2_world])

    @staticmethod
    def create_line_of_action(g, axis, width, obj, num_samples, convert_grid=True):
        """
        Creates a straight line of action, or list of grid points, from a given
        point and direction in world or grid coords
        Params:
            g - numpy 3 array of the start point
            axis - normalized numpy 3 array of grasp direction
            width - the grasp width
            num_samples - # discrete points along the line of action
            convert_grid - whether or not the points are specified in world coords
        Returns:
            line_of_action - list of numpy 3-arrays in grid coords to check surface contacts
        """
        line_of_action = [g + t * axis for t in np.linspace(0, width, num = num_samples)]
        if convert_grid:
            as_array = np.array(line_of_action).T
            transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
            line_of_action = list(transformed.T)
        return line_of_action

    @staticmethod
    def find_contact(line_of_action, obj, vis = True, stop = False):
        """
        Find the point at which a point travelling along a given line of action hits a surface
        Params:
            line_of_action - list of np 3-arrays (grid coords), the points visited as the fingers close
            obj - GraspableObject3D to check contacts on
            vis - whether or not to display the contact check (for debugging)
        Returns:
            contact_found - whether or not the point contacts the object surface
            pt_zc - np 3-array of surface along line of action in obj frame(None if contact not found)
        """
        contact_found = False
        pt_zc = None
        pt_zc_world = None
        num_pts = len(line_of_action)
        sdf_here = 0
        sdf_before = 0
        pt_grid = None
        pt_before = None

        # step along line of action, get points on surface when possible
        i = 0
        while i < num_pts and not contact_found:
            # update loop vars
            pt_before_before = pt_before
            pt_before = pt_grid
            sdf_before_before = sdf_before
            sdf_before = sdf_here
            pt_grid = line_of_action[i]

            # visualize
            if vis:
                ax = plt.gca(projection = '3d')
                ax.scatter(pt_grid[0], pt_grid[1], pt_grid[2], c=u'r')

            # check surface point
            on_surface, sdf_here = obj.sdf.on_surface(pt_grid)
            if on_surface:
                contact_found = True

                # quadratic approximation to find actual zero crossing
                if i == 0:
                    pt_after = line_of_action[i+1]
                    sdf_after = obj.sdf[pt_after]
                    pt_after_after = line_of_action[i+2]
                    sdf_after_after = obj.sdf[pt_after_after]
                    pt_zc = sdf.find_zero_crossing_quadratic(pt_grid, sdf_here, pt_after, sdf_after, pt_after_after, sdf_after_after)

                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False


                elif i == len(line_of_action) - 1:
                    pt_zc = sdf.find_zero_crossing_quadratic(pt_before_before, sdf_before_before, pt_before, sdf_before, pt_grid, sdf_here)

                    if pt_zc is None:
                        contact_found = False

                else:
                    pt_after = line_of_action[i+1]
                    sdf_after = obj.sdf[pt_after]
                    pt_zc = sdf.find_zero_crossing_quadratic(pt_before, sdf_before, pt_grid, sdf_here, pt_after, sdf_after)

                    # contact not yet found if next sdf value is smaller
                    if pt_zc is None or np.abs(sdf_after) < np.abs(sdf_here):
                        contact_found = False

            i = i+1

        # visualization
        if vis and contact_found:
            ax = plt.gca(projection = '3d')
            ax.scatter(pt_zc[0], pt_zc[1], pt_zc[2], s=80, c=u'g')

        if contact_found:
            pt_zc_world = obj.sdf.transform_pt_grid_to_obj(pt_zc)
        return contact_found, pt_zc_world

    def transform(self, tf, theta_res = 0):
        """
        Generates a set of grasps in the given frame of reference.
        Since parallel-jaw grasps have 5 DOF we discretize the rotation about the orthogonal
        direction and return grasps approaching along those directions
        Params:
           tf - SimilarityTransform3D to apply to the grasp
           theta_res - The angle resolution for equivalent rotations (defaults to zero, meaning only transform grasp)
        Returns:
           grasps_tf - (list of ParallelJawPtGrasp3D) list of grasps with the given transformation
        """
        # transform grasp to object basis
        grasp_center_obj = tf.apply(self.center_)
        grasp_axis_y_obj = tf.apply(self.axis_, direction=True)
        grasp_width_obj = tf.apply(self.grasp_width_)

        # store all identifications of the grasp around the rotational axes
        theta = 0
        grasps_tf = []
        while theta <= 2*np.pi - theta_res:
            grasps_tf.append(ParallelJawPtGrasp3D(grasp_center_obj, grasp_axis_y_obj, grasp_width_obj, self.jaw_width, theta, tf))
            theta = theta + theta_res
        return grasps_tf

    def gripper_pose(self, R_gripper_center = np.eye(3), t_gripper_center = PR2_GRASP_OFFSET):
        """
        Convert a grasp to a gripper pose in SE(3) using PR2 gripper_l_tool_frame as default
        Params:
           R_gripper_center: (numpy 3x3 array) rotation matrix from grasp basis to gripper basis
           t_gripper_center: (numpy 3 array) translation from grasp basis to gripper basis
        Returns:
           pose_gripper: (tfx transform) pose of gripper in the grasp frame
        """
        # convert gripper orientation to rotation matrix
        grasp_axis_y = self.axis_
        grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
        grasp_axis_x = grasp_axis_x / np.linalg.norm(grasp_axis_x)
        grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)

        R_center_ref = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        pose_center_ref = tfx.transform(R_center_ref, self.center_) # pose of grasp center in its reference frame

        # rotate along grasp approach angle
        R_center_rot_center = np.array([[ np.cos(self.approach_angle_), 0, np.sin(self.approach_angle_)],
                                        [                            0, 1,                            0],
                                        [-np.sin(self.approach_angle_), 0, np.cos(self.approach_angle_)]])
        pose_center_rot_center = tfx.transform(R_center_rot_center, np.zeros(3))
        pose_gripper_center_rot = tfx.transform(R_gripper_center, t_gripper_center)

        pose_gripper_ref = pose_center_ref.apply(pose_center_rot_center).apply(pose_gripper_center_rot)
        return pose_gripper_ref

        """
        R_center_grasp_rot = R_grasp_rot_grasp.dot(R_center_grasp)

        # rotation and translation in same frame as grasp
        R_gripper = R_gripper_center.dot(R_center_grasp_rot)
        t_gripper = self.center_ + R_center_grasp_rot.dot(t_gripper_center)

        # add new grasp with given pose in object basis
        pose_gripper = tfx.transform(R_gripper, t_gripper) #TODO: add correct frame
        return pose_gripper
        """

    @staticmethod
    def grasp_from_contact_and_axis_on_grid(obj, grasp_c1_world, grasp_axis_world, grasp_width_world, jaw_width_world = 0, vis = False, stop = False):
        """
        Creates a grasp from a single contact point in grid coordinates and direction in grid coordinates
        Params:
            obj - GraspableObject3D
            grasp_c1_grid - contact point 1 in world
            grasp_axis - normalized direction of the grasp in world
            grasp_width_world - grasp_width in world coords
            jaw_width_world - width of jaws in world coords
            vis - whether or not to visualize the grasp
        Returns:
            ParallelJawGrasp3D object
            numpy 3 array of 2nd contact on object
        """
        # transform to grid basis
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp_width_world)
        grasp_c1_grid = obj.sdf.transform_pt_obj_to_grid(grasp_c1_world) - (grasp_width_grid / 8) * grasp_axis_grid # subtract to find true point
        num_samples = int(2 * grasp_width_grid) # at least 2 samples per grid
        g2 = grasp_c1_grid + grasp_width_grid * grasp_axis_grid

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(grasp_c1_grid, grasp_axis_grid, grasp_width_grid, obj, num_samples,
                                                                     convert_grid = False)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2, -grasp_axis_grid, grasp_width_grid, obj, num_samples,
                                                                     convert_grid = False)
        if vis:
            obj.sdf.scatter()
            ax = plt.gca(projection = '3d')
            ax.scatter(grasp_c1_grid[0] - grasp_axis_grid[0], grasp_c1_grid[1] - grasp_axis_grid[1], grasp_c1_grid[2] - grasp_axis_grid[2], c=u'r')
            ax.scatter(grasp_c1_grid[0], grasp_c1_grid[1], grasp_c1_grid[2], s=80, c=u'b')

        # compute the contact points on the object
        contact1_found, c1_world = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis = vis, stop = stop)
        contact2_found, c2_world = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis = vis, stop = stop)

        if vis:
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()
        if not contact1_found or not contact2_found:
            logging.debug('No contacts found for grasp')
            return None, None, None

        # create grasp
        grasp_center = ParallelJawPtGrasp3D.grasp_center_from_endpoints(c1_world, c2_world)
        grasp_axis = ParallelJawPtGrasp3D.grasp_axis_from_endpoints(c1_world, c2_world)
        return ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width_world, jaw_width_world, grasp_angle=0, tf=obj.tf), c1_world, c2_world # relative to object

    def visualize(self, obj, arrow_len = 0.01, line_width = 20.0):
        """ Display point grasp as arrows on the contact points of the mesh """
        contacts_found, contacts = self.close_fingers(obj)

        if contacts_found:
            c1_world = contacts[0,:]
            c2_world = contacts[1,:]
            v = c2_world - c1_world
            v = arrow_len * v / np.linalg.norm(v)
            mv.quiver3d(c1_world[0] - v[0], c1_world[1] - v[1], c1_world[2] - v[2], v[0], v[1], v[2], scale_factor=1.0,
                        mode='arrow', line_width=line_width)
            mv.quiver3d(c2_world[0] + v[0], c2_world[1] + v[1], c2_world[2] + v[2], -v[0], -v[1], -v[2], scale_factor=1.0,
                        mode='arrow', line_width=line_width)

    def surface_information(self, graspable, width=2e-2, num_steps=21):
        """Return the surface information at the contacts that this grasp makes
        on a graspable.
        Params:
            graspable - GraspableObject3D instance
            width - float width of the window in obj frame
            num_steps - int number of steps
        Returns:
            list of windows, one for each point of contact
        """
        if graspable not in self.surface_info_:
            info = graspable.surface_information(self, width, num_steps)
            self.surface_info_[graspable] = info
        return self.surface_info_[graspable]

    def to_json(self, quality=0, method='PFC'):
        """Converts the grasp to a Python dictionary for serialization to JSON."""
        gripper_pose = self.gripper_pose()
        gripper_position = gripper_pose.position
        gripper_orientation = gripper_pose.orientation
        return {
            'flag': 0, # what's this?
            'gripper_width': self.grasp_width,
            'jaw_width': self.jaw_width,
            'gripper_pose': {
                'position': {
                    'x': gripper_position.x,
                    'y': gripper_position.y,
                    'z': gripper_position.z,
                },
                'orientation': {
                    'w': gripper_orientation.w,
                    'x': gripper_orientation.x,
                    'y': gripper_orientation.y,
                    'z': gripper_orientation.z,
                }
            },
            'frame': 'gripper_l_tool_frame', # ?
            'reference_frame': 'object',
            'quality': quality,
            'metric': method,
        }

class ParallelJawPtPose3D(object):
    """A skeleton class that exposes the same attributes as a tfx.transform."""
    def __init__(self, data):
        self.json_ = data
        gripper_pose = data['gripper_pose']
        orientation = gripper_pose['orientation']
        position = gripper_pose['position']
        self.orientation_ = tfx.rotation([orientation[c] for c in 'xyzw'])
        self.position_ = tfx.point([position[c] for c in 'xyz'])
        self.gripper_pose_ = tfx.pose(self.orientation_, self.position_)
        self.surface_info_ = {}

    @staticmethod
    def from_json(data):
        return ParallelJawPtPose3D(data)

    def to_json(self):
        return self.json_

    def gripper_pose(self, R_gripper_center=np.eye(3), t_gripper_center=PR2_GRASP_OFFSET):
        return self.gripper_pose_

    surface_information = ParallelJawPtGrasp3D.surface_information

def test_find_contacts():
    """ Should visually check for reasonable contacts (large green circles) """
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf3 = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = sf3.read()

    # create grasp
    plt.figure()
    test_grasp_center = np.zeros(3)
    test_grasp_axis = np.array([1, 0, 0])
    test_grasp_width = 1.0
    obj_3d = go.GraspableObject3D(sdf_3d)
    grasp = ParallelJawPtGrasp3D(test_grasp_center, test_grasp_axis, test_grasp_width)
    contact_found, contacts = grasp.close_fingers(obj_3d, vis=True)
    plt.ioff()
    plt.show()

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
    rand_surf_pt_grid = surf_pts[rand_pt_ind, :]
    rand_surf_pt = obj_3d.sdf.transform_pt_grid_to_obj(rand_surf_pt_grid)

    # get grasp direction
    axis = -obj_3d.sdf.gradient(rand_surf_pt)
    axis = axis / np.linalg.norm(axis)
    axis = obj_3d.sdf.transform_pt_grid_to_obj(axis, direction=True)

    plt.figure()
    test_grasp_width = 0.8
    ax = plt.gca(projection = '3d')
    ax.scatter(rand_surf_pt_grid[0], rand_surf_pt_grid[1], rand_surf_pt_grid[2], s=80, c=u'g')
    g, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(obj_3d, rand_surf_pt, axis, test_grasp_width, vis = True)
    plt.show()

def test_to_json():
    """ Should visually check for reasonable contacts (large green circles) """
    np.random.seed(100)
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf3 = sf.SdfFile(sdf_3d_file_name)
    sdf_3d = sf3.read()

    # create point on sdf surface
    obj_3d = go.GraspableObject3D(sdf_3d)
    surf_pts, surf_sdf = obj_3d.sdf.surface_points()
    rand_pt_ind = np.random.choice(surf_pts.shape[0])
    rand_surf_pt_grid = surf_pts[rand_pt_ind, :]
    rand_surf_pt = obj_3d.sdf.transform_pt_grid_to_obj(rand_surf_pt_grid)

    # get grasp direction
    axis = -obj_3d.sdf.gradient(rand_surf_pt)
    axis = axis / np.linalg.norm(axis)
    axis = obj_3d.sdf.transform_pt_grid_to_obj(axis, direction=True)

    test_grasp_width = 0.8
    g, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(obj_3d, rand_surf_pt, axis, test_grasp_width, vis = False)
    j = g.to_json()

    # TODO: hard checks

if __name__ == '__main__':
    test_find_contacts()
#    test_grasp_from_contacts()
#    test_to_json()
