"""
Grasp class that implements gripper endpoints and grasp functions
Author: Nikhil Sharma & Jeff Mahler
"""
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import logging
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import numpy as np
from numpy.linalg import inv, norm
import IPython
import sys
import time

sys.path.append('/home/jmahler/jeff_working/GPIS/src/grasp_selection/control/DexControls/')

from DexNumericSolvers import DexNumericSolvers
import contacts
import graspable_object as go
import sdf_file as sf
import sdf
import similarity_tf as stf
import tfx

PR2_GRASP_OFFSET = np.array([-0.05, 0, 0])

class Grasp:
    __metaclass__ = ABCMeta
    samples_per_grid = 2 # global resolution for line of action

    @abstractmethod
    def close_fingers(self, obj):
        """ Finds the contact points by closing on the given object """
        pass

    @abstractmethod
    def configuration(self):
        """ Returns a numpy array representing the hand configuration """
        pass

    @abstractmethod
    def frame(self):
        """ Returns the string name of the grasp reference frame  """
        pass

    @abstractmethod
    def params_from_configuration(configuration):
        """ Convert configuration vector to a set of params for the class """
        pass

    @abstractmethod
    def configuration_from_params(*params):
        """ Convert param list to a configuration vector for the class """
        pass

class PointGrasp(Grasp):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create_line_of_action(g, axis, width, obj, num_samples):
        """ Creates a line of action, or list of grid points, from a point g in world coordinates on an object """
        pass

    #NOTE: implementation of close_fingers must return success, array of contacts (one per column)    

class SoftHandGrasp(Grasp):
    def __init__(self, configuration, grasp_id=None):
        center, palm_axis = SoftHandGrasp.params_from_configuration(configuration)
        self.center = center
        self.palm_axis = palm_axis
        self.grasp_id = grasp_id

    def gripper_transform(self, gripper=None):
        """ Returns the transformation from the object frame to the gripper frame as a similarity transform 3D object """
        t_obj_gripper = self.center
        y_obj_gripper = self.palm_axis
        x_obj_gripper = np.array([y_obj_gripper[1], -y_obj_gripper[0], 0])
        if norm(x_obj_gripper) == 0:
            x_obj_gripper = np.array([y_obj_gripper[2], 0, -y_obj_gripper[0]])
        x_obj_gripper = x_obj_gripper / np.linalg.norm(x_obj_gripper)
        z_obj_gripper = np.cross(x_obj_gripper, y_obj_gripper)
        R_obj_gripper = np.c_[x_obj_gripper, np.c_[y_obj_gripper, z_obj_gripper]]
        return stf.SimilarityTransform3D(pose=tfx.pose(R_obj_gripper, t_obj_gripper),
                                         from_frame='gripper', to_frame='obj').inverse()

    def configuration(self):
        return self.T_obj_gripper_

    @staticmethod
    def params_from_configuration(configuration):
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 6):
            raise ValueError('Configuration must be numpy ndarray of size 6')
        return configuration[:3], configuration[3:]

    @staticmethod
    def configuration_from_params(center, palm_axis):
        configuration = np.zeros(6)
        configuration[:3] = center
        configuration[3:] = palm_axis
        return configuration

    def close_fingers(obj):
        """ Not valid for softhand right now """
        return None

    def frame(self):
        return 'obj'

class ParallelJawPtGrasp3D(PointGrasp):

    def __init__(self, configuration, frame='object', timestamp=None, grasp_id=None):
        """
        Create a point grasp for parallel jaws with given center and width (relative to object)
        Params: (Note: all in meters!)
           configuration: numpy array specifying the configuration of the hand as follows:
                          [grasp_center, grasp_axis, grasp_angle, grasp_width, jaw_width]
           frame: string specifying the frame of reference for the object
           timestamp : string specifying the timestamp of when the grasp was created (for database purposes)
           grasp_id: string key to index the grasp in the database
        """
        # get parameters from configuration array
        grasp_center, grasp_axis, grasp_width, grasp_angle, jaw_width, min_grasp_width = \
            ParallelJawPtGrasp3D.params_from_configuration(configuration)

        if jaw_width != 0:
            raise ValueError('Nonzero jaw width not yet supported')

        self.center_ = grasp_center
        self.axis_ = grasp_axis / np.linalg.norm(grasp_axis)
        self.grasp_width_ = grasp_width
        self.jaw_width_ = jaw_width
        self.min_grasp_width_ = min_grasp_width
        self.approach_angle_ = grasp_angle
        self.frame_ = frame
        self.timestamp_ = timestamp
        self.grasp_id_ = grasp_id

        self.unrotated_full_axis_ = None
        self.rotated_full_axis_ = None

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
    def min_grasp_width(self):
        return self.min_grasp_width_
    @property
    def jaw_width(self):
        return self.jaw_width_
    @property
    def approach_angle(self):
        return self.approach_angle_
    @property
    def configuration(self):
        return ParallelJawPtGrasp3D.configuration_from_params(self.center_, self.axis_, self.grasp_width_, self.approach_angle_, self.jaw_width_)
    @property
    def frame(self):
        return self.frame_
    @property
    def grasp_id(self):
        return self.grasp_id_
    @property
    def unrotated_full_axis(self):
        if self.unrotated_full_axis_ is None:
            grasp_axis_y = self.axis
            grasp_axis_x = np.array([grasp_axis_y[1], -grasp_axis_y[0], 0])
            if norm(grasp_axis_x) == 0:
                grasp_axis_x = np.array([grasp_axis_y[2], 0, -grasp_axis_y[0]])
            grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
            grasp_axis_z = np.cross(grasp_axis_x, grasp_axis_y)
            
            self.unrotated_full_axis_ = np.c_[grasp_axis_x, np.c_[grasp_axis_y, grasp_axis_z]]
        return self.unrotated_full_axis_
    
    @property
    def rotated_full_axis(self):
        if self.rotated_full_axis_ is None:
            R = ParallelJawPtGrasp3D._get_rotation_matrix_y(self.approach_angle)
            self.rotated_full_axis_ = self.unrotated_full_axis.dot(R)
        return self.rotated_full_axis_

    def gripper_transform(self, gripper=None):
        """ Returns the transformation from the object frame to the gripper frame as a similarity transform 3D object """
        R_grip_p_grip = self.rotated_full_axis
        R_grip_robot = np.eye(3)
        if gripper is not None:
            R_grip_robot = gripper.T_grasp_gripper.rotation
        R = R_grip_p_grip.dot(R_grip_robot)
        t = self.center
        try:
            return stf.SimilarityTransform3D(pose=tfx.pose(R, t), from_frame='gripper', to_frame='obj').inverse()
        except:
            pose = tfx.pose(R,t)
            print 'R', R
            print 't', t
            IPython.embed()

    def angle_with_table(self, stable_pose):
       def _angle_2d(u, v):
            u_norm = u / norm(u)
            R = np.array([[u_norm[0], u_norm[1]],
                          [-u_norm[1], u_norm[0]]])
            vp = R.dot(v)

            #returns angle between 2 vectors in degrees
            theta = DexNumericSolvers.get_cartesian_angle(vp[0], vp[1])
            return theta

       grasp_axes_obj = self.rotated_full_axis
       R_stp_obj = stable_pose.r
       grasp_axes_stp = R_stp_obj.dot(grasp_axes_obj)
       y_axis_stp = grasp_axes_stp[:,1]
       y_axis_stp_proj = y_axis_stp.copy()
       y_axis_stp_proj[2] = 0
       y_axis_stp = grasp_axes_stp.T.dot(y_axis_stp)
       y_axis_stp_proj = grasp_axes_stp.T.dot(y_axis_stp_proj)

       u_x = np.array([y_axis_stp[1], y_axis_stp[2]])
       v_x = np.array([y_axis_stp_proj[1], y_axis_stp_proj[2]])
       psi = _angle_2d(v_x, u_x)

       if psi > np.pi:
           psi = 2 * np.pi - psi

       return psi
        
    def set_approach_angle(self, angle):
        self.approach_angle_ = angle
        self.rotated_full_axis_ = None
        
    def endpoints(self):
        return self.center_ - (self.grasp_width_ / 2.0) * self.axis_, self.center_ + (self.grasp_width_ / 2.0) * self.axis_,

    @staticmethod
    def _get_rotation_matrix_y(theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.c_[[cos_t, 0, sin_t], np.c_[[0, 1, 0], [-sin_t, 0, cos_t]]]
        return R
        
    @staticmethod
    def configuration_from_params(center, axis, width, angle=0, jaw_width=0, min_width=0):
        configuration = np.zeros(10)
        configuration[0:3] = center
        configuration[3:6] = axis
        configuration[6] = width
        configuration[7] = angle
        configuration[8] = jaw_width
        configuration[9] = min_width
        return configuration

    @staticmethod
    def params_from_configuration(configuration):
        if not isinstance(configuration, np.ndarray) or (configuration.shape[0] != 9 and configuration.shape[0] != 10):
            raise ValueError('Configuration must be numpy ndarray of size 9 or 10')
        if configuration.shape[0] == 9:
            min_grasp_width = 0
        else:
            min_grasp_width = configuration[9]
        return configuration[0:3], configuration[3:6], configuration[6], configuration[7], configuration[8], min_grasp_width

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
            vis - (bool) whether or not to plot the shoe
        Returns:
            c1 - the Contact3D for jaw 1
            c2 - the Contact3D for jaw 2
        """
        # compute num samples to use based on sdf resolution
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(self.grasp_width_)
        num_samples = int(Grasp.samples_per_grid * float(grasp_width_grid) / 2) # at least 1 sample per grid

        # get grasp endpoints in sdf frame
        g1_world, g2_world = self.endpoints()

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(g1_world, self.axis_, self.grasp_width_, obj,
                                                                     num_samples, min_width = self.min_grasp_width)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2_world, -self.axis_, self.grasp_width_, obj,
                                                                     num_samples, min_width = self.min_grasp_width)

        if False:#vis:
            plt.figure()
            plt.clf()
            h = plt.gcf()
            plt.ion()
            obj.sdf.scatter()

        # find contacts
        c1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis)
        c2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis)

        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()

        contacts_found = c1_found and c2_found
        return contacts_found, [c1, c2]

    @staticmethod
    def create_line_of_action(g, axis, width, obj, num_samples, min_width = 0, convert_grid=True):
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
        num_samples = max(num_samples, 3) # always at least 3 samples
        line_of_action = [g + t * axis for t in np.linspace(0, float(width) / 2 - float(min_width) / 2, num = num_samples)]
        if convert_grid:
            as_array = np.array(line_of_action).T
            transformed = obj.sdf.transform_pt_obj_to_grid(as_array)
            line_of_action = list(transformed.T)
        return line_of_action

    @staticmethod
    def find_contact(line_of_action, obj, vis=True, stop=False):
        """
        Find the point at which a point travelling along a given line of action hits a surface
        Params:
            line_of_action - list of np 3-arrays (grid coords), the points visited as the fingers close
            obj - GraspableObject3D to check contacts on
            vis - whether or not to display the contact check (for debugging)
        Returns:
            contact_found - whether or not the point contacts the object surface
            contact - Contact3D found along line of action (None if contact not found)
        """
        contact_found = False
        pt_zc = None
        pt_zc_world = None
        contact = None
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
            in_direction_grid = line_of_action[-1] - line_of_action[0]
            in_direction_grid = in_direction_grid / np.linalg.norm(in_direction_grid)
            in_direction = obj.sdf.transform_pt_grid_to_obj(in_direction_grid, direction=True)
            contact = contacts.Contact3D(obj, pt_zc_world, in_direction=in_direction)
        return contact_found, contact

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
            configuration = ParallelJawPtGrasp3D.configuration_from_params(grasp_center_obj, grasp_axis_y_obj, grasp_width_obj, theta, self.jaw_width)
            grasps_tf.append(ParallelJawPtGrasp3D(configuration, frame=self.frame_))
            theta = theta + theta_res
        return grasps_tf

    def gripper_pose(self, R_gripper_center = np.eye(3), t_gripper_center = np.zeros(3)):
        """
        Convert a grasp to a gripper pose in SE(3). Could optionally support PR2 gripper_l_tool_frame if proper params are passed in
        Params:
           R_gripper_center: (numpy 3x3 array) rotation matrix from grasp basis to gripper basis
           t_gripper_center: (numpy 3 array) translation from grasp basis to gripper basis
        Returns:
           pose_gripper: (tfx transform) pose of gripper in the grasp frame
        """
        pose_center_rot_ref = tfx.transform(self.rotated_full_axis, self.center_) # pose of rotated grasp center in its reference frame
        pose_gripper_center_rot = tfx.transform(R_gripper_center, t_gripper_center)

        pose_gripper_ref = pose_center_rot_ref.apply(pose_gripper_center_rot)
        return pose_gripper_ref

    def gripper_pose_stf(self, R_gripper_center = np.eye(3), t_gripper_center = np.zeros(3)):
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
        return stf.SimilarityTransform3D(pose=tfx.pose(pose_gripper_ref), scale=1.0, from_frame='grasp', to_frame='obj')
        
    def _angle_aligned_with_stable_pose(self, stable_pose):
        '''
        Returns the y-axis rotation angle that'd allow the current pose to align with stable pose
        '''    
        def _argmin(f, a, b, n):
            #finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            min_y = f(a)
            min_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y <= min_y:
                    min_y = y
                    min_x = x
            return min_x
    
        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return abs(np.dot(normal, grasp_axis_rotated))
            return matrix_product
    
        stable_pose_normal = stable_pose.r[2,:]
        
        theta = _argmin(_get_matrix_product_x_axis(np.array([1,0,0]), np.dot(inv(self.unrotated_full_axis), stable_pose_normal)), 0, 2*np.pi, 1000)
        return theta

    def grasp_aligned_with_stable_pose(self, stable_pose):
        '''
        Returns the grasp with approach_angle set appropriately to align with stable pose
        '''
        theta = self._angle_aligned_with_stable_pose(stable_pose)
        new_grasp = deepcopy(self)
        new_grasp.set_approach_angle(theta)
        return new_grasp

    def _angle_aligned_with_table(self, stable_pose):
        '''
        Returns the y-axis rotation angle that'd allow the current pose to align with stable pose
        '''    
        def _argmax(f, a, b, n):
            #finds the argmax x of f(x) in the range [a, b) with n samples
            delta = (b - a) / n
            max_y = f(a)
            max_x = a
            for i in range(1, n):
                x = i * delta
                y = f(x)
                if y >= max_y:
                    max_y = y
                    max_x = x
            return max_x
    
        def _get_matrix_product_x_axis(grasp_axis, normal):
            def matrix_product(theta):
                R = ParallelJawPtGrasp3D._get_rotation_matrix_y(theta)
                grasp_axis_rotated = np.dot(R, grasp_axis)
                return np.dot(normal, grasp_axis_rotated)
            return matrix_product
    
        stable_pose_normal = stable_pose.r[2]
        theta = _argmax(_get_matrix_product_x_axis(np.array([1,0,0]), np.dot(inv(self.unrotated_full_axis), -stable_pose_normal)), 0, 2*np.pi, 1000)        
        return theta

    def grasp_aligned_with_table_normal(self, stable_pose):
        '''
        Returns the grasp with approach_angle set appropriately to align with stable pose
        '''
        theta = self._angle_aligned_with_table(stable_pose)
        new_grasp = deepcopy(self)
        new_grasp.set_approach_angle(theta)        
        return new_grasp
        
    def collides_with_stable_pose(self, stable_pose, debug = []):
        '''
        Checks whether or not the current grasp would collide with the stable pose plane given 
        the physical dimensions of _GRIPPER_BOUNDING_BOX. 
        Collision check is done on a rotated grasp that aligns with the given stable pose
        '''
        plane_center = stable_pose.x0
        plane_normal = stable_pose.r[2]
        
        half_height = ParallelJawPtGrasp3D._GRIPPER_BOUNDING_BOX["half_height"]
        half_width = ParallelJawPtGrasp3D._GRIPPER_BOUNDING_BOX["half_width"]
        half_length = ParallelJawPtGrasp3D._GRIPPER_BOUNDING_BOX["half_length"]
        
        grasp_axis_xyz = self.rotated_full_axis.T
        grasp_axis_x = grasp_axis_xyz[0,:]
        grasp_axis_y = grasp_axis_xyz[1,:]
        grasp_axis_z = grasp_axis_xyz[2,:]
        
        vertices = []
        top_center_front = self.center + half_height * grasp_axis_z
        bottom_center_front = self.center - half_height * grasp_axis_z
        top_center_back = top_center_front - half_length * 2 * grasp_axis_x
        bottom_center_back = bottom_center_front - half_length * 2 * grasp_axis_x
        ref_vertices = [top_center_back, top_center_front, bottom_center_back, bottom_center_front]
        
        debug.append(vertices)

        for ref_vertex in ref_vertices:
            vertices.append(ref_vertex + half_width * grasp_axis_y)
            vertices.append(ref_vertex - half_width * grasp_axis_y)
    
        for vertex in vertices:
            if np.dot(plane_normal, np.ravel(vertex - plane_center)) <= 0:
                return True
                
        return False
        
    @staticmethod
    def grasp_from_contact_and_axis_on_grid(obj, grasp_c1_world, grasp_axis_world, grasp_width_world, grasp_angle=0, jaw_width_world=0,
                                            min_grasp_width_world = 0, vis = False, stop = False, backup=0.5):
        """
        Creates a grasp from a single contact point in grid coordinates and direction in grid coordinates
        Params:
            obj - GraspableObject3D
            grasp_c1_grid - contact point 1 in world
            grasp_axis - normalized direction of the grasp in world
            grasp_width_world - grasp_width in world coords
            jaw_width_world - width of jaws in world coords
            min_grasp_width_world - min closing width of jaws
            vis - whether or not to visualize the grasp
        Returns:
            ParallelJawGrasp3D object
            Contact3D instance for 2nd contact on object
        """
        # transform to grid basis
        grasp_axis_world = grasp_axis_world / np.linalg.norm(grasp_axis_world)
        grasp_axis_grid = obj.sdf.transform_pt_obj_to_grid(grasp_axis_world, direction=True)
        grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(grasp_width_world)
        min_grasp_width_grid = obj.sdf.transform_pt_obj_to_grid(min_grasp_width_world)
        grasp_c1_grid = obj.sdf.transform_pt_obj_to_grid(grasp_c1_world) - backup * grasp_axis_grid # subtract to find true point
        num_samples = int(2 * grasp_width_grid) # at least 2 samples per grid
        g2 = grasp_c1_grid + (grasp_width_grid - backup) * grasp_axis_grid

        # get line of action
        line_of_action1 = ParallelJawPtGrasp3D.create_line_of_action(grasp_c1_grid, grasp_axis_grid, grasp_width_grid, obj, num_samples,
                                                                     min_width=min_grasp_width_grid, convert_grid = False)
        line_of_action2 = ParallelJawPtGrasp3D.create_line_of_action(g2, -grasp_axis_grid, 2*grasp_width_grid, obj, num_samples,
                                                                     min_width=0, convert_grid = False)
        if vis:
            obj.sdf.scatter()
            ax = plt.gca(projection = '3d')
            ax.scatter(grasp_c1_grid[0] - grasp_axis_grid[0], grasp_c1_grid[1] - grasp_axis_grid[1], grasp_c1_grid[2] - grasp_axis_grid[2], c=u'r')
            ax.scatter(grasp_c1_grid[0], grasp_c1_grid[1], grasp_c1_grid[2], s=80, c=u'b')

        # compute the contact points on the object
        contact1_found, c1 = ParallelJawPtGrasp3D.find_contact(line_of_action1, obj, vis=vis, stop=stop)
        contact2_found, c2 = ParallelJawPtGrasp3D.find_contact(line_of_action2, obj, vis=vis, stop=stop)

        if vis:
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.draw()
        if not contact1_found or not contact2_found or np.linalg.norm(c1.point - c2.point) < min_grasp_width_world:
            logging.debug('No contacts found for grasp')
            return None, None

        # create grasp
        grasp_center = ParallelJawPtGrasp3D.grasp_center_from_endpoints(c1.point, c2.point)
        grasp_axis = ParallelJawPtGrasp3D.grasp_axis_from_endpoints(c1.point, c2.point)
        configuration = ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width_world, grasp_angle, jaw_width_world)
        return ParallelJawPtGrasp3D(configuration), c2 # relative to object

    @staticmethod
    def distance(g1, g2, alpha=1.0):
        center_dist = np.linalg.norm(g2.center - g2.center)
        axis_dist = (2.0 / np.pi) * np.arccos(np.abs(g1.axis.dot(g2.axis)))
        return alpha * center_dist + axis_dist

    def visualize(self, obj, arrow_len=0.01, line_width=20.0):
        """ Display point grasp as arrows on the contact points of the mesh """
        contacts_found, contacts = self.close_fingers(obj)

        if contacts_found:
            c1_world = contacts[0].point
            c2_world = contacts[1].point
            v = c2_world - c1_world
            v = arrow_len * v / np.linalg.norm(v)
            mv.quiver3d(c1_world[0] - v[0], c1_world[1] - v[1], c1_world[2] - v[2], v[0], v[1], v[2], scale_factor=1.0,
                        mode='arrow', line_width=line_width)
            mv.quiver3d(c2_world[0] + v[0], c2_world[1] + v[1], c2_world[2] + v[2], -v[0], -v[1], -v[2], scale_factor=1.0,
                        mode='arrow', line_width=line_width)

    def surface_information(self, graspable, width=2e-2, num_steps=21, direction=None):
        """Return the surface information at the contacts that this grasp makes
        on a graspable.
        Params:
            graspable - GraspableObject3D instance
            width - float width of the window in obj frame
            num_steps - int number of steps
        Returns:
            list of SurfaceWindows, one for each contact
        """
        return graspable.surface_information(self, width, num_steps, direction1=self.axis_, direction2=-self.axis_)

    def to_json(self, quality=0, method='PFC', num_successes=0, num_failures=0):
        """Converts the grasp to a Python dictionary for serialization to JSON."""
        return {
            # parameters to reconstruct ParallelJawPtGrasp3D instance
            'grasp_center': self.center,
            'grasp_axis': self.axis,
            'grasp_width': self.grasp_width,
            'jaw_width': self.jaw_width,
            'grasp_angle': self.approach_angle,
            # additional data
            'flag': 0, # what's this?
            'frame': 'gripper_l_tool_frame', # ?
            'reference_frame': 'object',
            'quality': quality,
            'metric': method,
            'successes': num_successes,
            'failures': num_failures,
        }

    @staticmethod
    def from_json(data):
        grasp_center = data['grasp_center']
        grasp_axis = data['grasp_axis']
        grasp_width = data['grasp_width']
        jaw_width = data['jaw_width']
        grasp_angle = data['grasp_angle']
        grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width,
                                                                                    grasp_angle, jaw_width))

        # load other attributes
        grasp.quality = data['quality']
        grasp.successes = data['successes']
        grasp.failures = data['failures']
        return grasp

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
    contact_found, _ = grasp.close_fingers(obj_3d, vis=True)
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
    g, _ = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(obj_3d, rand_surf_pt, axis, test_grasp_width, vis = True)
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
    g, _ = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(obj_3d, rand_surf_pt, axis, test_grasp_width, vis = False)
    j = g.to_json()

    # TODO: hard checks

if __name__ == '__main__':
    test_find_contacts()
#    test_grasp_from_contacts()
#    test_to_json()
