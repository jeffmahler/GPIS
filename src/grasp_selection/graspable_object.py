'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
from abc import ABCMeta, abstractmethod

import copy
import itertools as it
import logging
import mayavi.mlab as mv
import numpy as np
import os
import scipy.ndimage.filters as spfilt
import sys

from feature_functions import SurfaceWindow
import grasp as g
import mesh as m
import sdf as s
import similarity_tf as stf
import tfx

import IPython
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class GraspableObject:
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', model_name='', category=''):
        if not isinstance(tf, stf.SimilarityTransform3D):
            raise ValueError('Must initialize graspable objects with 3D similarity transform')
        self.sdf_ = sdf
        self.mesh_ = mesh
        self.tf_ = tf

        self.key_ = key
        self.model_name_ = model_name # for OpenRave usage, gross!
        self.category_ = category

        self.features_ = {} # dictionary mapping feature type to "bag of features"

        # make consistent poses, scales
        self.sdf_.tf = self.tf_
        if self.mesh_ is not None:
            self.mesh_.tf = self.tf_

    @abstractmethod
    def transform(self, tf):
        """Transforms object by tf."""
        pass

    def sample_shapes(self, num_samples):
        """Samples shape perturbations."""
        todo = 1

    @property
    def sdf(self):
        return self.sdf_

    @property
    def mesh(self):
        return self.mesh_

    @property
    def features(self):
        return self.features_

    @property
    def tf(self):
        return self.tf_

    @property
    def pose(self):
        return self.tf_.pose

    @property
    def scale(self):
        return self.tf_.scale

    @tf.setter
    def tf(self, tf):
        """ Update the pose of the object wrt the world """
        self.tf_ = tf
        self.sdf.tf_ = tf
        if self.mesh_ is not None:
            self.mesh_.tf_ = tf

    @pose.setter
    def pose(self, pose):
        """ Update the pose of the object wrt the world """
        self.tf_.pose = pose
        self.sdf.tf_.pose = pose
        if self.mesh_ is not None:
            self.mesh_.tf_.pose = pose

    @scale.setter
    def scale(self, scale):
        """ Update the scale of the object wrt the world """
        self.tf_.scale = scale
        self.sdf.tf_.scale = scale
        if self.mesh_ is not None:
            self.mesh_.tf_.scale = scale

    @property
    def key(self):
        return self.key_

    @property
    def model_name(self):
        return self.model_name_

    @property
    def category(self):
        return self.category_

class GraspableObject2D(GraspableObject):
    # TODO: fix 2d with similiarity tfs
    def __init__(self, sdf, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf2D):
            raise ValueError('Must initialize graspable object 2D with 2D sdf')
        GraspableObject.__init__(self, sdf, tf=tf)

class GraspableObject3D(GraspableObject):
    def __init__(self, sdf, mesh = None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', category='',
                 model_name=''):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        if mesh is not None and not isinstance(mesh, m.Mesh3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')

        self.center_of_mass_ = sdf.center_world() # use SDF bb center for now
        GraspableObject.__init__(self, sdf, mesh=mesh, tf=tf, key=key, category=category, model_name=model_name)

    def visualize(self, com_scale = 0.01):
        """
        Display both mesh and center of mass at the given scale
        """
        if self.mesh_ is not None:
            self.mesh_.visualize()
            mv.points3d(self.center_of_mass_[0], self.center_of_mass_[1], self.center_of_mass_[2],
                        scale_factor=com_scale)

    def moment_arm(self, x):
        """ Computes the moment arm to point x """
        return x - self.center_of_mass_

    def transform(self, tf):
        """ Transforms object by tf """
        new_tf = tf.compose(self.tf_)
        sdf_tf = self.sdf_.transform(tf)

        # TODO: fix mesh class
        if self.mesh_ is not None:
            mesh_tf = copy.copy(self.mesh_)
            mesh_tf.tf_ = new_tf

        return GraspableObject3D(sdf_tf, mesh_tf, new_tf)

    def _contact_tangents(self, contact, direction=None):
        """Returns the direction vector and tangent vectors at a contact point.
        The direction vector defaults to the normal vector on the SDF.
        Params:
            contact - numpy 3 array of the surface contact in obj coords
            direction - numpy 3 array to find orthogonal plane for
        Returns:
            direction, t1, t2 - numpy 3 arrays in obj coords
        """
        if direction is None: # compute normal at contact
            contact_grid = self.sdf.transform_pt_obj_to_grid(contact)
            on_surf, sdf_val = self.sdf.on_surface(contact_grid)
            if not on_surf:
                logging.debug('Contact point not on surface')
                return None, None, None

            grad = self.sdf.gradient(contact_grid)
            if np.all(grad == 0):
                return None, None, None

            # transform normal to obj frame
            normal = grad / np.linalg.norm(grad)
            direction = self.sdf.transform_pt_grid_to_obj(normal, direction=True)
            direction = -direction

        direction = direction.reshape((3, 1)) # make 2D for SVD

        # get orthogonal plane
        U, _, _ = np.linalg.svd(direction)

        # U[:, 1:] spans the tanget plane at the contact
        t1, t2 = U[:, 1], U[:, 2]
        return np.squeeze(direction), t1, t2

    def contact_friction_cone(self, contact, num_cone_faces=4, friction_coef=0.5):
        """
        Computes the friction cone and normal for a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj coords
            num_cone_faces - int number of cone faces to use
            friction_coef - float friction coefficient
        Returns:
            success - False when cone can't be computed
            cone_support - numpy array where each column is a vector on the cone
            normal - direction vector
        """
        normal, t1, t2 = self._contact_tangents(contact)
        if normal is None:
            return False, None, None

        tan_len = friction_coef
        force = -normal
        cone_support = np.zeros((3, num_cone_faces))

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec

        return True, cone_support, normal

    def contact_torques(self, contact, forces):
        """
        Get the torques that can be applied by a set of vectors with a given friction cone
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            forces - numpt 3xN array of the forces applied at the contact
        Returns:
            success - bool, whether or not successful
            torques - numpy 3xN array of the torques that can be computed
        """
        contact_grid = self.sdf.transform_pt_obj_to_grid(contact)
        if not self.sdf.on_surface(contact_grid):
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self.moment_arm(contact)
        for i in range(num_forces):
            torques[:,i] = np.cross(moment_arm, forces[:,i])

        return True, torques

    def contact_surface_window_sdf(self, contact, width=1e-2, num_steps=21):
        """Returns a window of SDF values on the tangent plane at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            width - float width of the window in obj frame
            num_steps - int number of steps
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        normal, t1, t2 = self._contact_tangents(contact)
        if normal is None: # normal and tangents not found
            return False

        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        window = np.zeros(num_steps**2)
        for i, (c1, c2) in enumerate(it.product(scales, repeat=2)):
            curr_loc = contact + c1 * t1 + c2 * t2
            curr_loc_grid = self.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = -1e-2
                continue

            window[i] = self.sdf[curr_loc_grid]
        return window.reshape((num_steps, num_steps))

    def _find_projection(self, curr_loc, direction, max_projection, back_up, num_samples, vis=False):
        """Finds the point of contact when shooting a direction ray from curr_loc.
        Params:
            curr_loc - numpy 3 array of the starting point in obj frame
            direction - normalized numpy 3 array, direction to look for contact

            max_projection - float maximum amount to search forward for a contact (meters)

            back_up - float amount to back up before finding a contact (meters)
            num_samples - float number of samples when finding contacts
        Returns:
            found - True if projection contact is found
            projection_contact - numpy 3 array of the contact point in obj frame
        """
        # get start of projection
        projection_start = curr_loc - back_up * direction
        line_of_action = g.ParallelJawPtGrasp3D.create_line_of_action(
            projection_start, direction, (max_projection + back_up), self, num_samples
        )
        found, projection_contact = g.ParallelJawPtGrasp3D.find_contact(
            line_of_action, self, vis=vis
        )

        if vis:
            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, self.sdf.dims_[0])
            ax.set_ylim3d(0, self.sdf.dims_[1])
            ax.set_zlim3d(0, self.sdf.dims_[2])
            plt.show()

        return found, projection_contact

    def _compute_surface_window_projection(self, contact, u1=None, u2=None, width=1e-2,
        num_steps=21, max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma=1.5, direction=None, vis=False, compute_weighted_covariance=False):
        """Compute the projection window at a contact point onto the basis
        defined by u1 and u2.

        Params:
            contact - numpy 3 array of the surface contact in obj frame
            u1, u2 - orthogonal numpy 3 arrays

            width - float width of the window in obj frame
            num_steps - int number of steps
            max_projection - float maximum amount to search forward for a
                contact (meters)

            back_up_units - float amount to back up before finding a contact (grid coords)
            samples_per_grid - float number of samples per grid when finding contacts
            sigma - bandwidth of gaussian filter on window
            direction - dir to do the projection along
            compute_weighted_covariance - whether to return the weighted
               covariance matrix, along with the window
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        direction, t1, t2 = self._contact_tangents(contact, direction)
        if direction is None: # normal and tangents not found
            raise ValueError('Direction could not be computed')
        if u1 is not None and u2 is not None: # use given basis
            t1, t2 = u1, u2

        # display sdf for debugging
        if vis:
            plt.figure()
            self.sdf.scatter()

        # number of samples used when looking for contacts
        no_contact = 0.2
        back_up = back_up_units * self.sdf.resolution
        num_samples = int(samples_per_grid * (max_projection + back_up) / self.sdf.resolution)
        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        window = np.zeros(num_steps**2)

        # start computing weighted covariance matrix
        if compute_weighted_covariance:
            cov = np.zeros((3, 3))
            cov_weight = 0

        for i, (c1, c2) in enumerate(it.product(scales, repeat=2)):
            curr_loc = contact + c1 * t1 + c2 * t2
            curr_loc_grid = self.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = 0.0
                continue

            found, projection_contact = self._find_projection(
                curr_loc, direction, max_projection, back_up, num_samples, vis)

            if found:
                # logging.debug('%d found.' %(i))
                sign = -direction.dot(projection_contact - curr_loc)
                projection = (sign / abs(sign)) * np.linalg.norm(projection_contact - curr_loc)
                if compute_weighted_covariance:
                    # weight according to SHOT: R - d_i
                    weight = width / np.sqrt(2) - np.sqrt(c1**2 + c2**2)
                    diff = (projection_contact - contact).reshape((3, 1))
                    cov += weight * np.dot(diff, diff.T)
                    cov_weight += weight
            else:
                logging.debug('%d not found.' %(i))
                projection = no_contact

            window[i] = projection

        window = window.reshape((num_steps, num_steps))

        # apply gaussian filter to window (should be narrow bandwidth)
        if sigma > 0.0:
            window = spfilt.gaussian_filter(window, sigma)
        if compute_weighted_covariance:
            return window, cov / cov_weight
        return window, 0.0

    def contact_surface_window_projection_unaligned(self, contact, width=1e-2,
        num_steps=21, max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma=1.5, direction=None, vis=False):
        """Projects the local surface onto the tangent plane at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            width - float width of the window in obj frame
            num_steps - int number of steps

            max_projection - float maximum amount to search forward for a contact (meters)

            back_up_units - float amount to back up before finding a contact (grid coords)
            samples_per_grid - float number of samples per grid when finding contacts
            sigma - bandwidth of gaussian filter on window
            direction - dir to do the projection along
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        return self._compute_surface_window_projection(contact,
            width=width, num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=vis)

    def contact_surface_window_projection(self, contact, width=1e-2,
        num_steps=21, max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma=1.5, direction=None, vis=False):
        """Projects the local surface onto the tangent plane at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            width - float width of the window in obj frame
            num_steps - int number of steps

            max_projection - float maximum amount to search forward for a contact (meters)

            back_up_units - float amount to back up before finding a contact (grid coords)
            samples_per_grid - float number of samples per grid when finding contacts
            sigma - bandwidth of gaussian filter on window
            direction - dir to do the projection along
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        direction, t1, t2 = self._contact_tangents(contact, direction)
        window, cov = self._compute_surface_window_projection(contact, t1, t2,
            width=width, num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=vis, compute_weighted_covariance=True)

        # compute principal axis
        pca = PCA()
        pca.fit(cov)
        R = pca.components_
        principal_axis = R[0, :]
        if np.isclose(abs(np.dot(principal_axis, direction)), 1):
            # principal axis is aligned with direction of projection, use secondary axis
            principal_axis = R[1, :]

        if vis:
            # reshape window
            window = window.reshape((num_steps, num_steps))

            # project principal axis onto tangent plane (t1, t2) to get u1
            u1t = np.array([np.dot(principal_axis, t1), np.dot(principal_axis, t2)])
            u2t = np.array([-u1t[1], u1t[0]])
            if sigma > 0:
                window = spfilt.gaussian_filter(window, sigma)
            plt.figure()
            plt.title('Principal Axis')
            plt.imshow(window, extent=[0, num_steps-1, num_steps-1, 0],
                    interpolation='none', cmap=plt.cm.binary)
            plt.colorbar()
            plt.clim(-0.01, 0.01) # fixing color range for visual comparisons
            center = num_steps // 2
            plt.scatter([center, center*u1t[0] + center], [center, -center*u1t[1] + center], color='blue')
            plt.scatter([center, center*u2t[0] + center], [center, -center*u2t[1] + center], color='green')

        u1 = np.dot(principal_axis, t1) * t1 + np.dot(principal_axis, t2) * t2
        u2 = np.cross(direction, u1) # u2 must be orthogonal to u1 on plane
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)

        window = self._compute_surface_window_projection(contact, u1, u2,
            width=width, num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=vis)

        # arbitrarily require that top_avg > bottom_avg (inspired by SHOT)
        top_avg = np.average(window[:num_steps//2, :])
        bottom_avg = np.average(window[num_steps//2:, :])
        if bottom_avg > top_avg:
            # need to flip both u1 and u2, i.e. rotate 180 degrees
            window = np.rot90(window, k=2)
        return window

    def surface_window_grad_curvature(self, contact, width, num_steps,
                                      back_up_units=3.0, direction=None):
        """
        Returns the local surface window, gradient, and curvature for a single contact
        """
        proj_window = self.contact_surface_window_projection(
            contact, width, num_steps,
            back_up_units=back_up_units, direction=direction)

        if proj_window is None:
            raise ValueError('Surface window could not be computed')
             
        grad_win = np.gradient(proj_window)
        hess_x = np.gradient(grad_win[0])
        hess_y = np.gradient(grad_win[1])

        gauss_curvature = np.zeros(proj_window.shape)
        for i in range(num_steps):
            for j in range(num_steps):
                local_hess = np.array([[hess_x[0][i, j], hess_x[1][i, j]],
                                       [hess_y[0][i, j], hess_y[1][i, j]]])
                # symmetrize
                local_hess = (local_hess + local_hess.T) / 2.0
                # curvature
                gauss_curvature[i, j] = np.linalg.det(local_hess)

        return SurfaceWindow(proj_window, grad_win, hess_x, hess_y, gauss_curvature)

    def surface_information(self, grasp, width, num_steps, plot=False):
        """
        Returns the local surface window, gradient, and curvature for the two point contacts of a grasp
        """
        contacts_found, contacts = grasp.close_fingers(self)
        contact1, contact2 = contacts
        if not contacts_found:
            raise ValueError('Failed to find contacts')

        if plot:
            plot_graspable(self, contact1)
            plot_graspable(self, contact2)

        window1 = self.surface_window_grad_curvature(
            contact1, width, num_steps, direction=None)#grasp.axis)
        window2 = self.surface_window_grad_curvature(
            contact2, width, num_steps, direction=None)#-grasp.axis)
        return window1, window2

def test_windows(width, num_steps, plot=None):
    import sdf_file, obj_file
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = GraspableObject3D(sdf, mesh)

    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1

    grasp1_center = np.array([0, 0, -0.025])
    grasp1 = g.ParallelJawPtGrasp3D(grasp1_center, grasp_axis, grasp_width)
    _, contacts1 = grasp1.close_fingers(graspable)
    contact1 = contacts1[0]

    grasp2_center = np.array([0, 0, -0.030])
    grasp2 = g.ParallelJawPtGrasp3D(grasp2_center, grasp_axis, grasp_width)
    _, contacts2 = grasp2.close_fingers(graspable)
    contact2 = contacts2[0]

    if plot:
        plot_graspable(graspable, contact1)

    print 'sdf window'
    sdf_window1 = graspable.contact_surface_window_sdf(
        contact1, width, num_steps)
    sdf_window2 = graspable.contact_surface_window_sdf(
        contact2, width, num_steps)

    print 'unaligned projection window'
    unaligned_window1 = graspable.contact_surface_window_projection_unaligned(
        contact1, width, num_steps)
    unaligned_window2 = graspable.contact_surface_window_projection_unaligned(
        contact2, width, num_steps)

    print 'aligned projection window'
    aligned_window1 = graspable.contact_surface_window_projection(
        contact1, width, num_steps)
    aligned_window2 = graspable.contact_surface_window_projection(
        contact2, width, num_steps)

    print 'proj, sdf, proj - sdf at contact'
    contact_index = num_steps // 2
    if num_steps % 2 == 0:
        contact_index += 1
    contact_index = (contact_index, contact_index)
    print aligned_window1[contact_index], sdf_window1[contact_index], aligned_window1[contact_index] - sdf_window1[contact_index]
    print aligned_window2[contact_index], sdf_window2[contact_index], aligned_window2[contact_index] - sdf_window2[contact_index]

    if plot:
        plot(sdf_window1, num_steps, 'SDF Window 1')
        plot(unaligned_window1, num_steps, 'Unaligned Projection Window 1')
        plot(aligned_window1, num_steps, 'Aligned Projection Window 1')
        plot(sdf_window2, num_steps, 'SDF Window 2')
        plot(unaligned_window2, num_steps, 'Unaligned Projection Window 2')
        plot(aligned_window2, num_steps, 'Aligned Projection Window 2')
        plt.show()

def test_window_distance(width, num_steps, plot=None):
    import sdf_file, obj_file
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = GraspableObject3D(sdf, mesh)

    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1

    grasp1_center = np.array([0, 0, -0.025])
    grasp1 = g.ParallelJawPtGrasp3D(grasp1_center, grasp_axis, grasp_width)
    grasp2_center = np.array([0, 0, -0.030])
    grasp2 = g.ParallelJawPtGrasp3D(grasp2_center, grasp_axis, grasp_width)

    w1, w2 = graspable.surface_information(grasp1, width, num_steps)
    v1, v2 = graspable.surface_information(grasp2, width, num_steps)

    # IPython.embed()

    if plot:
        plot(w1.proj_win, num_steps)
        plot(w2.proj_win, num_steps)
        plot(v1.proj_win, num_steps)
        plot(v2.proj_win, num_steps)
        plt.show()

    IPython.embed()

    return

def test_window_curvature(width, num_steps, plot=None):
    import sdf_file, obj_file
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = GraspableObject3D(sdf, mesh)

    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1

    for i, z in enumerate([-0.030, -0.035, -0.040, -0.045], 1):
        print 'w%d' %(i)
        grasp_center = np.array([0, 0, z])
        grasp = g.ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width)
        w, _ = graspable.surface_information(grasp, width, num_steps)
        for info in (w.proj_win_, w.gauss_curvature_):
            print 'min:', np.min(info), np.argmin(info)
            print 'max:', np.max(info), np.argmax(info)
        if plot:
            plot(w.proj_win_, num_steps, 'w%d proj_win' %(i), save=True)
            plot(1e4 * w.gauss_curvature_, num_steps, 'w%d curvature' %(i), save=True)

def test_window_correlation(width, num_steps):
    import scipy
    import sdf_file, obj_file
    import discrete_adaptive_samplers as das
    import experiment_config as ec
    import feature_functions as ff
    import graspable_object as go # weird Python issues
    import kernels
    import models
    import objectives
    import pfc
    import termination_conditions as tc

    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    config = ec.ExperimentConfig('cfg/correlated.yaml')
    config['window_width'] = width
    config['window_steps'] = num_steps
    brute_force_iter = 100
    snapshot_rate = config['bandit_snapshot_rate']

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = go.GraspableObject3D(sdf, mesh)
    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1

    grasps = []
    for z in [-0.030, -0.035, -0.040, -0.045]:
        grasp_center = np.array([0, 0, z])
        grasp = g.ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width)
        grasps.append(grasp)

    graspable_rv = pfc.GraspableObjectGaussianPose(graspable, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    # compute feature vectors for all grasps
    feature_extractor = ff.GraspableFeatureExtractor(graspable, config)
    all_features = feature_extractor.compute_all_features(grasps)

    candidates = []
    for grasp, features in zip(grasps, all_features):
        logging.info('Adding grasp %d' %len(candidates))
        grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
        pfc_rv = pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config)
        pfc_rv.set_features(features)
        candidates.append(pfc_rv)

    objective = objectives.RandomBinaryObjective()
    ua = das.UniformAllocationMean(objective, candidates)
    logging.info('Running uniform allocation for true pfc.')
    ua_result = ua.solve(termination_condition=tc.MaxIterTerminationCondition(brute_force_iter),
                         snapshot_rate=snapshot_rate)
    estimated_pfc = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)

    print 'true pfc'
    print estimated_pfc

    def phi(rv):
        return rv.features
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)

    print 'kernel matrix'
    print kernel.matrix(candidates)

def plot_graspable(graspable, contact, c1=0, c2=0, draw_plane=False):
    """Plot a graspable and the tangent plane at the point of contact."""
    # Plot SDF
    fig = plt.figure()
    graspable.sdf.scatter()

    # Plotting tangent plane and projection
    normal, t1, t2 = graspable._contact_tangents(contact)
    ax = plt.gca()
    contact_ = graspable.sdf.transform_pt_obj_to_grid(contact)
    n_ = graspable.sdf.transform_pt_obj_to_grid(normal)
    t1_ = graspable.sdf.transform_pt_obj_to_grid(t1)
    t2_ = graspable.sdf.transform_pt_obj_to_grid(t2)
    n_ = n_ / np.linalg.norm(n_)
    t1_ = t1_ / np.linalg.norm(t1_)
    t2_ = t2_ / np.linalg.norm(t2_)

    t1_x, t1_y, t1_z = zip(contact_, contact_ + t1_)
    t2_x, t2_y, t2_z = zip(contact_, contact_ + t2_)
    n_x, n_y, n_z = zip(contact_ + c1 * t1_ + c2 * t2_,
                        contact_ + c1 * t1_ + c2 * t2_ + n_)

    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            kwargs.update(dict(mutation_scale=20, lw=1, arrowstyle='-|>'))
            FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
            FancyArrowPatch.draw(self, renderer)
    t1_vec = Arrow3D(t1_x, t1_y, t1_z, color='c')
    t2_vec = Arrow3D(t2_x, t2_y, t2_z, color='m')
    n_vec =  Arrow3D(n_x,  n_y,  n_z,  color='k')
    ax.add_artist(t1_vec)
    ax.add_artist(t2_vec)
    ax.add_artist(n_vec)

    if draw_plane:
        n_ = np.cross(t1_, t2_)
        n_ = n_ / np.linalg.norm(n_)
        d = -contact_.dot(n_)
        xx, yy = np.meshgrid(range(8, 16), range(8, 16))
        z = (-n_[0] * xx - n_[1] * yy - d) * 1. / n_[2]
        ax.plot_surface(xx, yy, z, rstride=1, cstride=1, color='r')

def plot_window_3d(window, num_steps, title=''):
    """Make a 3D histogram of a window."""
    if num_steps % 2 == 1: # center window at contact
        indices = np.array(range(-(num_steps // 2), (num_steps // 2) + 1))
    else: # arbitrarily cut off upper bound to preserve window size
        indices = np.array(range(-(num_steps // 2), (num_steps // 2)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = np.repeat(indices, num_steps)
    j = np.tile(indices, num_steps)
    k = window.flatten()
    nonzero = (k != 0)

    i, j, k = i[nonzero], j[nonzero], k[nonzero]

    color = plt.cm.binary(k / k.max())
    barplot = ax.bar3d(i, j, np.zeros_like(i), 1, 1, k, color=color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

def plot_window_2d(window, num_steps, title='', save=False):
    """Plot a 2D image of a window."""
    if num_steps % 2 == 1: # center window at contact
        indices = np.array(range(-(num_steps // 2), (num_steps // 2) + 1))
    else: # arbitrarily cut off upper bound to preserve window size
        indices = np.array(range(-(num_steps // 2), (num_steps // 2)))
    indices = np.array(range(num_steps)) # for easier debugging

    fig = plt.figure()
    plt.title(title)
    imgplot = plt.imshow(window, extent=[indices[0], indices[-1], indices[-1], indices[0]],
                         interpolation='none', cmap=plt.cm.binary)
    plt.colorbar()
    plt.clim(-0.01, 0.01) # fixing color range for visual comparisons

    if save and title:
        plt.tight_layout()
        plt.savefig(title.replace(' ', '-'), bbox_inches='tight')
        plt.close()

def plot_window_both(window, num_steps):
    """Plot window as both 2D and 3D."""
    plot_window_2d(window, num_steps)
    plot_window_3d(window, num_steps)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_windows(width=2e-2, num_steps=21, plot=plot_window_2d)
    test_window_distance(width=2e-2, num_steps=21)
    test_window_curvature(width=2e-2, num_steps=21, plot=plot_window_2d)
    test_window_correlation(width=2e-2, num_steps=21)
