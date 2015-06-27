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
import sys

import grasp as g
import mesh as m
import sdf as s
import similarity_tf as stf
import tfx

import IPython
import matplotlib.pyplot as plt

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

    def _contact_normal_and_tangent(self, contact):
        """Returns the normal vector and tangent vectors at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj coords
        Returns:
            normal, t1, t2 - numpy 3 arrays in obj coords
        """
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
        normal = self.sdf.transform_pt_grid_to_obj(normal, direction=True)
        normal = normal.reshape((3, 1)) # make 2D for SVD

        # get tangent plane
        U, _, _ = np.linalg.svd(normal)

        # U[:, 1:] spans the tanget plane at the contact
        t1, t2 = U[:, 1], U[:, 2]
        return np.squeeze(normal), t1, t2

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
        normal, t1, t2 = self._contact_normal_and_tangent(contact)
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

    def contact_surface_window_sdf(self, contact, width=1e-1, num_steps=21):
        """Returns a window of SDF values on the tangent plane at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            width - float width of the window in obj frame
            num_steps - int number of steps
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        normal, t1, t2 = self._contact_normal_and_tangent(contact)
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

    def contact_surface_window_projection(self, contact, width=1e-1, num_steps=21,
                                          back_up=1e-5, max_projection=5.0, samples_per_unit=500.0):
        """Projects the local surface onto the tangent plane at a contact point.
        Params:
            contact - numpy 3 array of the surface contact in obj frame
            width - float width of the window in obj frame
            num_steps - int number of steps

            back_up - float amount to back up before finding a contact
            max_projection - float maximum amount to search forward for a contact
            samples_per_unit - float number of samples per unit vector when
                finding contacts
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        normal, t1, t2 = self._contact_normal_and_tangent(contact)
        if normal is None: # normal and tangents not found
            return False

        # number of samples used when looking for contacts
        num_samples = int((max_projection + back_up) * samples_per_unit)
        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        window = np.zeros(num_steps**2)
        for i, (c1, c2) in enumerate(it.product(scales, repeat=2)):
            curr_loc = contact + c1 * t1 + c2 * t2
            curr_loc_grid = self.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = -1e-2
                continue

            projection_start = curr_loc - back_up * normal
            line_of_action = g.ParallelJawPtGrasp3D.create_line_of_action(
                projection_start, normal, max_projection, self, num_samples
            )
            found, projection_contact = g.ParallelJawPtGrasp3D.find_contact(
                line_of_action, self, vis=False
            )
            if found:
                logging.debug('{} found.'.format(i))
                # take mean to reduce rounding issues
                projection = ((projection_contact - curr_loc) / normal).mean()
            else:
                projection = 0.0
            # if i in (20,):
            #     IPython.embed()

            window[i] = projection
        return window.reshape((num_steps, num_steps))
        

def test_windows(width, num_steps, plot=None):
    import sdf_file, obj_file
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = GraspableObject3D(sdf, mesh)

    grasp_center = np.array([0, 0, -0.025])
    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1
    grasp = g.ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width)

    _, contacts = grasp.close_fingers(graspable)
    contact = contacts[0]

    if plot:
        plot_graspable(graspable, contact)

    print 'sdf window'
    sdf_window = graspable.contact_surface_window_sdf(
        contact, width, num_steps)

    print 'projection window'
    proj_window = graspable.contact_surface_window_projection(
        contact, width, num_steps)

    print 'proj, sdf, proj - sdf at contact'
    contact_index = num_steps // 2
    if num_steps % 2 == 0:
        contact_index += 1
    contact_index = (contact_index, contact_index)
    print proj_window[contact_index], sdf_window[contact_index], proj_window[contact_index] - sdf_window[contact_index]

    if plot:
        plot(sdf_window, num_steps)
        plot(proj_window, num_steps)
        plt.show()

    IPython.embed()

    return sdf_window, proj_window

def plot_graspable(graspable, contact, c1=0, c2=0, draw_plane=False):
    """Plot a graspable and the tangent plane at the point of contact."""
    # Plot SDF
    fig = plt.figure()
    graspable.sdf.scatter()

    # Plotting tangent plane and projection
    normal, t1, t2 = graspable._contact_normal_and_tangent(contact)
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

def plot_window_3d(window, num_steps):
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

def plot_window_2d(window, num_steps):
    """Plot a 2D image of a window."""
    if num_steps % 2 == 1: # center window at contact
        indices = np.array(range(-(num_steps // 2), (num_steps // 2) + 1))
    else: # arbitrarily cut off upper bound to preserve window size
        indices = np.array(range(-(num_steps // 2), (num_steps // 2)))
    indices = np.array(range(num_steps)) # for easier debugging

    fig = plt.figure()
    imgplot = plt.imshow(window, extent=[indices[0], indices[-1], indices[-1], indices[0]],
                         interpolation='none', cmap=plt.cm.binary)
    plt.colorbar()

def plot_window_both(window, num_steps):
    """Plot window as both 2D and 3D."""
    plot_window_2d(window, num_steps)
    plot_window_3d(window, num_steps)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_windows(width=1e-3, num_steps=21, plot=plot_window_2d)
