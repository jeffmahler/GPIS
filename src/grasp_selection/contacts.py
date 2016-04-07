"""
Contact class that encapsulates friction cone and surface window computation.
Author: Brian Hou
"""

from abc import ABCMeta, abstractmethod
import itertools as it
import logging
import numpy as np
import scipy.ndimage.filters as spfilt

import IPython
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

NO_CONTACT_DIST = 0.2 # distance to points that are not in contact for window extraction
WIN_DIST_LIM = 0.02 # limits for window plotting

class Contact:
    __metaclass__ = ABCMeta

class Contact3D(Contact):
    def __init__(self, graspable, contact_point, in_direction=None):
        self.graspable_ = graspable
        self.point_ = contact_point # in world coordinates

        # cached attributes
        self.in_direction_ = in_direction # inward facing grasp axis
        self.friction_cone_ = None
        self.normal_ = None # outward facing normal
        self.surface_info_ = None

        self._compute_normal()

    @property
    def graspable(self):
        return self.graspable_

    @property
    def point(self):
        return self.point_

    @property
    def normal(self):
        return self.normal_

    @normal.setter
    def normal(self, normal):
        self.normal_ = normal

    def _compute_normal(self):
        """Compute outward facing normal at contact, according to in_direction """
        # tf to grid
        as_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        on_surface, _ = self.graspable.sdf.on_surface(as_grid)
        if not on_surface:
            logging.debug('Contact point not on surface')
            return None

        # Use Hessian to compute outward facing normal
        curvature = self.graspable.sdf.curvature(as_grid)
        U, _, _ = np.linalg.svd(curvature)
        normal = U[:, 0]

        # flip normal to point outward if in_direction is defined
        if self.in_direction_ is not None and np.dot(self.in_direction_, normal) > 0:
            normal = -normal

        # transform to world frame
        normal = self.graspable.sdf.transform_pt_grid_to_obj(normal, direction=True)
        self.normal_ = normal

    def tangents(self, direction=None, max_samples=1000):
        """Returns the direction vector and tangent vectors at a contact point.
        The direction vector defaults to the *inward-facing* normal vector at
        this contact.
        Params:
            in_direction - numpy 3 array to find orthogonal plane for
        Returns:
            direction, t1, t2 - numpy 3 arrays in obj coords, where direction
                points into the object
        """
        # illegal contact, cannot return tangents
        if self.normal_ is None:
            return None, None, None

        # default to inward pointing normal
        if direction is None:
            direction = -self.normal_

        # force direction to face inward
        if np.dot(self.normal_, direction) > 0:
            direction = -direction

        # transform to 
        direction = direction.reshape((3, 1)) # make 2D for SVD

        # get orthogonal plane
        U, _, _ = np.linalg.svd(direction)

        # U[:, 1:] spans the tanget plane at the contact
        x, y = U[:, 1], U[:, 2]

        # make sure t1 and t2 obey right hand rule
        z_hat = np.cross(x, y)
        if z_hat.dot(direction) < 0:
            y = -y

        # redefine tangent x axis to maximally align with the object y axis
        max_ip = 0
        max_a = 0
        obj_y = np.array([0,1,0])
        for i in range(max_samples+1):
            a = -1 + i * (2.0 / max_samples)
            v = a * x + np.sqrt(1 - a**2) * y
            if v.dot(obj_y) > max_ip:
                max_ip = v.dot(obj_y)
                max_a = a
        v = max_a * x + np.sqrt(1 - max_a**2) * y
        w = np.cross(direction.ravel(), v)

        return np.squeeze(direction), v, w

    def normal_force_magnitude(self):
        """ Returns the magnitude of the force that the contact would apply along the normal direction"""
        normal_force_mag = 1.0
        if self.in_direction_ is not None and self.normal_ is not None:
            in_normal = -self.normal_
            in_direction_norm = self.in_direction_ / np.linalg.norm(self.in_direction_)
            normal_force_mag = np.dot(in_direction_norm, in_normal)
        return max(normal_force_mag, 0.0)

    def friction_cone(self, num_cone_faces=8, friction_coef=0.5):
        """
        Computes the friction cone and normal for a contact point.
        Params:
            num_cone_faces - int number of cone faces to use
            friction_coef - float friction coefficient
        Returns:
            success - False when cone can't be computed
            cone_support - numpy array where each column is a vector on the cone
            normal - outward facing direction vector
        """
        if self.friction_cone_ is not None and self.normal_ is not None:
            return True, self.friction_cone_, self.normal_

        # get normal and tangents
        in_normal, t1, t2 = self.tangents()
        if in_normal is None:
            return False, self.friction_cone_, self.normal_

        friction_cone_valid = True

        # check whether contact would slip, which is whether or not the tangent force is always greater than the frictional force
        if self.in_direction_ is not None:
            in_direction_norm = self.in_direction_ / np.linalg.norm(self.in_direction_)
            normal_force_mag = self.normal_force_magnitude()
            tan_force_x = np.dot(in_direction_norm, t1)
            tan_force_y = np.dot(in_direction_norm, t2)
            tan_force_mag = np.sqrt(tan_force_x**2 + tan_force_y**2)
            friction_force_mag = friction_coef * normal_force_mag

            if friction_force_mag < tan_force_mag:
                logging.debug('Contact would slip')
                return False, self.friction_cone_, self.normal_

        # set up friction cone
        tan_len = friction_coef
        force = in_normal
        cone_support = np.zeros((3, num_cone_faces))

        # find convex combinations of tangent vectors
        for j in range(num_cone_faces):
            tan_vec = t1 * np.cos(2 * np.pi * (float(j) / num_cone_faces)) + t2 * np.sin(2 * np.pi * (float(j) / num_cone_faces))
            cone_support[:, j] = force + friction_coef * tan_vec

        self.friction_cone_ = cone_support
        return True, self.friction_cone_, self.normal_

    def torques(self, forces):
        """
        Get the torques that can be applied by a set of vectors with a given
        friction cone.
        Params:
            forces - numpy 3xN array of the forces applied at the contact
        Returns:
            success - bool, whether or not successful
            torques - numpy 3xN array of the torques that can be computed
        """
        as_grid = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        on_surface, _ = self.graspable.sdf.on_surface(as_grid)
        if not on_surface:
            logging.debug('Contact point not on surface')
            return False, None

        num_forces = forces.shape[1]
        torques = np.zeros([3, num_forces])
        moment_arm = self.graspable.moment_arm(self.point)
        for i in range(num_forces):
            torques[:,i] = np.cross(moment_arm, forces[:,i])

        return True, torques

    def surface_window_sdf(self, width=1e-2, num_steps=21):
        """Returns a window of SDF values on the tangent plane at a contact point.
        Params:
            width - float width of the window in obj frame
            num_steps - int number of steps
        Returns:
            window - numpy NUM_STEPSxNUM_STEPS array of distances from tangent
                plane to obj, False if surface window can't be computed
        """
        in_normal, t1, t2 = self.tangents()
        if in_normal is None: # normal and tangents not found
            return False

        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        window = np.zeros(num_steps**2)
        for i, (c1, c2) in enumerate(it.product(scales, repeat=2)):
            curr_loc = self.point + c1 * t1 + c2 * t2
            curr_loc_grid = self.graspable.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.graspable.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = -1e-2
                continue

            window[i] = self.graspable.sdf[curr_loc_grid]
        return window.reshape((num_steps, num_steps))

    def _compute_surface_window_projection(self, u1=None, u2=None, width=1e-2,
        num_steps=21, max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma=1.5, direction=None, vis=False, compute_weighted_covariance=False, disc=False, num_radial_steps=5):
        """Compute the projection window onto the basis defined by u1 and u2.
        Params:
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
        direction, t1, t2 = self.tangents(direction)
        if direction is None: # normal and tangents not found
            raise ValueError('Direction could not be computed')
        if u1 is not None and u2 is not None: # use given basis
            t1, t2 = u1, u2

        # number of samples used when looking for contacts
        no_contact = NO_CONTACT_DIST
        back_up = back_up_units * self.graspable.sdf.resolution
        num_samples = int(samples_per_grid * (max_projection + back_up) / self.graspable.sdf.resolution)
        window = np.zeros(num_steps**2)

        scales = np.linspace(-width / 2.0, width / 2.0, num_steps)
        scales_it = it.product(scales, repeat=2)
        if disc:
            scales_it = []
            for i in range(num_steps):
                theta = 2.0 * np.pi / i
                for j in range(num_radial_steps):
                    r = (j+1) * width / num_radial_steps
                    p = (r * np.cos(theta), r * np.sin(theta))
                    scales_it.append(p)

        # start computing weighted covariance matrix
        if compute_weighted_covariance:
            cov = np.zeros((3, 3))
            cov_weight = 0

        if vis:
            ax = plt.gca(projection = '3d')
            self.graspable_.sdf.scatter()

        for i, (c1, c2) in enumerate(scales_it):
            curr_loc = self.point + c1 * t1 + c2 * t2
            curr_loc_grid = self.graspable.sdf.transform_pt_obj_to_grid(curr_loc)
            if self.graspable.sdf.is_out_of_bounds(curr_loc_grid):
                window[i] = 0.0
                continue

            if vis:
                ax.scatter(curr_loc_grid[0], curr_loc_grid[1], curr_loc_grid[2], s=130, c=u'y')

            found, projection_contact = self.graspable._find_projection(
                curr_loc, direction, max_projection, back_up, num_samples, vis=vis)

            if found:
                # logging.debug('%d found.' %(i))
                sign = direction.dot(projection_contact.point - curr_loc)
                projection = (sign / abs(sign)) * np.linalg.norm(projection_contact.point - curr_loc)
                projection = min(projection, max_projection)
                if compute_weighted_covariance:
                    # weight according to SHOT: R - d_i
                    weight = width / np.sqrt(2) - np.sqrt(c1**2 + c2**2)
                    diff = (projection_contact.point - self.point).reshape((3, 1))
                    cov += weight * np.dot(diff, diff.T)
                    cov_weight += weight
            else:
                logging.debug('%d not found.' %(i))
                projection = no_contact

            window[i] = projection

        if not disc:
            window = window.reshape((num_steps, num_steps))

            # apply gaussian filter to window (should be narrow bandwidth)
            if sigma > 0.0:
                window = spfilt.gaussian_filter(window, sigma)
            if compute_weighted_covariance:
                if cov_weight > 0:
                    return window, cov / cov_weight
                return window, cov
        return window

    def surface_window_projection_unaligned(self, width=1e-2, num_steps=21,
        max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma=1.5, direction=None, vis=False):
        """Projects the local surface onto the tangent plane at a contact point.
        Params:
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
        return self._compute_surface_window_projection(width=width,
            num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=vis)

    def surface_window_projection(self, width=1e-2, num_steps=21,
        max_projection=0.1, back_up_units=3.0, samples_per_grid=2.0,
        sigma_mult=0.07, direction=None, compute_pca=False, vis=False):
        """Projects the local surface onto the tangent plane at a contact point.
        Params:
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
        # normalize sigma by num steps
        sigma = sigma_mult * num_steps

        # get initial projection
        direction, t1, t2 = self.tangents(direction)
        window, cov = self._compute_surface_window_projection(t1, t2,
            width=width, num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=False, compute_weighted_covariance=True)

        if not compute_pca:
            return window
        
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
            plt.clim(-WIN_DIST_LIM, WIN_DIST_LIM) # fixing color range for visual comparisons
            center = num_steps // 2
            plt.scatter([center, center*u1t[0] + center], [center, -center*u1t[1] + center], color='blue')
            plt.scatter([center, center*u2t[0] + center], [center, -center*u2t[1] + center], color='green')

        u1 = np.dot(principal_axis, t1) * t1 + np.dot(principal_axis, t2) * t2
        u2 = np.cross(direction, u1) # u2 must be orthogonal to u1 on plane
        u1 = u1 / np.linalg.norm(u1)
        u2 = u2 / np.linalg.norm(u2)

        window = self._compute_surface_window_projection(u1, u2,
            width=width, num_steps=num_steps, max_projection=max_projection,
            back_up_units=back_up_units, samples_per_grid=samples_per_grid,
            sigma=sigma, direction=direction, vis=False)

        # arbitrarily require that right_avg > left_avg (inspired by SHOT)
        left_avg = np.average(window[:, :num_steps//2])
        right_avg = np.average(window[:, num_steps//2:])
        if left_avg > right_avg:
            # need to flip both u1 and u2, i.e. rotate 180 degrees
            window = np.rot90(window, k=2)

        if vis:
            if sigma > 0:
                window = spfilt.gaussian_filter(window, sigma)
            plt.figure()
            plt.title('Tfd')
            plt.imshow(window, extent=[0, num_steps-1, num_steps-1, 0],
                    interpolation='none', cmap=plt.cm.binary)
            plt.colorbar()
            plt.clim(-WIN_DIST_LIM, WIN_DIST_LIM) # fixing color range for visual comparisons            
            plt.show()

        return window

    def surface_information(self, width, num_steps, back_up_units=3.0, direction=None):
        """
        Returns the local surface window, gradient, and curvature for a single contact.
        """
        if self.surface_info_ is not None:
            return self.surface_info_

        proj_window = self.surface_window_projection(width, num_steps,
            back_up_units=back_up_units, direction=direction, vis=False)

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

    def plot_friction_cone(self, color='r', scale=1.0):
        success, cone, in_normal = self.friction_cone()

        if not success:
            logging.warning('Friction cone does not exist')
            return

        ax = plt.gca(projection='3d')
        self.graspable.sdf.scatter() # object
        x, y, z = self.graspable.sdf.transform_pt_obj_to_grid(self.point)
        nx, ny, nz = self.graspable.sdf.transform_pt_obj_to_grid(in_normal, direction=True)
        ax.scatter([x], [y], [z], c=color, s=60) # contact
        ax.scatter([x - nx], [y - ny], [z - nz], c=color, s=60) # normal
        ax.scatter(x + scale*cone[0], y + scale*cone[1], z + scale*cone[2], c=color, s=40) # cone

        ax.set_xlim3d(0, self.graspable.sdf.dims_[0])
        ax.set_ylim3d(0, self.graspable.sdf.dims_[1])
        ax.set_zlim3d(0, self.graspable.sdf.dims_[2])

        return plt.Rectangle((0, 0), 1, 1, fc=color) # return a proxy for legend

class SurfaceWindow:
    """Struct for encapsulating local surface window features."""
    def __init__(self, proj_win, grad, hess_x, hess_y, gauss_curvature):
        self.proj_win_ = proj_win
        self.grad_ = grad
        self.hess_x_ = hess_x
        self.hess_y_ = hess_y
        self.gauss_curvature_ = gauss_curvature

    @property
    def proj_win_2d(self):
        return self.proj_win_

    @property
    def proj_win(self):
        return self.proj_win_.flatten()

    @property
    def grad_x(self):
        return self.grad_[0].flatten()

    @property
    def grad_y(self):
        return self.grad_[1].flatten()

    @property
    def curvature(self):
        return self.gauss_curvature_.flatten()

    def asarray(self, proj_win_weight=0.0, grad_x_weight=0.0,
                grad_y_weight=0.0, curvature_weight=0.0):
        proj_win = proj_win_weight * self.proj_win
        grad_x = grad_x_weight * self.grad_x
        grad_y = grad_y_weight * self.grad_y
        curvature = curvature_weight * self.gauss_curvature
        return np.append([], [proj_win, grad_x, grad_y, curvature])


def test_plot_friction_cone():
    import sdf_file, obj_file, grasp as g, graspable_object
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = graspable_object.GraspableObject3D(sdf, mesh)

    grasp_axis = np.array([0, 1, 0])
    grasp_width = 0.1
    grasp_center = np.array([0, 0, -0.025])
    grasp = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, 0, grasp_width, 0))

    _, (c1, c2) = grasp.close_fingers(graspable)
    plt.figure()
    c1_proxy = c1.plot_friction_cone(color='m')
    c2_proxy = c2.plot_friction_cone(color='y')
    plt.legend([c1_proxy, c2_proxy], ['Cone 1', 'Cone 2'])
    plt.show()
    IPython.embed()

if __name__ == '__main__':
    test_plot_friction_cone()
