'''
Encapsulates data and operations on a 2D or 3D object that can be grasped
Author: Jeff Mahler
'''
from abc import ABCMeta, abstractmethod

import copy
import logging
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import numpy as np

import grasp as g
import mesh as m
import sdf as s
import similarity_tf as stf
import tfx

import IPython
import matplotlib.pyplot as plt

class GraspableObject:
    __metaclass__ = ABCMeta

    def __init__(self, sdf, mesh=None, features=None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', model_name='', category=''):
        if not isinstance(tf, stf.SimilarityTransform3D):
            raise ValueError('Must initialize graspable objects with 3D similarity transform')
        self.sdf_ = sdf
        self.mesh_ = mesh
        self.tf_ = tf

        self.key_ = key
        self.model_name_ = model_name # for OpenRave usage, gross!
        self.category_ = category

        self.features_ = features # shot features

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

    def set_model_name(self, model_name):
        self.model_name_ = model_name

class GraspableObject2D(GraspableObject):
    # TODO: fix 2d with similiarity tfs
    def __init__(self, sdf, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0)):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf2D):
            raise ValueError('Must initialize graspable object 2D with 2D sdf')
        GraspableObject.__init__(self, sdf, tf=tf)

class GraspableObject3D(GraspableObject):
    def __init__(self, sdf, mesh = None, features=None, tf = stf.SimilarityTransform3D(tfx.identity_tf(), 1.0), key='', category='',
                 model_name=''):
        """ 2D objects are initialized with sdfs only"""
        if not isinstance(sdf, s.Sdf3D):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')
        if mesh is not None and not (isinstance(mesh, m.Mesh3D) or isinstance(mesh,mx.Mesh)):
            raise ValueError('Must initialize graspable object 3D with 3D sdf')

        self.center_of_mass_ = sdf.center_world() # use SDF bb center for now
        GraspableObject.__init__(self, sdf, mesh=mesh, features=features, tf=tf, key=key, category=category, model_name=model_name)

    def visualize(self, com_scale = 0.01):
        """
        Display both mesh and center of mass at the given scale
        """
        if self.mesh_ is not None:
            self.mesh_.visualize()
            mv.points3d(self.center_of_mass_[0], self.center_of_mass_[1], self.center_of_mass_[2],
                        scale_factor=com_scale)

    def plot_sdf_vs_mesh(self):
        def plot_surface(points, color):
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            ax.scatter(x, y, z, c=color)
        def plot_plane(normal, point):
            d = -point.dot(normal)
            # print('{}x + {}y + {}z + {} = 0'.format(normal[0], normal[1], normal[2], d))
            xx, yy = np.meshgrid(range(dim), range(dim))
            z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
            ax.plot_surface(xx, yy, z)

        sdf_surface_points, _ = self.sdf.surface_points()
        mesh_surface_points = np.array([graspable.sdf.transform_pt_obj_to_grid(np.array(v))
                                        for v in self.mesh.vertices()])
        dim = max(self.sdf.dimensions)

        ax = plt.gca(projection = '3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim3d(0, dim)
        ax.set_ylim3d(0, dim)
        ax.set_zlim3d(0, dim)

        plot_surface(sdf_surface_points, 'b')
        plot_surface(mesh_surface_points, 'r')
        plt.show(block=False)

    def moment_arm(self, x):
        """ Computes the moment arm to point x """
        return x - self.center_of_mass_

    def transform(self, tf):
        """ Transforms object by tf """
        tf.from_frame = self.tf_.to_frame
        tf.to_frame = self.tf_.to_frame
        new_tf = tf.dot(self.tf_)
        sdf_tf = self.sdf_.transform(tf)

        # TODO: fix mesh class
        if self.mesh_ is not None:
            mesh_tf = self.mesh_.transform(tf)

        return GraspableObject3D(sdf_tf, mesh_tf, new_tf)
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
            projection_contact - Contact3D instance
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

        return found, projection_contact

    def surface_information(self, grasp, width, num_steps, sigma_range=0.1, sigma_spatial=1, plot=False, direction1=None, 
                                direction2=None, debug_objs=None):
        """
        Returns the local surface window, gradient, and curvature for the two
        point contacts of a grasp.
        """
        contacts_found, contacts = grasp.close_fingers(self)#, vis=True)
        if not contacts_found:
            raise ValueError('Failed to find contacts')
        contact1, contact2 = contacts

        if plot:
            plt.figure()
            contact1.plot_friction_cone()
            contact2.plot_friction_cone()

            ax = plt.gca(projection = '3d')
            ax.set_xlim3d(0, self.sdf.dims_[0])
            ax.set_ylim3d(0, self.sdf.dims_[1])
            ax.set_zlim3d(0, self.sdf.dims_[2])

        window1 = contact1.surface_information(width, num_steps, sigma_range=sigma_range, sigma_spatial=sigma_spatial, direction=direction1, debug_objs=debug_objs)
        window2 = contact2.surface_information(width, num_steps, sigma_range=sigma_range, sigma_spatial=sigma_spatial, direction=direction2, debug_objs=debug_objs)

        return window1, window2, contact1, contact2

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
    grasp1 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp1_center, grasp_axis, grasp_width))
    _, contacts1 = grasp1.close_fingers(graspable)
    contact1 = contacts1[0]

    grasp2_center = np.array([0, 0, -0.030])
    grasp2 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp2_center, grasp_axis, grasp_width))
    _, contacts2 = grasp2.close_fingers(graspable)
    contact2 = contacts2[0]

    if plot:
        plt.figure()
        contact1.plot_friction_cone()
        contact2.plot_friction_cone()

    print 'sdf window'
    sdf_window1 = contact1.surface_window_sdf(width, num_steps)
    sdf_window2 = contact2.surface_window_sdf(width, num_steps)

    print 'unaligned projection window'
    unaligned_window1 = contact1.surface_window_projection_unaligned(width, num_steps)
    unaligned_window2 = contact2.surface_window_projection_unaligned(width, num_steps)

    print 'aligned projection window'
    aligned_window1 = contact1.surface_window_projection(width, num_steps)
    aligned_window2 = contact2.surface_window_projection(width, num_steps)

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
    grasp1 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp1_center, grasp_axis, grasp_width))
    grasp2_center = np.array([0, 0, -0.030])
    grasp2 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp2_center, grasp_axis, grasp_width))

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
        grasp = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width))
        w, _ = graspable.surface_information(grasp, width, num_steps)
        for info in (w.proj_win_, w.gauss_curvature_):
            print 'min:', np.min(info), np.argmin(info)
            print 'max:', np.max(info), np.argmax(info)
        if plot:
            plot(w.proj_win_, num_steps, 'w%d proj_win' %(i), save=True)
            # plot(1e4 * w.gauss_curvature_, num_steps, 'w%d curvature' %(i), save=True)

def test_window_correlation(width, num_steps, vis=True):
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
        grasp = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp_center, grasp_axis, grasp_width))
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

        if vis:
            _, (c1, c2) = grasp.close_fingers(graspable)
            plt.figure()
            c1_proxy = c1.plot_friction_cone(color='m')
            c2_proxy = c2.plot_friction_cone(color='y')
            plt.legend([c1_proxy, c2_proxy], ['Cone 1', 'Cone 2'])
            plt.title('Grasp %d' %(len(candidates)))

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

    if vis:
        plt.show()

def plot_graspable(graspable, contact, c1=0, c2=0, draw_plane=False):
    """Plot a graspable and the tangent plane at the point of contact."""
    # Plot SDF
    fig = plt.figure()
    graspable.sdf.scatter()

    # Plotting tangent plane and projection
    normal, t1, t2 = contact.tangents()
    ax = plt.gca()
    contact_ = graspable.sdf.transform_pt_obj_to_grid(contact.point)
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
    plt.clim(-0.004, 0.004) # fixing color range for visual comparisons

    if save and title:
        plt.tight_layout()
        plt.savefig(title.replace(' ', '-'), bbox_inches='tight')
        plt.close()

def plot_window_both(window, num_steps):
    """Plot window as both 2D and 3D."""
    plot_window_2d(window, num_steps)
    plot_window_3d(window, num_steps)

def generate_window_for_figure(width, num_steps, plot=None):
    import sdf_file, obj_file
    np.random.seed(100)

    mesh_file_name = '/mnt/terastation/shape_data/MASTER_DB_v2/SHREC14LSGTB/M000385.obj'
    sdf_3d_file_name = '/mnt/terastation/shape_data/MASTER_DB_v2/SHREC14LSGTB/M000385.sdf'

    sdf = sdf_file.SdfFile(sdf_3d_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = GraspableObject3D(sdf, mesh)

    grasp_axis = np.array([1, 0, 0])
    grasp_width = 0.15

    grasp1_center = np.array([0, 0, 0.075])
    grasp1 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp1_center, grasp_axis, grasp_width))
    _, contacts1 = grasp1.close_fingers(graspable, vis=True)
    contact1 = contacts1[0]

    grasp2_center = np.array([0, 0, 0.025])
    grasp2 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp2_center, grasp_axis, grasp_width))
    _, contacts2 = grasp2.close_fingers(graspable, vis=True)
    contact2 = contacts2[0]

    grasp3_center = np.array([0, 0.012, -0.089])
    grasp3 = g.ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(grasp3_center, grasp_axis, grasp_width))
    _, contacts3 = grasp3.close_fingers(graspable, vis=True)
    contact3 = contacts3[0]

    if plot:
        plt.figure()
        contact1.plot_friction_cone()
        contact2.plot_friction_cone()

    print 'aligned projection window'
    aligned_window1 = contact1.surface_window_projection(width, num_steps, vis=True)
    aligned_window2 = contact2.surface_window_projection(width, num_steps, vis=True)
    aligned_window3 = contact3.surface_window_projection(width, num_steps, vis=True)
    plt.show()
    IPython.embed()

    if plot:
        plot(sdf_window1, num_steps, 'SDF Window 1')
        plot(unaligned_window1, num_steps, 'Unaligned Projection Window 1')
        plot(aligned_window1, num_steps, 'Aligned Projection Window 1')
        plot(sdf_window2, num_steps, 'SDF Window 2')
        plot(unaligned_window2, num_steps, 'Unaligned Projection Window 2')
        plot(aligned_window2, num_steps, 'Aligned Projection Window 2')
        plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
#    test_windows(width=2e-2, num_steps=21, plot=plot_window_2d)
#    test_window_distance(width=2e-2, num_steps=21)
#    test_window_curvature(width=2e-2, num_steps=21, plot=plot_window_2d)
#    test_window_correlation(width=2e-2, num_steps=21)
    generate_window_for_figure(width=5e-2, num_steps=11)
