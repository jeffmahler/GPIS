"""
Class for sampling grasps using an antipodallity heuristic derived from "Computing Parallel-Jaw Grasps"
by Smith et al., ICRA 1999 *from Ken's old ALPHA lab

Author: Jeff Mahler
"""
import logging
import matplotlib.pyplot as plt
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')
import numpy as np
import random
import time

import openravepy as rave

import contacts
import experiment_config as ec
import grasp
import graspable_object
from grasp import ParallelJawPtGrasp3D
import grasp_sampler as gs
import obj_file
import pr2_grasp_checker as pgc
import quality as pgq
import sdf_file

import IPython

# basically a bucket for todo vis code, don't want to lose it
def vis_graspable_grasp(graspable, grasp):

    graspable.mesh.visualize()
    mv.axes()

    x, v = graspable.sdf.surface_points(grid_basis = False)
    mv.points3d(x[:,0], x[:,1], x[:,2], scale_factor = 0.01)

    g1, g2 = grasp.endpoints()
    mv.points3d(g1[0], g1[1], g1[2], scale_factor = 0.01, color=(0.5,0.5,0.5))
    mv.points3d(g2[0], g2[1], g2[2], scale_factor = 0.01, color=(0.5,0.5,0.5))
    mv.points3d(grasp.center[0], grasp.center[1], grasp.center[2], scale_factor = 0.01, color=(0.5,0.5,0.5))
    mv.show()

class AntipodalGraspParams:
    """ Struct to hold antipodal grasp with statistics """
    def __init__(self, obj, grasp, alpha1, alpha2, rho1, rho2):
        self.obj = obj
        self.grasp = grasp
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.rho1 = rho1
        self.rho2 = rho2

class AntipodalGraspSampler(gs.ExactGraspSampler):
    def sample_from_cone(self, cone, num_samples=1):
        """
        Samples points from within the cone.
        Params:
            cone - friction cone's supports
        Returns:
            v_samples - list of self.num_samples vectors in the cone
        """
        num_faces = cone.shape[1]
        v_samples = np.empty((num_samples, 3))
        for i in range(num_samples):
            lambdas = np.random.gamma(self.dir_prior, self.dir_prior, num_faces)
            lambdas = lambdas / sum(lambdas)
            v_sample = lambdas * cone
            v_samples[i, :] = np.sum(v_sample, 1)
        return v_samples

    def within_cone(self, cone, n, v):
        """
        Returns True if a grasp will not slip due to friction.
        Params:
            cone - friction cone's supports
            n - outward pointing normal vector at c1
            v - direction vector between c1 and c2
        Returns:
            in_cone - True if alpha is within the cone
            alpha - the angle between the normal and v
        """
        if (v.dot(cone) < 0).any(): # v should point in same direction as cone
            return False, None
        f = -n / np.linalg.norm(n)
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= np.arctan(self.friction_coef), alpha

    def perturb_point(self, x, scale):
        """ Uniform random perturbations to a point """
        x_samp = x + (scale / 2.0) * (np.random.rand(3) - 0.5)
        return x_samp

    def _generate_grasps(self, graspable, num_grasps,
                         check_collisions=False, vis=False):
        """Returns a list of candidate grasps for graspable object.
        Params:
            graspable - (GraspableObject3D) the object to grasp
            num_grasps - currently unused TODO
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        # get surface points
        ap_grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)
        random.shuffle(surface_points)
        shuffled_surface_points = surface_points[:min(self.max_num_surface_points_, len(surface_points))]
        logging.info('Num surface: %d' %(len(surface_points)))
        
        for x_surf in shuffled_surface_points:
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = self.perturb_point(x_surf, graspable.sdf.resolution)

                # compute friction cone faces
                c1 = contacts.Contact3D(graspable, x1, in_direction=None)
                cone_succeeded, cone1, n1 = c1.friction_cone(self.num_cone_faces, self.friction_coef)
                if not cone_succeeded:
                    continue
                cone_time = time.clock()

                # sample grasp axes from friction cone
                v_samples = self.sample_from_cone(cone1, num_samples=1)
                sample_time = time.clock()

                for v in v_samples:
                    if vis:
                        x1_grid = graspable.sdf.transform_pt_obj_to_grid(x1)
                        cone1_grid = graspable.sdf.transform_pt_obj_to_grid(cone1, direction=True)
                        plt.clf()
                        h = plt.gcf()
                        plt.ion()
                        ax = plt.gca(projection = '3d')
                        for i in range(cone1.shape[1]):
                            ax.scatter(x1_grid[0] - cone1_grid[0], x1_grid[1] - cone1_grid[1], x1_grid[2] - cone1_grid[2], s = 50, c = u'm')

                    # start searching for contacts
                    grasp, c2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.grasp_width, vis = vis)
                    if grasp is None or c2 is None:
                        continue

                    # make sure grasp is wide enough
                    x2 = c2.point
                    if np.linalg.norm(x1 - x2) < self.min_contact_dist:
                        continue

                    v_true = grasp.axis
                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = c2.friction_cone(self.num_cone_faces, self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        plt.figure()
                        ax = plt.gca(projection='3d')
                        c1_proxy = c1.plot_friction_cone(color='m')
                        c2_proxy = c2.plot_friction_cone(color='y')
                        ax.view_init(elev=5.0, azim=0)
                        plt.show(block=False)
                        time.sleep(0.5)
                        plt.close() # lol

                    # check friction cone
                    in_cone1, alpha1 = self.within_cone(cone1, n1, v_true.T)
                    in_cone2, alpha2 = self.within_cone(cone2, n2, -v_true.T)
                    within_cone_time = time.clock()

                    # add points if within friction cone
                    if in_cone1 and in_cone2:
                        # get moment arms
                        x1_world, x2_world = grasp.endpoints()
                        rho1 = np.linalg.norm(graspable.moment_arm(x1_world))
                        rho2 = np.linalg.norm(graspable.moment_arm(x2_world))

                        antipodal_grasp = AntipodalGraspParams(graspable, grasp, alpha1, alpha2, rho1, rho2)
                        ap_grasps.append(antipodal_grasp)

        # randomly sample max num grasps from total list
        max_grasp_index = min(len(ap_grasps), self.max_num_grasps)
        random.shuffle(ap_grasps)
        ap_grasps = ap_grasps[:max_grasp_index]

        # load openrave
        if check_collisions:
            rave.raveSetDebugLevel(rave.DebugLevel.Error)
            grasp_checker = pgc.OpenRaveGraspChecker(view=vis)

        # go back through grasps and threshold
        grasps = []
        pr2_grasps = []
        alpha_thresh = self.alpha_thresh
        rho_thresh = self.rho_thresh * graspable.sdf.max_dim()
        while len(ap_grasps) > 0 and len(grasps) < self.min_num_grasps and len(pr2_grasps) < self.min_num_collision_free and \
                alpha_thresh < np.pi / 2:
            # prune grasps above thresholds
            next_ap_grasps = []
            for ap_grasp in ap_grasps:
                if max(ap_grasp.alpha1, ap_grasp.alpha2) < alpha_thresh and \
                        max(ap_grasp.rho1, ap_grasp.rho2) < rho_thresh:
                    # convert grasps to PR2 gripper poses
                    rotated_grasps = ap_grasp.grasp.transform(graspable.tf, self.theta_res)

                    # prune collision grasps if necessary
                    if check_collisions:
                        rotated_grasps = grasp_checker.prune_grasps_in_collision(graspable, rotated_grasps, auto_step=True, delay=0.0)

                    # only add grasp if at least 1 is collision free
                    if len(rotated_grasps) > 0:
                        grasps.append(ap_grasp.grasp)
                        pr2_grasps.extend(rotated_grasps)
                else:
                    next_ap_grasps.append(ap_grasp)

            # update alpha and rho thresholds
            alpha_thresh = alpha_thresh + self.alpha_inc #np.arctan(friction_coef)
            rho_thresh = rho_thresh + self.rho_inc
            ap_grasps = list(next_ap_grasps)

        return grasps

def test_antipodal_grasp_sampling(vis=False):
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    config_file = 'cfg/correlated.yaml'
    config = ec.ExperimentConfig(config_file)
    sampler = AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    if vis:
        plt.close() # lol
        for i, grasp in enumerate(grasps, 1):
            plt.figure()
            ax = plt.gca(projection='3d')
            found, (c1, c2) = grasp.close_fingers(graspable)
            c1_proxy = c1.plot_friction_cone(color='m')
            c2_proxy = c2.plot_friction_cone(color='y')
            ax.set_xlim([5, 20])
            ax.set_ylim([5, 20])
            ax.set_zlim([5, 20])
            plt.title('Grasp %d' %(i))
            plt.axis('off')
            plt.show(block=False)
            for angle in range(0, 360, 10):
                ax.view_init(elev=5.0, azim=angle)
                plt.draw()
            plt.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_antipodal_grasp_sampling()
