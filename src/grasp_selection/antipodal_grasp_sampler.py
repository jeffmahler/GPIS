import logging
import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np
import time

import openravepy as rave

import grasp
import graspable_object
from grasp import ParallelJawPtGrasp3D
import obj_file
import pr2_grasp_checker as pgc
import quality as pgq
import sdf_file

import IPython

class AntipodalGraspParams:
    """ Struct to hold antipodal grasp with statistics """
    def __init__(self, obj, grasp, alpha1, alpha2, rho1, rho2):
        self.obj = obj
        self.grasp = grasp
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.rho1 = rho1
        self.rho2 = rho2
        
class AntipodalGraspSampler(object):
    def __init__(self, config):
        self._configure(config)

    def _configure(self, config):
        """Configures the grasp generator."""
        self.grasp_width = config['grasp_width']
        self.friction_coef = config['friction_coef']
        self.num_cone_faces = config['num_cone_faces']
        self.num_samples = config['num_samples']
        self.dir_prior = config['dir_prior']
        self.alpha_thresh = config['alpha_thresh']
        self.rho_thresh = config['rho_thresh']
        self.vis = config['vis_antipodal']
        self.min_num_grasps = config['min_num_grasps']
        self.alpha_inc = config['alpha_inc']
        self.rho_inc = config['rho_inc']

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
        f = -n / np.linalg.norm(n)
        cone_norms = np.linalg.norm(cone, axis = 0)
        theta = np.max(np.arccos(f.T.dot(cone) / cone_norms));
        alpha = np.arccos(f.T.dot(v) / np.linalg.norm(v))
        return alpha <= theta, alpha

    def generate_grasps(self, graspable, vis = False):
        """Returns a list of candidate grasps for graspable object.
        Params: GraspableObject3D
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        # TODO: remove this!
        mesh_name = 'data/test/meshes/Co_clean.obj'

        # get surface points
        ap_grasps = []
        surface_points, _ = graspable.sdf.surface_points(grid_basis=False)

        for x1 in surface_points:
            start_time = time.clock()

            # perturb grasp for num samples
            for i in range(self.num_samples):
                # perturb contact (TODO: sample in tangent plane to surface)
                x1 = x1 + (graspable.sdf.resolution / 2.0) * (np.random.rand(3) - 0.5)

                # compute friction cone faces
                cone_succeeded, cone1, n1 = \
                    graspable.contact_friction_cone(x1, num_cone_faces = self.num_cone_faces, friction_coef = self.friction_coef)
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
                    grasp, x2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.grasp_width, vis = vis)
                    v_true = grasp.axis

                    # compute friction cone for contact 2
                    cone_succeeded, cone2, n2 = graspable.contact_friction_cone(x2, num_cone_faces = self.num_cone_faces,
                                                                                friction_coef = self.friction_coef)
                    if not cone_succeeded:
                        continue

                    if vis:
                        x2_grid = graspable.sdf.transform_pt_obj_to_grid(x2)
                        cone2_grid = graspable.sdf.transform_pt_obj_to_grid(cone2, direction=True)

                        ax = plt.gca(projection = '3d')
                        for i in range(cone1.shape[1]):
                            ax.scatter(x2_grid[0] - cone2_grid[0], x2_grid[1] - cone2_grid[1], x2_grid[2] - cone2_grid[2], s = 50, c = u'm') 
                        ax.set_xlim3d(0, graspable.sdf.dims_[0])
                        ax.set_ylim3d(0, graspable.sdf.dims_[1])
                        ax.set_zlim3d(0, graspable.sdf.dims_[2])
                        plt.draw()

                        time.sleep(0.01)

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

                        #logging.error('Cone time: %f' %(cone_time - start_time))
                        #logging.error('Sample time: %f' %(sample_time - cone_time))
                        #logging.error('Within cone time: %f' %(within_cone_time - sample_time))

        # load openrave
        """
        rave.raveSetDebugLevel(rave.DebugLevel.Error)
        e = rave.Environment()
        e.Load(pgc.PR2_MODEL_FILE)
        e.SetViewer("qtcoin")
        r = e.GetRobots()[0]
        grasp_checker = pgc.PR2GraspChecker(e, r, mesh_name)
        """

        # go back through grasps and threshold            
        grasps = []
        alpha_thresh = self.alpha_thresh
        rho_thresh = self.rho_thresh * graspable.sdf.max_dim()
        while len(grasps) < self.min_num_grasps:
            # prune grasps above thresholds
            grasps = []
#            pr2_grasps = []
            for ap_grasp in ap_grasps:
                if max(ap_grasp.alpha1, ap_grasp.alpha2) < alpha_thresh and \
                        max(ap_grasp.rho1, ap_grasp.rho2) < rho_thresh:
                    # convert grasps to PR2 gripper poses
                    grasps.append(ap_grasp.grasp)
#                    pr2_grasps.extend(ap_grasp.grasp.pr2_gripper_poses(graspable))

            # prune grasps in collision (TODO)
#            pr2_grasps = grasp_checker.prune_grasps_in_collision(pr2_grasps, vis = True)

            # update alpha and rho thresholds
            alpha_thresh = alpha_thresh * self.alpha_inc
            rho_thresh = alpha_thresh * self.rho_inc
    
        """
        for grasp in grasps:
            q = pgq.PointGraspMetrics3D.grasp_quality(grasp, graspable, "force_closure", soft_fingers = True)
            print "Quality", q
            if q > 0:
                h = mv.figure(1)
                mv.clf(h)
                grasp.visualize(graspable)
                graspable.visualize()
                mv.draw()
                time.sleep(1)
        """

        return grasps, alpha_thresh, rho_thresh

def test_antipodal_grasp_sampling():
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = graspable_object.GraspableObject3D(sdf_3d, mesh=m)

    config = {
        'grasp_width': 0.1,
        'friction_coef': 0.5,
        'num_cone_faces': 8,
        'num_samples': 4,
        'dir_prior': 1.0,
        'alpha_thresh': np.pi / 32,
        'rho_thresh': 0.75, # as pct of object max moment
        'vis_antipodal': False,
        'min_num_grasps': 100,
        'alpha_inc': 1.1,
        'rho_inc': 1.1
    }
    sampler = AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.error('Antipodal grasp candidate generation took %f sec' %(duration))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    test_antipodal_grasp_sampling()
