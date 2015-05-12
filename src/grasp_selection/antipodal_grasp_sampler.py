import logging
import matplotlib.pyplot as plt
import numpy as np
import IPython
import time

import openravepy as rave

import grasp
import graspable_object
from grasp import ParallelJawPtGrasp3D
import pr2_grasp_checker as pgc
import sdf_file

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
        self.n_cone_faces = config['n_cone_faces']
        self.num_samples = config['num_samples']
        self.dir_prior = config['dir_prior']
        self.alpha_thresh = config['alpha_thresh']
        self.rho_thresh = config['rho_thresh']
        self.vis = config['vis_antipodal']
        self.min_num_grasps = config['min_num_grasps']
        self.alpha_inc = config['alpha_inc']
        self.rho_inc = config['rho_inc']

    def compute_friction_cone(self, contact, graspable):
        """
        Computes the friction cone and normal for a contact point.
        Params:
            contact - numpy 3 array of the start point
            graspable - a GraspableObject3D
        Returns:
            cone_support - numpy array where each column is a vector on the cone
            normal - direction vector
            success - False when cone can't be computed
        """
        contact = tuple(np.around(contact).astype(np.uint16))
        gx, gy, gz = graspable.sdf.gradients

        grad = np.array([gx[contact], gy[contact], gz[contact]])
        grad = grad.reshape((3, 1))
        if np.all(grad == 0):
            return None, None, False

        normal = grad / np.linalg.norm(grad)
        U, _, _ = np.linalg.svd(normal)

        # U[:, 1:] spans the tanget plane at the contact
        t1, t2 = U[:, 1], U[:, 2]
        tan_len = self.friction_coef
        force = np.squeeze(normal) # shape of (3,) rather than (3, 1)

        cone_support = np.zeros((3, 4 * self.n_cone_faces))
        support_index = 0
        ts = np.linspace(0, 1, self.n_cone_faces + 1)

        for t in ts:
            # find convex combinations of tangent vectors
            tan_dir = t * t1 + (1 - t) * t2
            tan_dir = tan_dir / np.linalg.norm(tan_dir)
            tan_vec = tan_len * tan_dir

            cone_support[:, support_index] = -(force + tan_vec)
            support_index += 1
            cone_support[:, support_index] = -(force - tan_vec)
            support_index += 1

        for t in ts[1:-1]:
            tan_dir = t * t1 - (1 - t) * t2
            tan_dir = tan_dir / np.linalg.norm(tan_dir)
            tan_vec = tan_len * tan_dir

            cone_support[:, support_index] = -(force + tan_vec)
            support_index += 1
            cone_support[:, support_index] = -(force - tan_vec)
            support_index += 1

        return cone_support, normal, False

    def sample_from_cone(self, cone):
        """
        Samples points from within the cone.
        Params:
            cone - friction cone's supports
        Returns:
            v_samples - list of self.num_samples vectors in the cone
        """
        num_faces = cone.shape[1]
        v_samples = np.empty((self.num_samples, 3))
        for i in range(self.num_samples):
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
        ap_grasps = []
        surface_points, _ = graspable.sdf.surface_points()
        for x1 in surface_points:
            # compute friction cone faces
            cone1, n1, failed = \
                self.compute_friction_cone(x1, graspable)
            if failed:
                continue

            v_samples = self.sample_from_cone(cone1)
            for v in v_samples:
                
                if vis:
                    plt.clf()
                    h = plt.gcf()
                    plt.ion()
                    ax = plt.gca(projection = '3d')
                    for i in range(cone1.shape[1]):
                        ax.scatter(x1[0] - cone1[0], x1[1] - cone1[1], x1[2] - cone1[2], s = 50, c = u'm') 
#                    plt.draw()

                # start searching for contacts
                grasp, x2 = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(graspable, x1, v, self.grasp_width, vis = vis)
                v_true = grasp.axis

                # compute friction cone for contact 2
                cone2, n2, failed = self.compute_friction_cone(x2, graspable)
                if failed:
                    continue

                if vis:
                    ax = plt.gca(projection = '3d')
                    for i in range(cone1.shape[1]):
                        ax.scatter(x2[0] - cone2[0], x2[1] - cone2[1], x2[2] - cone2[2], s = 50, c = u'm') 
                    ax.set_xlim3d(0, graspable.sdf.dims_[0])
                    ax.set_ylim3d(0, graspable.sdf.dims_[1])
                    ax.set_zlim3d(0, graspable.sdf.dims_[2])
                    plt.draw()
                   
                    time.sleep(0.01)
#                    plt.ioff()

                # check friction cone
                in_cone1, alpha1 = self.within_cone(cone1, n1, v_true.T)
                in_cone2, alpha2 = self.within_cone(cone2, n2, -v_true.T)
                if in_cone1 and in_cone2:
                    #if vis:
                    #    plt.ioff()
                    #    plt.show()

                    x1_world, x2_world = grasp.endpoints()
                    sdf_centroid = graspable.sdf.center_world()
                    rho1 = np.linalg.norm(x1_world - sdf_centroid)
                    rho2 = np.linalg.norm(x2_world - sdf_centroid)

                    antipodal_grasp = AntipodalGraspParams(graspable, grasp, alpha1, alpha2, rho1, rho2)
                    ap_grasps.append(antipodal_grasp)

        # load openrave
        rave.raveSetDebugLevel(rave.DebugLevel.Error)
        e = rave.Environment()
        e.Load(pgc.PR2_MODEL_FILE)
        e.SetViewer("qtcoin")
        r = e.GetRobots()[0]
        grasp_checker = pgc.PR2GraspChecker(e, r, 'data/test/meshes/Co_clean.obj')

        # go back through grasps and threshold            
        grasps = []
        alpha_thresh = self.alpha_thresh
        rho_thresh = self.rho_thresh * graspable.sdf.max_dim()
        while len(grasps) < self.min_num_grasps:
            # prune grasps above thresholds
            grasps = []
            pr2_grasps = []
            for ap_grasp in ap_grasps:
                if max(ap_grasp.alpha1, ap_grasp.alpha2) < alpha_thresh and \
                        max(ap_grasp.rho1, ap_grasp.rho2) < rho_thresh:
                    # convert grasps to PR2 gripper poses
                    grasps.append(ap_grasp.grasp)
                    pr2_grasps.extend(ap_grasp.grasp.pr2_gripper_poses(graspable))

            # prune grasps in collision (TODO)
            pr2_grasps = grasp_checker.prune_grasps_in_collision(pr2_grasps)

            # update alpha and rho thresholds
            alpha_thresh = alpha_thresh * self.alpha_inc
            rho_thresh = alpha_thresh * self.rho_inc

        return grasps, alpha_thresh, rho_thresh

def test_antipodal_grasp_sampling():
    np.random.seed(100)

    h = plt.figure()
    ax = h.add_subplot(111, projection = '3d')

    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()
    graspable = graspable_object.GraspableObject3D(sdf_3d)

    config = {
        'grasp_width': 0.1,
        'friction_coef': 0.25,
        'n_cone_faces': 3,
        'num_samples': 2,
        'dir_prior': 1.0,
        'alpha_thresh': np.pi / 32,
        'rho_thresh': 0.75, # as pct of object max moment
        'vis_antipodal': False,
        'min_num_grasps': 50,
        'alpha_inc': 1.1,
        'rho_inc': 1.1
    }
    sampler = AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    test_antipodal_grasp_sampling()
