import numpy as np

import grasp
from grasp import ParallelJawPtGrasp3D

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
        contact = tuple(np.around(contact))
        gx, gy, gz = graspable.sdf.gradients

        grad = np.reshape([gx[contact], gy[contact], gz[contact]], (3, 1))
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
            n - normal vector at c1
            v - direction vector between c1 and c2
        Returns:
            in_cone - True if alpha is within the cone
            alpha - the angle between the normal and v
        """
        theta = np.max(np.arcos(n.T * cone));
        alpha = np.arcos(n.T * v)
        return alpha <= theta, alpha

    def generate_grasps(self, graspable):
        """Returns a list of candidate grasps for graspable object.
        Params: GraspableObject3D
        Returns:
            list of ParallelJawPtGrasp3D objects
        """
        grasps = []
        surface_points, _ = graspable.sdf.surface_points()
        for x1 in surface_points:
            # compute friction cone faces
            cone1, n1, failed = \
                self.compute_friction_cone(x1, graspable)
            if failed:
                continue

            v_samples = self.sample_from_cone(cone1)
            for v in v_samples:
                # start searching for contacts
                line_of_action = ParallelJawPtGrasp3D.create_line_of_action(
                    x1, v, self.grasp_width, graspable, self.grasp_width / graspable.sdf.resolution
                )
                found, x2 = ParallelJawPtGrasp3D.find_contact(
                    line_of_action, graspable
                )
                if not found:
                    continue

                v_true = ParallelJawPtGrasp3D.grasp_axis_from_endpoints(x1, x2)

                # compute friction cone for contact 2
                cone2, n2, failed = self.compute_friction_cone(x2, graspable)
                if failed:
                    continue

                # check friction cone
                in_cone1, alpha1 = self.within_cone(cone1, n1, -v_true.T)
                in_cone2, alpha2 = self.within_cone(cone2, n2, v_true.T)
                if in_cone1 and in_cone2:
                    sdf_centroid = np.mean(surface_points)
                    rho1 = np.linalg.norm(x1 - sdf_centroid)
                    rho2 = np.linalg.norm(x2 - sdf_centroid)

                    if max(alpha1, alpha2) < self.alpha_thresh and max(rho1, rho2) < self.rho_thresh:
                        sample = ParallelJawPtGrasp3D.grasp_from_contact_and_axis_on_grid(
                            graspable, x1, v_true, self.grasp_width
                        )
                        grasps.append(sample)
        # import IPython; IPython.embed()
        return grasps


def main():
    import sdf_file, graspable_object
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()
    graspable = graspable_object.GraspableObject3D(sdf_3d)

    config = {
        'grasp_width': 5.0,
        'friction_coef': 0.5,
        'n_cone_faces': 3,
        'num_samples': 2,
        'dir_prior': 1.0,
        'alpha_thresh': np.pi / 32,
        'rho_thresh': 0.9,
        'vis_antipodal': False
    }
    sampler = AntipodalGraspSampler(config)
    grasps = sampler.generate_grasps(graspable)

if __name__ == '__main__':
    main()
