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
        """Returns the cone support and normal."""
        gx, gy, gz = graspable.sdf.gradients
        todo = np.ones(3)
        return todo, todo, False

    def sample_from_cone(self, cone):
        """Returns a list of points in the cone."""
        todo = 1
        return [todo]

    def within_cone(self, cone, n, v):
        theta = max(np.arccos(n.T * cone))
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

            # sample dirichlet
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

                v_true = ParallelJawPtGrasp3D(x1, x2)

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
        return grasps


def main():
    import sdf_file, graspable_object
    sdf_3d_file_name = 'data/test/sdf/Co_clean_dim_25.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()
    graspable = graspable_object.GraspableObject3D(sdf_3d)

    config = {
        'grasp_width': 0.15,
        'friction_coef': 0.5,
        'n_cone_faces': 2,
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
