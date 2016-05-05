from __future__ import print_function

import numpy as np

import unittest
from test_dexnet import SRC_PATH

import sys
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import grasp
import graspable_object
import obj_file
import quality
import sdf_file

class QualityTestCase(unittest.TestCase):
    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    def setUp(self):
        np.random.seed(100)

        # Construct GraspableObject
        sf = sdf_file.SdfFile(self.sdf_3d_file_name)
        sdf_3d = sf.read()
        of = obj_file.ObjFile(self.mesh_file_name)
        mesh_3d = of.read()
        self.graspable = graspable_object.GraspableObject3D(sdf_3d, mesh_3d)

        # Construct Grasps
        self.grasps = []
        z_vals = np.linspace(-0.025, 0.025, 3)
        for z_val in z_vals:
            grasp_center = np.array([0, 0, z_val])
            grasp_axis = np.array([0, 1, 0])
            grasp_width = 0.1
            grasp_params = grasp.ParallelJawPtGrasp3D.configuration_from_params(
                grasp_center, grasp_axis, grasp_width)
            g = grasp.ParallelJawPtGrasp3D(grasp_params)
            self.grasps.append(g)

    def test_quality_metrics(self, vis=False):
        metrics = ['force_closure', 'force_closure_qp', 'min_singular',
                   'wrench_volume', 'grasp_isotropy', 'ferrari_canny_L1']
        for i, g in enumerate(self.grasps):
            print('Evaluating grasp {}'.format(i))
            for metric in metrics:
                q = quality.PointGraspMetrics3D.grasp_quality(
                    g, self.graspable, metric, soft_fingers=True)
                print('Grasp quality according to {}: {}'.format(metric, q))

            if vis:
                cf, contacts = g.close_fingers(self.graspable, vis=True)
                for contact in contacts:
                    contact.plot_friction_cone(color='y', scale=-2.0)
                plt.show()
                IPython.embed()
