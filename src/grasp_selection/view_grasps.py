"""
Wrapper module around pr2_grasp_checker to visualize grasps on a graspable.

$ python view_grasps.py dataset graspable grasp_dir
$ python view_grasps.py amazon_picking_challenge feline_greenies_dental_treats \
    results/gce_grasps/amazon_picking_challenge
"""

import argparse

import database
import pr2_grasp_checker as pgc
import numpy as np
import IPython

CONFIG = {
    'database_dir': '/mnt/terastation/shape_data/MASTER_DB_v2/'
}
THETA_RES = 2.0 * np.pi / 20.0

def visualize(graspable, grasps):
    """Visualize a list of grasps on a graspable.

    graspable - GraspableObject3D instance
    grasps - list of ParallelJawPtGrasp3D or ParallelJawPtPose3D instances
    """
    viewer = pgc.OpenRaveGraspChecker()
    # viewer.view_grasps(graspable, grasps)
    viewer.prune_grasps_in_collision(graspable, grasps, delay=1.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('graspable')
    parser.add_argument('--grasp_dir', default = None)
    args = parser.parse_args()

    dataset = database.Dataset(args.dataset, CONFIG)
    graspable = dataset[args.graspable]
    grasps = dataset.load_grasps(args.graspable, args.grasp_dir)

    rotated_grasps = []
    for g in grasps:
        rotated_grasps.extend(g.transform(graspable.tf, THETA_RES))
    visualize(graspable, rotated_grasps)

if __name__ == '__main__':
    main()
