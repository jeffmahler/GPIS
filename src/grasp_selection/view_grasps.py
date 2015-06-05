"""
Wrapper module around pr2_grasp_checker to visualize grasps on a graspable.

$ python view_grasps.py dataset graspable grasp_dir
$ python view_grasps.py amazon_picking_challenge feline_greenies_dental_treats \
    results/gce_grasps/amazon_picking_challenge
"""

import argparse

import database
import pr2_grasp_checker as pgc

CONFIG = {
    'database_dir': '/mnt/terastation/shape_data/MASTER_DB_v1/'
}

def visualize(graspable, grasps):
    """Visualize a list of grasps on a graspable.

    graspable - GraspableObject3D instance
    grasps - list of ParallelJawPtGrasp3D or ParallelJawPtPose3D instances
    """
    viewer = pgc.OpenRaveGraspChecker()
    viewer.view_grasps(graspable, grasps)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('graspable')
    parser.add_argument('grasp_dir')
    args = parser.parse_args()

    dataset = database.Dataset(args.dataset, CONFIG)
    graspable = dataset[args.graspable]
    grasps = dataset.load_grasps(args.graspable, args.grasp_dir)

    visualize(graspable, grasps)

if __name__ == '__main__':
    main()
