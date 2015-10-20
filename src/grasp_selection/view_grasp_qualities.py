"""
Script to plot histogram of grasp qualities.

$ python view_grasp_qualities.py dataset graspable [grasp_dir]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

import database

CONFIG = {
    'database_dir': '/mnt/terastation/shape_data/MASTER_DB_v1/'
}

def visualize(grasps):
    """Visualize a list of grasps on a graspable.

    grasps - list of ParallelJawPtGrasp3D or ParallelJawPtPose3D instances
    """
    grasp_qualities = [g.quality for g in grasps]
    num_bins = 100
    bin_edges = np.linspace(0, 1, num_bins+1)
    font_size = 15
    plt.figure()
    n, bins, patches = plt.hist(grasp_qualities, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('graspable')
    parser.add_argument('--grasp-dir')
    args = parser.parse_args()

    dataset = database.Dataset(args.dataset, CONFIG)
    graspable = dataset[args.graspable]
    grasps = dataset.load_grasps(args.graspable, args.grasp_dir)

    visualize(grasps)

if __name__ == '__main__':
    main()
