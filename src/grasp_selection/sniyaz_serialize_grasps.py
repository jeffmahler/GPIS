import argparse
import copy
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import database as db
import experiment_config as ec
import gripper

if __name__ == '__main__':

    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('output_dest')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)


    zeke_gripper = gripper.RobotGripper.load("zeke")

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)


        # label each object in the dataset with grasps
  
        obj_grasps = dataset.grasps("textured-0008192", "pr2")
        for grasp in obj_grasps:
            gripper_pose = grasp.gripper_transform(zeke_gripper)
            print(gripper_pose.pose.matrix)


    database.close()






