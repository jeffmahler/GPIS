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

    # output_db_filename = os.path.join(args.output_dest, config['results_database_name'])
    # output_db = db.Hdf5Database(output_db_filename, config, access_level=db.WRITE_ACCESS)

    zeke_gripper = gripper.RobotGripper.load("zeke")

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # make dataset output directory
        dest = os.path.join(args.output_dest, dataset.name)
        try:
            os.makedirs(dest)
        except os.error:
            pass
        output_ds = output_db.create_dataset(dataset_name)

        # label each object in the dataset with grasps
  
        obj_grasps = dataset.grasps(obj.key, "zeke")
        for grasp in obj_grasps:
            gripper_pose = grasp.gripper_transform(zeke_gripper)
            print(gripper_pose)






    database.close()
    output_db.close()






