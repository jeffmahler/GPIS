"""
Tests for patches generation from contacts
Author: Jacky Liang
"""
import random
import os

import numpy as np
import logging
import experiment_config as ec
import database as db
import matplotlib.pyplot as plt

import IPython

def visualize_imshow(patch):
    plt.imshow(patch, cmap='gray')
    plt.show()
   
def test_view_all_patches(dataset, config):
    gripper_name = config['grippers'][0]
    width = 2e-3
    num_steps = 21
    sigma = 1.5
    
    done = False
    for obj in dataset:
        logging.info("Loading object " + obj.key)
        grasps = dataset.grasps(obj.key, gripper=gripper_name)
        n_grasps = len(grasps)

        i = 0
        while True:
            if i >= len(grasps):
                break
            
            grasp = grasps[i]
            logging.info("Patches for grasp {0}/{1}".format(i+1, n_grasps))
            logging.info("Current settings: Width {0}, n steps {1}, sigma {2}".format(width, num_steps, sigma))
            
            w1, w2, c1, c2 = obj.surface_information(grasp, width, num_steps, sigma=sigma)
            
            logging.info("Visualizing w1")
            visualize_imshow(w1.proj_win_2d)
            logging.info("Visualizing w2")
            visualize_imshow(w2.proj_win_2d)
            
            to_exit = raw_input("Do you wish to exit? [Y/N=default]: ")
            if to_exit.lower() == 'y':
                done = True
                break
            
            change_params = raw_input("Do you wish to change parameters? [Y/N=default]: ")
            if change_params.lower() == 'y':
                while True:
                    new_sigma_str = raw_input("New sigma = ")
                    try:
                        new_sigma = float(new_sigma_str)
                        break
                    except ValueError:
                        new_sigma_str = raw_input("Invalid input! Please enter float new sigma = ")
                logging.info("Changing sigma from {0} to {1}".format(sigma, new_sigma))
                sigma = new_sigma
                i -= 1
            
            i += 1
        if done:
            break

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_patches.yaml'    

    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_view_all_patches(dataset, config)