"""
Tests for patches generation from contacts
Author: Jacky Liang
"""
import random
import os
import argparse
import numpy as np
import logging
import experiment_config as ec
import database as db
import matplotlib.pyplot as plt

import IPython

def visualize_imshow(w1, w2, w1_raw, w2_raw, name, grasp, settings, output=None):
    fig = plt.figure(figsize=(15,15))
    clim = (-1e-2, 1e-2)
    
    fig.suptitle("Patch Visualizations for {0} grasp {1}\n{2}".format(name, grasp, settings))
    
    ax = plt.subplot("221")
    plt.imshow(w1_raw, cmap=plt.cm.binary, interpolation='none', clim=clim)
    ax.set_title("W1 Raw")
    
    ax = plt.subplot("222")
    plt.imshow(w1.proj_win_2d, cmap=plt.cm.binary, interpolation='none', clim=clim)
    ax.set_title("W1 Filtered")
    
    ax = plt.subplot("223")
    plt.imshow(w2_raw, cmap=plt.cm.binary, interpolation='none', clim=clim)
    ax.set_title("W2 Raw")
    
    ax = plt.subplot("224")
    plt.imshow(w2.proj_win_2d, cmap=plt.cm.binary, interpolation='none', clim=clim)
    ax.set_title("W1 Filtered")
    
    if output is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output, "{0}_{1}.png".format(name, grasp)), format='png')
   
def test_view_all_patches(dataset, config, args):
    gripper_name = config['grippers'][0]
    width = 5e-2
    num_steps = 15
    sigma_range = 0.1
    sigma_spatial = 1
    settings = "Patch Settings: Width {0}, n steps {1}, sigma range {2}, sigma spatial {3}".format(width, num_steps, sigma_range, sigma_spatial)
    objs = ['pipe_connector']
    
    done = False
    n_figs_saved = 0
    n_figs_to_save = int(args.num_outputs)
    
    #only go into interactive if no output path specified 
    interactive = args.output is None
    if not interactive:
        logging.info("Interactive mode disabled. Visualizations will be saved directly to " + args.output)

    for obj_key in objs:
        logging.info("Loading object " + obj_key)
        grasps = dataset.grasps(obj_key, gripper=gripper_name)
        obj = dataset.graspable(obj_key)
        n_grasps = len(grasps)
        
        if interactive:
            i = 0
            while True:
                if i >= len(grasps):
                    break
                
                grasp = grasps[i]
                logging.info("Patches for grasp {0}/{1}".format(i+1, n_grasps))
                settings = "Patch Settings: Width {0}, n steps {1}, sigma range {2}, sigma spatial {3}".format(width, num_steps, sigma_range, sigma_spatial)
                logging.info(settings)
                
                pre_blur = []
                w1, w2, c1, c2 = obj.surface_information(grasp, width, num_steps, sigma_range=sigma_range, 
                                            sigma_spatial=sigma_spatial, debug_objs=pre_blur)
                
                logging.info("Visualizing patches")
                visualize_imshow(w1, w2, pre_blur[0], pre_blur[1], obj_key, i, settings)
                
                to_exit = raw_input("Do you wish to exit? [Y/N=default]: ")
                if to_exit.lower() == 'y':
                    done = True
                    break
                
                change_params = raw_input("Do you wish to change parameters? [Y/N=default]: ")
                if change_params.lower() == 'y':
                    while True:
                        new_sigma_range_str = raw_input("New sigma range = ")
                        try:
                            new_sigma_range = float(new_sigma_range_str)
                            break
                        except ValueError:
                            new_sigma_range_str = raw_input("Invalid input! Please enter float new sigma range = ")
                    logging.info("Changing sigma range from {0} to {1}".format(sigma_range, new_sigma_range))
                    sigma_range = new_sigma_range
                    while True:
                        new_sigma_spatial_str = raw_input("New sigma spatial = ")
                        try:
                            new_sigma_spatial = float(new_sigma_spatial_str)
                            break
                        except ValueError:
                            new_sigma_spatial_str = raw_input("Invalid input! Please enter float new sigma spatial = ")
                    logging.info("Changing sigma spatial from {0} to {1}".format(sigma_spatial, new_sigma_spatial))
                    sigma_spatial = new_sigma_spatial
                    
                    i -= 1
                
                i += 1
            
        else:
            for i, grasp in enumerate(grasps):
                if n_figs_to_save != -1 and n_figs_saved >= n_figs_to_save:
                    done = True
                    break

                pre_blur = []
                logging.info("Saving patches for grasp {0}/{1}".format(i+1, n_grasps))
                w1, w2, c1, c2 = obj.surface_information(grasp, width, num_steps, sigma_range=sigma_range, 
                                            sigma_spatial=sigma_spatial, debug_objs=pre_blur)
                visualize_imshow(w1, w2, pre_blur[0], pre_blur[1], obj_key, i, settings, args.output)
                
                n_figs_saved += 1
        if done:
            break

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_patches.yaml'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None)
    parser.add_argument('--num_outputs', default=0)
    args = parser.parse_args()
    
    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_view_all_patches(dataset, config, args)