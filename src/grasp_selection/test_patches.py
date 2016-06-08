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
import similarity_tf as stf
from mayavi_visualizer import MayaviVisualizer as mvis
import matplotlib.pyplot as plt
import similarity_tf as stf
import mayavi.mlab as mv
import mayavi.mlab as mlab
import IPython

def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def visualize_patches(w1, w2, w1_raw, w2_raw, name, grasp, settings, output=None):
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
    ax.set_title("W2 Filtered")

    plt.colorbar()    
    if output is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output, "patches_{0}_{1}.png".format(name, grasp)), format='png')

def set_mayavi_scene_for_contacts(obj, grasp, c1, c2):
    alpha = 0.025
    tube_radius = 0.001
    scale = 0.0025
    T_c1_obj = c1.reference_frame()
    T_c2_obj = c2.reference_frame()
    T_obj_world = stf.SimilarityTransform3D(from_frame='world', to_frame='obj')
    T_c1_world = T_c1_obj.dot(T_obj_world)
    T_c2_world = T_c2_obj.dot(T_obj_world)

    mvis.plot_mesh(obj.mesh, T_obj_world)
    mvis.plot_grasp(grasp, T_obj_world, alpha=alpha)
    mvis.plot_pose(T_c1_world, alpha=alpha, tube_radius=tube_radius, center_scale=scale)
    mvis.plot_pose(T_c2_world, alpha=alpha, tube_radius=tube_radius, center_scale=scale)
    mvis.plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=scale)

    return T_c1_world, T_c2_world
     
def visualize_contacts(T_c1_world, T_c2_world, obj_key, grasp_num, 
                       width, num_steps, w1, w2, output=None):
    res = float(width) / float(num_steps)
    scale = 0.0025
    points1, points2 = mvis.plot_patches_contacts(T_c1_world, T_c2_world, w1, w2, res, scale)

    if output is None:
        mlab.show()
    else:
        num_views = 10
        cam_dist = 0.5
        delta_view = 360. / num_views
        for i in range(num_views):
            az = i * delta_view
            mv.view(azimuth=az, focalpoint=(0,0,0), distance=cam_dist)
            mv.savefig(os.path.join(output, "contacts_{0}_grasp_{1}_{2}.png".format(obj_key, grasp_num, i)))

    return points1, points2

def test_view_all_patches(dataset, config, args):
    gripper_name = config['grippers'][0]
    width = 5e-2
    num_steps = 15
    sigma_range = 0.01
    sigma_spatial = 1
    back_up = 0.025
    settings = "Patch Settings: Width {0}, n steps {1}, sigma range {2}, sigma spatial {3}".format(width, num_steps, sigma_range, sigma_spatial)
    objs = ['pipe_connector']
    
    done = False
    n_grasps_saved = 0
    n_grasps_to_save = int(args.num_grasps)

    mv.figure(size=(1000, 1000))
    
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
                w1, w2, c1, c2 = obj.surface_information(grasp, width, num_steps,
                                                         back_up=back_up, sigma_range=sigma_range, 
                                                         sigma_spatial=sigma_spatial, debug_objs=pre_blur)
                logging.info("Visualizing patches")
                mv.clf()
                T_c1_world, T_c2_world = set_mayavi_scene_for_contacts(obj, grasp, c1, c2)

                #visualize_patches(w1, w2, pre_blur[0], pre_blur[1], obj_key, i, settings)
                points1, points2 = visualize_contacts(T_c1_world, T_c2_world, obj_key, i, width, num_steps, w1, w2, c1, c2)
                points1.remove()
                points2.remove()

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
            obj_output_path = os.path.join(args.output, obj_key)
            for i, grasp in enumerate(grasps):
                if n_grasps_to_save != -1 and n_grasps_saved >= n_grasps_to_save:
                    done = True
                    break

                grasp_output_path = os.path.join(obj_output_path, "grasp_{0}".format(i))
                ensure_dir_exists(grasp_output_path)

                pre_blur = []
                logging.info("Saving patches for grasp {0}/{1}".format(i+1, n_grasps))
                w1, w2, c1, c2 = obj.surface_information(grasp, width, num_steps, sigma_range=sigma_range, 
                                            sigma_spatial=sigma_spatial, debug_objs=pre_blur)
                mv.clf()
                T_c1_world, T_c2_world = set_mayavi_scene_for_contacts(obj, grasp, c1, c2)
                #visualize_patches(w1, w2, pre_blur[0], pre_blur[1], obj_key, i, settings, grasp_output_path)
                points1, points2 = visualize_contacts(T_c1_world, T_c2_world, obj_key, i, width, num_steps, w1, w2, grasp_output_path)
                points1.remove()
                points2.remove()

                n_grasps_saved += 1
        if done:
            break

        mv.clf()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_patches.yaml'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None)
    parser.add_argument('--num_grasps', default=0)
    args = parser.parse_args()
    
    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_view_all_patches(dataset, config, args)
