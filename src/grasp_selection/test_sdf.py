"""
Tests for the SDF class
Author: Jeff Mahler
"""
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
try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_collision_checker as gcc
import grasp_sampler as gs
import gripper as gr
import json_serialization as jsons
import kernels
import mayavi_visualizer as mv
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

def test_sdf_surface_normals_grasp(dataset, config):
    # load grasps
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)
    
    # visualize finger close on everythang
    indices = [166]#, 11, 29, 102, 179]
    for ind in indices:
        grasp = grasps[ind]
        contacts_found, contacts = grasp.close_fingers(obj, vis=True)

        if not contacts_found:
            logging.info('Contacts not found')

        as_grid = obj.sdf.transform_pt_obj_to_grid(contacts[0].point)
        normal, c1_pts = obj.sdf.surface_normal(as_grid)
        as_grid = obj.sdf.transform_pt_obj_to_grid(contacts[1].point)
        normal, c2_pts = obj.sdf.surface_normal(as_grid)

        contacts[0].plot_friction_cone(color='y')
        contacts[1].plot_friction_cone(color='c')
        
        ax = plt.gca()
        ax.scatter(c1_pts[:,0], c1_pts[:,1], c1_pts[:,2], c='g', s=120)
        ax.scatter(c2_pts[:,0], c2_pts[:,1], c2_pts[:,2], c='k', s=120)
        ax.set_xlim3d(0, obj.sdf.dims_[0])
        ax.set_ylim3d(0, obj.sdf.dims_[1])
        ax.set_zlim3d(0, obj.sdf.dims_[2])
        plt.show()

def test_sdf_surface_normals(dataset, config, num_test=200, plot=True):
    delta = 0.025
    font_size = 15
    num_bins = 100
    dpi = 200

    sn_delta = 0.05
    init_sn_delta = 1.0
    max_sn_delta = 1.0
    
    output_dir = 'results/test/surface_normals'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # setup param sweep
    normal_devs = {}
    sn_deltas = []
    d = init_sn_delta
    while d <= max_sn_delta:
        normal_devs[d] = []
        sn_deltas.append(d)
        d += sn_delta

    # get normal devs for each object in the dataset
    for obj in dataset:
        logging.info('Testing surface normals for object %s' %(obj.key))
        logging.info('SDF Res %.4f' %(obj.sdf.resolution))

        surface_points, _ = obj.sdf.surface_points()
        np.random.shuffle(surface_points)
        
        tri_centers = obj.mesh.tri_centers()
        tri_normals = obj.mesh.tri_normals()
        tri_centers_and_normals = zip(tri_centers, tri_normals)
        random.shuffle(tri_centers_and_normals)
        tri_centers = [t[0] for t in tri_centers_and_normals]
        tri_normals = [t[1] for t in tri_centers_and_normals]

        if plot:
            mlab.figure()
            mv.MayaviVisualizer.plot_mesh(obj.mesh)
    
        for tri_center, tri_normal in zip(tri_centers[:num_test], tri_normals[:num_test]):
            # compute normal from sdf
            pt_3d = np.array(tri_center)
            normal_3d = np.array(tri_normal)
            pt_grid = obj.sdf.transform_pt_obj_to_grid(pt_3d)

            for d in sn_deltas:
                est_normal_grid, comp_pts = obj.sdf.surface_normal(pt_grid, delta=d)
                est_normal_3d = obj.sdf.transform_pt_grid_to_obj(est_normal_grid, direction=True)
            
                # compute deviation (don't penalize sign since it can't be disambiguated for non-closed meshes)
                if est_normal_3d.dot(normal_3d) < 0:
                    est_normal_3d = -est_normal_3d
                dot_prod = est_normal_3d.dot(normal_3d)
                angle_dev = np.arccos(dot_prod)
                normal_devs[d].append(angle_dev)
                if np.isnan(angle_dev):
                    logging.warning('Found NaN angle')
                    #IPython.embed()

            # plot normals
            if plot:
                normal_plot = np.array([pt_3d,
                                        pt_3d + 0.01 * normal_3d])
                est_normal_plot = np.array([pt_3d,
                                            pt_3d + 0.01 * est_normal_3d])
                mlab.plot3d(normal_plot[:,0], normal_plot[:,1], normal_plot[:,2],
                            tube_radius=0.0005, color=(1,0,0))
                mlab.plot3d(est_normal_plot[:,0], est_normal_plot[:,1], est_normal_plot[:,2],
                            tube_radius=0.0005, color=(0,0,1))
                mlab.points3d(pt_3d[0], pt_3d[1], pt_3d[2], scale_factor=0.0025, color=(0,1,0))
        
        if plot:
            mlab.show()

    logging.info('Compiling results')
    mean_devs = []
    med_devs = []
    for d in sn_deltas:
        normal_dev_arr = np.array(normal_devs[d])
        mean_dev = np.mean(normal_dev_arr)
        med_dev = np.median(normal_dev_arr)
        max_dev = np.max(normal_dev_arr)
        std_dev = np.std(normal_dev_arr)
        mean_devs.append(mean_dev)
        med_devs.append(med_dev)

        logging.info('')
        logging.info('Stats for d=%.2f' %(d))
        logging.info('Mean angle dev (rad) = %.3f' %(mean_dev))
        logging.info('Median angle dev (rad) = %.3f' %(med_dev))
        logging.info('Max angle dev (rad) = %.3f' %(max_dev))
        logging.info('Std angle dev (rad) = %.3f' %(std_dev))
        
        normal_dev_hist, normal_dev_bins = np.histogram(normal_dev_arr,
                                                        bins=num_bins, range=(0,np.pi/2))
        width = (normal_dev_bins[1] - normal_dev_bins[0])
        
        plt.figure()
        plt.bar(normal_dev_bins[:-1], normal_dev_hist, width=width, color='b')
        plt.title('Normal Deviation Histogram For d=%0.2f' %(d), fontsize=font_size)
        plt.xlabel('Deviation (radians)', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)
        plt.xlim(0, np.pi / 2)
        plt.ylim(0, num_test / 2)

        figname = os.path.join(output_dir, 'normal_dev_hist_%.2f.pdf' %(d))
        plt.savefig(figname, dpi=dpi)

    mean_devs = np.array(mean_devs)
    best_mean_ind = np.where(mean_devs == np.min(mean_devs))[0][0]
    med_devs = np.array(med_devs)
    best_med_ind = np.where(med_devs == np.min(med_devs))[0][0]
    
    logging.info('Best mean dev was %.2f for d=%.2f' %(mean_devs[best_mean_ind], sn_deltas[best_mean_ind]))
    logging.info('Best med dev was %.2f for d=%.2f' %(med_devs[best_med_ind], sn_deltas[best_med_ind]))

    IPython.embed()

    """
    plt.figure()
    ax = plt.gca(projection='3d')
    obj.sdf.scatter()

    normals = []
    for pt in surface_points.tolist()[:num_test]:
        normal, comp_pts = obj.sdf.surface_normal(pt)
        pt = np.array(pt)
        normal = np.array(normal)
        pt_3d = obj.sdf.transform_pt_grid_to_obj(pt)
        normal_3d = obj.sdf.transform_pt_grid_to_obj(normal, direction=True)
        normal_plot = np.array([pt_3d,
                                pt_3d + 0.01 * normal_3d])
        mlab.plot3d(normal_plot[:,0], normal_plot[:,1], normal_plot[:,2],
                    tube_radius=0.001, color=(1,0,0))
        mlab.points3d(pt_3d[0], pt_3d[1], pt_3d[2], scale_factor=0.0025, color=(0,1,0))
        
        t = delta
        norm_plot = []
        while t < 2.0:
            norm_plot.append(np.array(pt) + t*normal)
            t += delta
        norm_plot = np.array(norm_plot)
        ax.scatter(comp_pts[:,0], comp_pts[:,1], comp_pts[:,2], c='g', s=120)
        ax.scatter(norm_plot[:,0], norm_plot[:,1], norm_plot[:,2], s=25, c='g')

    mlab.show()
    """

    #ax.set_xlim3d(0, obj.sdf.dims_[0])
    #ax.set_ylim3d(0, obj.sdf.dims_[1])
    #ax.set_zlim3d(0, obj.sdf.dims_[2])
    #plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_sdf.yaml'    

    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    test_sdf_surface_normals_grasp(dataset, config)
