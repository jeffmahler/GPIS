"""
Tests for grasp quality
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
import models
import mayavi_visualizer as mv
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import similarity_tf as stf
import termination_conditions as tc
import tfx

GRAVITY_ACCEL = 9.81

def test_ferrari_canny_L1(dataset, config):
    # load grasps
    num_bins = 25
    font_size = 15
    obj_name = 'pipe_connector'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)
    
    # visualize finger close on everythang
    vis_indices = [166]
    fc = []
    frcl = []
    for i, grasp in enumerate(grasps[166:]):
        # Compute qualitya
        logging.debug('FC for hard fingers')
        eps_hf = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'ferrari_canny_L1', friction_coef=config['friction_coef'],
                                                           num_cone_faces=config['num_cone_faces'], soft_fingers=False)
        logging.debug('FC for soft fingers')
        eps = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'ferrari_canny_L1', friction_coef=config['friction_coef'],
                                                       num_cone_faces=config['num_cone_faces'], soft_fingers=True)
        logging.debug('Force closure')
        fcl = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'force_closure', friction_coef=config['friction_coef'],
                                                       num_cone_faces=config['num_cone_faces'], soft_fingers=True)
        fc.append(eps)
        frcl.append(fcl)

        logging.info('Grasp %d ferrari canny = %.5f, force closure = %d' %(grasp.grasp_id, eps, fcl))

        # Check that hard finger fc is basically zero
        assert(eps_hf < 1e-5)

        # Visualize
        if grasp.grasp_id in vis_indices:
            contacts_found, contacts = grasp.close_fingers(obj, vis=True)
            if not contacts_found:
                logging.info('Contacts not found')
                continue

            contacts[0].plot_friction_cone(color='y')
            contacts[1].plot_friction_cone(color='c')
        
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

    # display top grasps
    gripper = gr.RobotGripper.load(gripper_name)
    grasps_and_qualities = zip(grasps, fc)
    grasps_and_qualities.sort(key = lambda x: x[1], reverse=True)
    for grasp, f in grasps_and_qualities[:5]:
        logging.info('Grasp %d quality %f' %(grasp.grasp_id, f))
        mlab.figure()
        mv.MayaviVisualizer.plot_mesh(obj.mesh)
        mv.MayaviVisualizer.plot_gripper(grasp, gripper=gripper)
        mlab.show()

    # histogram
    i = 0
    for eps, force in zip(fc, frcl):
        if eps > 0 and force == 0:
            logging.info('Ferrari canny positive for out of fc at %d' %(i))
        elif eps == 0 and force == 1:
            logging.info('Ferrari canny zero for fc at %d' %(i))
        i += 1

    grasp_success_hist, grasp_success_bins = np.histogram(fc, bins=num_bins, range=(0,np.max(fc)))
    width = (grasp_success_bins[1] - grasp_success_bins[0])
    
    plt.figure()
    plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
    plt.title('Histogram', fontsize=font_size)
    plt.xlabel('Ferrari Canny', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.show()

def test_partial_closure_forces(dataset, config):
    # load grasps
    num_bins = 25
    font_size = 15
    obj_name = 'pipe_connector'
    stp_id = 'pose_6'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)

    # form gravity wrench
    grasp_force_limit = config['grasp_force_limit']
    object_mass = config['object_mass']
    stable_pose = dataset.stable_pose(obj.key, stp_id)
    gravity_magnitude = object_mass * GRAVITY_ACCEL
    stable_pose_normal = stable_pose.r[2]
    gravity_force = -gravity_magnitude * stable_pose_normal
    gravity_resist_wrench = -np.append(gravity_force, [0,0,0])

    # setup params
    params = {}
    params['force_limits'] = grasp_force_limit
    params['target_wrench'] = gravity_resist_wrench
    
    # visualize finger close on everythang
    vis_indices = [18]#58]
    pcs = []
    wrrs = []
    for i, grasp in enumerate(grasps[18:19]):
        # Compute quality
        logging.debug('Partial closure for soft fingers')
        pc = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'partial_closure', friction_coef=config['friction_coef'],
                                                       num_cone_faces=config['num_cone_faces'], soft_fingers=True,
                                                       params=params)
        logging.debug('Wrench resist ratio for soft fingers')
        wrr = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, 'wrench_resist_ratio', friction_coef=config['friction_coef'],
                                                        num_cone_faces=config['num_cone_faces'], soft_fingers=True,
                                                        params=params)

        pcs.append(pc)
        wrrs.append(wrr)

        logging.info('Grasp %d partial closure = %d, wrench resist ratio = %.4f' %(grasp.grasp_id, pc, wrr))

        # Visualize
        if grasp.grasp_id in vis_indices:
            contacts_found, contacts = grasp.close_fingers(obj, vis=True)
            if not contacts_found:
                logging.info('Contacts not found')

            contacts[0].plot_friction_cone(color='y')
            contacts[1].plot_friction_cone(color='c')
        
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

    # display top grasps
    gripper = gr.RobotGripper.load(gripper_name)
    grasps_and_qualities = zip(grasps, wrrs)
    grasps_and_qualities.sort(key = lambda x: x[1], reverse=True)
    for grasp, f in grasps_and_qualities[:5]:
        logging.info('Grasp %d quality %f' %(grasp.grasp_id, f))
        mlab.figure()
        mv.MayaviVisualizer.plot_mesh(obj.mesh)
        mv.MayaviVisualizer.plot_gripper(grasp, gripper=gripper)
        mlab.show()

    # histogram
    i = 0
    for pc, wrr in zip(pcs, wrrs):
        if wrr > 0 and pc == 0:
            logging.info('WRR positive for out of fc at %d' %(i))
        elif wrr == 0 and pc == 1:
            logging.info('WRR zero for fc at %d' %(i))
        i += 1

    grasp_success_hist, grasp_success_bins = np.histogram(wrrs, bins=num_bins, range=(0,np.max(wrrs)))
    width = (grasp_success_bins[1] - grasp_success_bins[0])
    
    plt.figure()
    plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
    plt.title('Histogram', fontsize=font_size)
    plt.xlabel('Wrench Resist Ratio', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.show()

def test_robust_partial_closure(dataset, config):
    # load grasps
    num_bins = 25
    font_size = 15
    obj_name = 'pipe_connector'
    stp_id = 'pose_6'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)

    # form gravity wrench
    grasp_force_limit = config['grasp_force_limit']
    object_mass = config['object_mass']
    stable_pose = dataset.stable_pose(obj.key, stp_id)
    gravity_magnitude = object_mass * GRAVITY_ACCEL
    stable_pose_normal = stable_pose.r[2]
    gravity_force = -gravity_magnitude * stable_pose_normal
    gravity_resist_wrench = -np.append(gravity_force, [0,0,0])

    stp_norm_line = np.array([np.zeros(3), 0.1*stable_pose_normal])

    # setup uncertainty config
    physical_u_config = config['physical_uncertainty']
    Sigma_g_t = np.diag([physical_u_config['sigma_gripper_x'], physical_u_config['sigma_gripper_y'], physical_u_config['sigma_gripper_z']])
    Sigma_g_r = physical_u_config['sigma_gripper_rot'] * np.eye(3)
    Sigma_o_t = np.diag([physical_u_config['sigma_obj_x'], physical_u_config['sigma_obj_y'], physical_u_config['sigma_obj_z']])
    Sigma_o_r = np.diag([physical_u_config['sigma_obj_rot_x'], physical_u_config['sigma_obj_rot_y'], physical_u_config['sigma_obj_rot_z']])
    Sigma_o_s = physical_u_config['sigma_scale']

    zeke_u_config = copy.deepcopy(config)
    zeke_u_config['sigma_rot_grasp'] = Sigma_g_r
    zeke_u_config['sigma_trans_grasp'] = Sigma_g_t
    zeke_u_config['sigma_rot_obj'] = Sigma_o_r
    zeke_u_config['sigma_trans_obj'] = Sigma_o_t
    zeke_u_config['sigma_scale_obj'] = Sigma_o_s

    R_obj_stp = stable_pose.r.T
    zeke_u_config['R_sample_sigma'] = R_obj_stp

    # setup params
    params = {}
    params['force_limits'] = grasp_force_limit
    params['target_wrench'] = gravity_resist_wrench
    
    # setup random variables
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, zeke_u_config)
    f_rv = scipy.stats.norm(zeke_u_config['friction_coef'], zeke_u_config['sigma_mu'])
    params_rv = rvs.ArtificialSingleRV(params)

    # visualize finger close on everythang
    vis_indices = []#58]
    ppcs = []
    for i, grasp in enumerate(grasps):
        # Setup grasp rv
        grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, zeke_u_config)

        # Compute quality
        logging.debug('Prob partial closure for soft fingers')
        ppc, vpc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, zeke_u_config, quality_metric='partial_closure',
                                                              params_rv=params_rv, num_samples=zeke_u_config['ppc_num_samples'], compute_variance=True)


        ppcs.append(ppc)
        logging.info('Grasp %d prob partial closure = %.3f, var partial closure = %.3f' %(grasp.grasp_id, ppc, vpc))

        # Visualize
        if grasp.grasp_id in vis_indices:
            contacts_found, contacts = grasp.close_fingers(obj, vis=True)
            if not contacts_found:
                logging.info('Contacts not found')

            contacts[0].plot_friction_cone(color='y')
            contacts[1].plot_friction_cone(color='c')
        
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

    # display top grasps
    gripper = gr.RobotGripper.load(gripper_name)
    grasps_and_qualities = zip(grasps, ppcs)
    grasps_and_qualities.sort(key = lambda x: x[1], reverse=True)
    for grasp, f in grasps_and_qualities[:5]:
        logging.info('Grasp %d quality %f' %(grasp.grasp_id, f))
        mlab.figure()
        mv.MayaviVisualizer.plot_mesh(obj.mesh)
        mv.MayaviVisualizer.plot_gripper(grasp, gripper=gripper)
        mlab.show()

    # histogram
    grasp_success_hist, grasp_success_bins = np.histogram(ppcs, bins=num_bins, range=(0,np.max(ppcs)))
    width = (grasp_success_bins[1] - grasp_success_bins[0])
    
    plt.figure()
    plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
    plt.title('Histogram', fontsize=font_size)
    plt.xlabel('Prob Partial Closure', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.show()

def test_robust_wrr(dataset, config):
    # load grasps
    num_bins = 25
    font_size = 15
    obj_name = 'pipe_connector'
    stp_id = 'pose_6'
    obj = dataset[obj_name]
    gripper_name = config['grippers'][0]
    grasps = dataset.grasps(obj.key, gripper=gripper_name)

    # form gravity wrench
    grasp_force_limit = config['grasp_force_limit']
    object_mass = config['object_mass']
    stable_pose = dataset.stable_pose(obj.key, stp_id)
    gravity_magnitude = object_mass * GRAVITY_ACCEL
    stable_pose_normal = stable_pose.r[2]
    gravity_force = -gravity_magnitude * stable_pose_normal
    gravity_resist_wrench = -np.append(gravity_force, [0,0,0])

    stp_norm_line = np.array([np.zeros(3), 0.1*stable_pose_normal])

    # setup uncertainty configs

    # zeke uncertainty
    physical_u_config = config['physical_uncertainty']
    Sigma_g_t = np.diag([physical_u_config['sigma_gripper_x'], physical_u_config['sigma_gripper_y'], physical_u_config['sigma_gripper_z']])
    Sigma_g_r = np.diag([physical_u_config['sigma_gripper_rot_x'], physical_u_config['sigma_gripper_rot_y'], physical_u_config['sigma_gripper_rot_z']])
    Sigma_o_t = np.diag([physical_u_config['sigma_obj_x'], physical_u_config['sigma_obj_y'], physical_u_config['sigma_obj_z']])
    Sigma_o_r = np.diag([physical_u_config['sigma_obj_rot_x'], physical_u_config['sigma_obj_rot_y'], physical_u_config['sigma_obj_rot_z']])
    Sigma_o_s = physical_u_config['sigma_scale']

    zeke_u_config = copy.deepcopy(config)
    zeke_u_config['sigma_rot_grasp'] = Sigma_g_r
    zeke_u_config['sigma_trans_grasp'] = Sigma_g_t
    zeke_u_config['sigma_rot_obj'] = Sigma_o_r
    zeke_u_config['sigma_trans_obj'] = Sigma_o_t
    zeke_u_config['sigma_scale_obj'] = Sigma_o_s

    R_obj_stp = stable_pose.r.T
    zeke_u_config['R_sample_sigma'] = R_obj_stp
    zeke_u_config['u_name'] = 'zeke'

    # isotropic uncertainty
    iso_u_config = copy.deepcopy(config)
    iso_u_config['u_name'] = 'isotropic'

    u_configs = [zeke_u_config, iso_u_config]

    # setup params
    params = {}
    params['force_limits'] = grasp_force_limit
    params['target_wrench'] = gravity_resist_wrench

    # visualize finger close on everythang
    vis_indices = []#58]
    ews = {}

    # compute metrics for each u config
    for u_config in u_configs:
        ews[u_config['u_name']] = []

        # setup random variables
        graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, u_config)
        f_rv = scipy.stats.norm(u_config['friction_coef'], u_config['sigma_mu'])
        params_rv = rvs.ArtificialSingleRV(params)

        for i, grasp in enumerate(grasps):
            # Setup grasp rv
            grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, u_config)

            # Compute quality
            logging.debug('Expected wrench resist ratio for soft fingers using %s uncertainty' %(u_config['u_name']))
            eq = rgq.RobustGraspQuality.expected_quality(graspable_rv, grasp_rv, f_rv, u_config, quality_metric='wrench_resist_ratio',
                                                         params_rv=params_rv, num_samples=u_config['eq_num_samples'])


            ews[u_config['u_name']].append(eq)
            logging.info('Grasp %d expected wrr = %.3f' %(grasp.grasp_id, eq))

            # Visualize
            if grasp.grasp_id in vis_indices:
                contacts_found, contacts = grasp.close_fingers(obj, vis=True)
                if not contacts_found:
                    logging.info('Contacts not found')

                contacts[0].plot_friction_cone(color='y')
                contacts[1].plot_friction_cone(color='c')
        
                ax = plt.gca()
                ax.set_xlim3d(0, obj.sdf.dims_[0])
                ax.set_ylim3d(0, obj.sdf.dims_[1])
                ax.set_zlim3d(0, obj.sdf.dims_[2])
                plt.show()

        # display top grasps
        gripper = gr.RobotGripper.load(gripper_name)
        grasps_and_qualities = zip(grasps, ews[u_config['u_name']])
        grasps_and_qualities.sort(key = lambda x: x[1], reverse=True)
        for grasp, f in grasps_and_qualities[:5]:
            logging.info('Grasp %d quality %f under u %s' %(grasp.grasp_id, f, u_config['u_name']))
            mlab.figure()
            mv.MayaviVisualizer.plot_mesh(obj.mesh)
            mv.MayaviVisualizer.plot_gripper(grasp, gripper=gripper)
            mlab.show()

        # histogram
        grasp_success_hist, grasp_success_bins = np.histogram(ews[u_config['u_name']], bins=num_bins, range=(0, np.max(ews[u_config['u_name']])))
        width = (grasp_success_bins[1] - grasp_success_bins[0])
    
        plt.figure()
        plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
        plt.title('Histogram', fontsize=font_size)
        plt.xlabel('Expected Wrench Resist Ratio under U %s' %(u_config['u_name']), fontsize=font_size)
        plt.ylabel('Num Grasps', fontsize=font_size)
        plt.show()

    IPython.embed()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    np.random.seed(100)
    random.seed(100)
    config_filename = 'cfg/test/test_quality.yaml'

    # read config file
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    # read some shit
    dataset = database.dataset('dexnet_physical_experiments')
    
    # run tests
    #test_ferrari_canny_L1(dataset, config)
    #test_partial_closure_forces(dataset, config)
    #test_robust_partial_closure(dataset, config)
    test_robust_wrr(dataset, config)
