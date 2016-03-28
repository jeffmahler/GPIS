"""
Script to visualize the computed coverage results
Author: Jeff Mahler
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython

from mayavi import mlab

import os
import json
import logging
import pickle as pkl
import random
import shutil
import sys
import time

import similarity_tf as stf
import tfx

import contacts
import privacy_coverage as coverage
import database as db
import experiment_config as ec
import gripper as gr
import grasp as g
import graspable_object as go
import mayavi_visualizer as mv
import obj_file
import quality as q
import sdf_file
import similarity_tf as stf
import stp_file as stp
import random

masked_object_tags = ['_no_mask', '_masked_bbox', '_masked_hull']

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    # load the models
    data_dir = config['data_dir']
    object_keys = coverage.read_index(data_dir)
    objects = []
    masks = []
    masked_objects = []

    # setup experiment output dir
    experiment_id = gen_experiment_id()
    output_dir = config['out_dir']
    experiment_dir = os.path.join(output_dir, 'vis-%s' %(experiment_id))
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    _, config_root = os.path.split(config_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_root))
    logging.info('Saving visuals to %s' %(experiment_dir))

    # get the mode
    gripper = gr.RobotGripper.load(config['gripper'])
    mode = config['contact_point_mode']
    logging.info('Using contact point mode %d' %(mode))

    # read plotting params
    dpi = config['dpi']
    font_size = config['font_size']
    line_width = config['line_width']
    num_views = config['num_views']
    dist = config['cam_distance']
    delta_view = 360.0 / num_views
    elevs = [55.0, 30.0]
    mlab.figure(bgcolor=(1,1,1), size=(1000,1000))

    # load all of the objects
    for object_key in object_keys:
        logging.info('Loading object %s' %(object_key))
        
        # first load the clean mesh
        subdiv = False
        mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+db.OBJ_EXT)).read()
        mesh, tri_ind_mapping = mesh.subdivide(min_triangle_length=config['min_tri_length'])

        sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+db.SDF_EXT)).read()
        objects.append(go.GraspableObject3D(sdf, mesh, key=object_key))
        objects[-1].model_name_ = os.path.join(data_dir, object_key+db.OBJ_EXT)
        masks.append(np.load(os.path.join(data_dir, object_key+'_mask.npy')))

        new_mask = []
        mask = masks[-1]
        for index in mask.tolist():
            for new_index in tri_ind_mapping[index]:
                new_mask.append(new_index)
        masks[-1] = np.array(new_mask)

        masked_objects.append([])

        # then load the masked versions
        for mask_tag in masked_object_tags:
            mesh = obj_file.ObjFile(os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)).read()
            mesh, tri_ind_mapping = mesh.subdivide(min_triangle_length=config['min_tri_length'])
            sdf = sdf_file.SdfFile(os.path.join(data_dir, object_key+mask_tag+db.SDF_EXT)).read()
            masked_objects[-1].append(go.GraspableObject3D(sdf, mesh, key=object_key+mask_tag))
            masked_objects[-1][-1].model_name_ = os.path.join(data_dir, object_key+mask_tag+db.OBJ_EXT)

    for graspable, mask in zip(objects, masks):
        obj_name = graspable.key
        logging.info('Analyzing coverage for object %s' %(obj_name))

        # compute stable poses (for later down the pipeline, maybe we should remove this...)
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=config['min_prob'])
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)

        # break into private and public areas
        public_mask = np.setdiff1d(np.arange(len(graspable.mesh.triangles())), mask)
        public_mesh = graspable.mesh.mask(np.array(public_mask))
        private_mesh = graspable.mesh.mask(mask)
        private_mesh_cvh = private_mesh.convex_hull()
        private_mesh_cvh.rescale(1.025, center=True) # rescaling for weird triangle flicker (which could be fixed with fancy zippering later)
        private_mesh_bb = private_mesh.bounding_box_mesh()

        # plot raw mesh
        for i, stable_pose in enumerate(stable_poses):
            stable_pose.id = 'pose_%d' %(i)

            # table version
            mlab.clf()
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(graspable.mesh, stable_pose, d=0.15)
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'table_mesh_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            # solo mesh version
            mlab.clf()
            T_mesh_world = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(4)))
            T_mesh_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r))
            mesh_tf = graspable.mesh.transform(T_mesh_stp)
            mv.MayaviVisualizer.plot_mesh(mesh_tf, T_mesh_world, color=(0.6, 0.6, 0.6))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'solo_mesh_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            # mesh privacy
            public_mesh_tf = public_mesh.transform(T_mesh_stp)
            private_mesh_tf = private_mesh.transform(T_mesh_stp)
            private_mesh_cvh_tf = private_mesh_cvh.transform(T_mesh_stp)
            private_mesh_bb_tf = private_mesh_bb.transform(T_mesh_stp)

            mlab.clf()
            mv.MayaviVisualizer.plot_mesh(public_mesh_tf, T_mesh_world, color=(0.6, 0.6, 0.6))
            mv.MayaviVisualizer.plot_mesh(private_mesh_tf, T_mesh_world, color=(0, 0, 1))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            mlab.clf()
            mv.MayaviVisualizer.plot_mesh(public_mesh_tf, T_mesh_world, color=(0.6, 0.6, 0.6))
            mv.MayaviVisualizer.plot_mesh(private_mesh_cvh_tf, T_mesh_world, color=(0, 0, 1))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_cvh_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            mlab.clf()
            mv.MayaviVisualizer.plot_mesh(public_mesh_tf, T_mesh_world, color=(0.6, 0.6, 0.6))
            mv.MayaviVisualizer.plot_mesh(private_mesh_bb_tf, T_mesh_world, color=(0, 0, 1))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_bb_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            mlab.clf()
            mv.MayaviVisualizer.plot_mesh(public_mesh_tf, T_mesh_world, color=(0.6, 0.6, 0.6))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_deleted_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            # mesh privacy with table
            mlab.clf()
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(public_mesh, stable_pose, d=0.15, color=(0.6, 0.6, 0.6))
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(private_mesh, stable_pose, d=0.15, color=(0, 0, 1))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_table_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

            mlab.clf()
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(public_mesh, stable_pose, d=0.15, color=(0.6, 0.6, 0.6))
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(private_mesh_cvh, stable_pose, d=0.15, color=(0, 0, 1))
            for i, elev in enumerate(elevs):
                for j in range(num_views):
                    az = j * delta_view
                    mlab.view(azimuth=az, elevation=elev, distance=dist)
                    figname = 'private_mesh_cvh_table_%s_%s_view_%d_%d.png' %(obj_name, stable_pose.id, i, j)                
                    mlab.savefig(os.path.join(experiment_dir, figname))

