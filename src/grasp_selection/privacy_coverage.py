"""
Script to compute the quality metrics for all pairs of contact points whilst maintaining privacy
Author: Brian Hou
"""

from __future__ import print_function

import copy
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython
import sklearn.neighbors as skn

from mayavi import mlab

import itertools
import os
import json
import logging
import math
import mayavi.mlab as mlab
import pickle as pkl
import random
import scipy
import sys
import time

import similarity_tf as stf
import tfx

import contacts
import database as db
import experiment_config as ec
import gripper as gr
import grasp as g
import graspable_object as go
import grasp_collision_checker as gcc
import mayavi_visualizer as mv
import obj_file
import quality as q
import random_variables as rvs
import robust_grasp_quality as rgq
import sdf_file

import stp_file as stp
import random

FRICTION_COEF = 0.5
MAX_WIDTH = 0.1

CONTACT_MODE_SDF = 0
CONTACT_MODE_VERTICES = 1
CONTACT_MODE_TRI_CENTERS = 2

masked_object_tags = ['_no_mask']#, '_masked_bbox', '_masked_hull']

def read_index(data_dir):
    """ Read in keys from index file """
    index_file = os.path.join(data_dir, db.INDEX_FILE)
    data_keys = []
    index_file_lines = open(index_file, 'r').readlines()

    for i, line in enumerate(index_file_lines):
        tokens = line.split()
        if not tokens: # ignore empty lines
            continue
        
        data_keys.append(tokens[0])

    return data_keys

def load_vertex_contacts(graspable, mode=CONTACT_MODE_TRI_CENTERS, vis=False):
    """Create contact points from the mesh vertices and normals.

    graspable -- GraspableObject3D
    vis       -- whether to draw the sdf and mesh (and the planes to determine if points are on the handle)
    """
    if vis:
        graspable.plot_sdf_vs_mesh()

    start_loading_contacts = time.time()

    vertex_contacts = []
    on_surface_count = 0

    # load vertices
    normals = None
    if mode == CONTACT_MODE_SDF:
        vertices, vals = graspable.sdf.surface_points(grid_basis=False)
    elif mode == CONTACT_MODE_VERTICES:
        vertices = graspable.mesh.vertices()
        normals = graspable.mesh.normals()
    elif mode == CONTACT_MODE_TRI_CENTERS:
        vertices = graspable.mesh.tri_centers()
        normals = graspable.mesh.tri_normals()

    for i, vertex in enumerate(vertices):
        if mode == CONTACT_MODE_SDF or normals is None:
            contact = contacts.Contact3D(graspable, np.array(vertex))
        else:
            normal = np.array(normals[i]) # outward facing normal
            contact = contacts.Contact3D(graspable, np.array(vertex), -normal)
            contact.normal = normal
        contact.friction_cone() # friction cone is cached for later
        vertex_contacts.append(contact)

    """
    all_points = np.array([c.point for c in vertex_contacts])
    ax = plt.gca(projection = '3d')
    ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2])
    plt.show()
    """

    # loading ~4100 contacts and computing their friction cones takes ~20 seconds
    logging.info('Loading contacts took %f seconds', time.time() - start_loading_contacts)
    return vertex_contacts, vertices, normals

def filter_contacts(vertex_contacts, friction_coef=FRICTION_COEF, max_vertices=None):
    """Return the contacts that won't slip."""
    start_filtering_contacts = time.time()
    valid_indices = []
    valid_contacts = []
    for i, contact in enumerate(vertex_contacts):
        success, _, _ = contact.friction_cone(friction_coef=friction_coef)
        if success and contact.normal is not None and not np.isinf(np.sum(contact.normal)) \
                and not np.isnan(np.sum(contact.normal)):
            valid_indices.append(i) # for indexing into vertices
            valid_contacts.append(contact)
    logging.info('Filtering contacts took %f seconds', time.time() - start_filtering_contacts)

    if max_vertices is not None and len(valid_contacts) > max_vertices:
        indices_and_contacts = zip(valid_indices, valid_contacts)
        random.shuffle(indices_and_contacts)
        valid_indices = [c[0] for c in indices_and_contacts]
        valid_contacts = [c[1] for c in indices_and_contacts]
        valid_indices = valid_indices[:max_vertices]
        valid_contacts = valid_contacts[:max_vertices]

    return valid_indices, valid_contacts

def in_force_closure(c1, c2, friction_coef=FRICTION_COEF):
    """Returns True if c1 and c2 are antipodal."""
    if np.linalg.norm(c1.point - c2.point) > MAX_WIDTH: # max width
        return False
    return bool(q.PointGraspMetrics3D.force_closure(c1, c2, friction_coef))

def grasp_from_contacts(c1, c2, gripper):
    """Constructs a ParallelJawPtGrasp3D object from two contact points.
    Default width is MAX_WIDTH, approx the PR2 gripper width.
    """
    grasp_center = 0.5 * (c1.point + c2.point)
    grasp_axis = c2.point - c1.point
    grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
    grasp_configuration = g.ParallelJawPtGrasp3D.configuration_from_params(
        grasp_center, grasp_axis, width=gripper.max_width)
    return g.ParallelJawPtGrasp3D(grasp_configuration)

def randos(n):
    while True:
        yield np.random.randint(n)
    yield -100

def cdist(x, y):
    return np.arccos(np.abs(x.dot(y)) / (np.linalg.norm(x) * np.linalg.norm(y)))

def compute_qualities(graspable, indices, contacts, config, deterministic_metrics=['ferrari_canny_L1'],
                      robust_metrics=None, num_samples=10000, num_theta=16, mask=[]):
    """ Compute qualities for a graspable using the specified metric """
    # setup buffers
    n = len(contacts)
    friction_coef = config['friction_coef']
    grasps = []
    index_pairs = []
    qualities = []
    valid_contacts = []

    # setup random variables
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(graspable, config)
    f_rv = scipy.stats.norm(friction_coef, config['sigma_mu'])
    gripper = gr.RobotGripper.load(config['gripper'])
    #collision_checker = gcc.OpenRaveGraspChecker(gripper, view=False)
    #collision_checker.set_object(graspable)

    # start computing qualities
    start_quality = time.time()
    logging.info('%d possible grasps', n * (n+1) / 2)
  
    np.random.seed(0)
    c_ind = np.arange(n).tolist()
    random.shuffle(c_ind)

    # setup nearest neighbors
    alpha = 2 * np.arctan(friction_coef)
    dist_thresh = 1.0 - np.cos(alpha)
    surface_normals = np.array([c.normal for c in contacts])
    contact_points = np.array([c.point for c in contacts])
    normal_nbrs = skn.NearestNeighbors(algorithm='ball_tree', metric='pyfunc', func=cdist)
    normal_nbrs.fit(surface_normals)
    contact_nbrs = skn.NearestNeighbors(algorithm='ball_tree', metric='euclidean')
    contact_nbrs.fit(contact_points)

    cnt = 0
    seen = set()
    for i in c_ind:
        # lookup potential opposite facing vertices
        c1 = contacts[i]
        contact_ind = contact_nbrs.radius_neighbors(c1.point.reshape(1,3), gripper.max_width, return_distance=False)[0]
        normal_ind = normal_nbrs.radius_neighbors(c1.normal.reshape(1,3), alpha, return_distance=False)[0]
        ind = np.intersect1d(contact_ind, normal_ind).tolist()
        random.shuffle(ind)

        num_contact_grasps = 0
        
        for j in ind:
            # get valid index and contact pairs
            if cnt % config['out_rate'] == 0:
                logging.info('Computing qualities for pair %d: (%d, %d)', cnt, i, j)
                logging.info('Num grasps: %d' %(len(grasps)))
            x, y = indices[i], indices[j]
            c1, c2 = contacts[i], contacts[j]

            # exit if one contact is in the mask
            if x in mask or y in mask:
                continue

            # only save force closure grasps
            fc = q.PointGraspMetrics3D.force_closure(c1, c2, friction_coef)
            if fc and (i, j) not in seen and (j, i) not in seen:
                seen.add((i, j))
                seen.add((j, i))
                if c1 not in valid_contacts:
                    valid_contacts.append(c1)
                if c2 not in valid_contacts:
                    valid_contacts.append(c2)
            
                # create grasp
                grasp = grasp_from_contacts(c1, c2, gripper)

                # add to set
                qualities.append({})
                grasps.append(grasp)
                index_pairs.append((x,y))

                # compute deterministic metrics
                if deterministic_metrics is not None:
                    for metric in deterministic_metrics:
                        quality = q.PointGraspMetrics3D.grasp_quality(grasp, graspable, metric, True, friction_coef)
                        qualities[-1][metric] = quality

                # compute robust metrics
                if robust_metrics is not None:
                    grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, config)
                    for metric in robust_metrics:
                        metric_tag = 'robust_'+metric
                        quality = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric=metric,
                                                                             num_samples=config['num_robust_samples'])
                        qualities[-1][metric_tag] = quality

                # break if we've found enough grasps for this contact
                num_contact_grasps = num_contact_grasps + 1
                if num_contact_grasps >= config['grasps_per_contact']:
                    break

            cnt += 1

        # break if max number reached
        if len(grasps) > num_samples or cnt > n**2:
            break

    logging.info('Took %f seconds to compute all qualities',
                 time.time() - start_quality)
    IPython.embed()
    return grasps, index_pairs, qualities

# Functions for plotting
def vis_coverage(public_mesh, private_meshes, grasps, qualities, stp, config, mask=None, gripper=None, color_gripper=True,
                 color_grasps=True, plot_table=True, max_display=None, rank=False):
    if len(grasps) != len(qualities):
        raise ValueError('Must supply grasp and quality lists of same length')

    if max_display is None:
        max_display = len(grasps)

    # take random subset of size |max_display|
    grasps_and_qualities = zip(grasps, qualities)
    if not rank:
        random.shuffle(grasps_and_qualities)
    else:
        grasps_and_qualities.sort(key=lambda x: x[1], reverse=True)

    grasps_and_qualities = grasps_and_qualities[:max_display]
    grasps = [gq[0] for gq in grasps_and_qualities]
    qualities = [gq[1] for gq in grasps_and_qualities]

    # plot mesh
    if stp is not None:
        for private_mesh in private_meshes:
            T_obj_world = mv.MayaviVisualizer.plot_stable_pose(private_mesh, stp, d=config['table_extent'], color=(0,0,1), plot_table=plot_table)
        T_obj_world = mv.MayaviVisualizer.plot_stable_pose(public_mesh, stp, d=config['table_extent'], color=(0.6,0.6,0.6), plot_table=plot_table)
            
    # plot grasps colored by quality
    low = config['min_q']
    high = config['max_q']
    q_to_c = lambda quality: config['quality_scale'] * (quality - low) / (high - low)
    for grasp, quality in zip(grasps, qualities):
        color = plt.get_cmap('hsv')(q_to_c(quality))[:-1]
        if gripper is None:
            if color_grasps:
                mv.MayaviVisualizer.plot_grasp(grasp, T_obj_world, tube_radius=config['tube_radius'], grasp_axis_color=color,
                                               endpoint_color=color, endpoint_scale=config['endpoint_scale'])
            else:
                mv.MayaviVisualizer.plot_grasp(grasp, T_obj_world, tube_radius=config['tube_radius'], grasp_axis_color=(0,1,0),
                                               endpoint_color=(0,1,0), endpoint_scale=config['endpoint_scale'])                
        else:
            if color_gripper:
                mv.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper, color=color)
            else:
                mv.MayaviVisualizer.plot_gripper(grasp, T_obj_world, gripper, color=(0.4, 0.4, 0.4))

def prune_grasps_in_collision(graspable, grasps, qualities, gripper, stp, config):
    if len(grasps) != len(qualities):
        raise ValueError('Must supply grasp and quality lists of same length')

    # setup object collision checker
    collision_checker = gcc.OpenRaveGraspChecker(gripper, view=False)
    collision_checker.set_object(graspable)

    # prune grasps in collision with the surface
    logging.info('Pruning %d grasps', len(grasps))
    i = 0
    coll_free_grasps = []
    coll_free_qualities = []
    for grasp, quality in zip(grasps, qualities):
        if i % config['out_rate'] == 0:
            logging.info('Checking collisions for grasp %d' %(i))
        aligned_grasp = grasp.grasp_aligned_with_table(stp)
        if not collision_checker.in_collision(aligned_grasp) and not gripper.collides_with_table(aligned_grasp, stp):
            coll_free_grasps.append(aligned_grasp)
            coll_free_qualities.append(quality)
        i = i+1
    return coll_free_grasps, coll_free_qualities

if __name__ == '__main__':
    np.random.seed(106)
    random.seed(106)

    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)

    # load the models
    data_dir = config['data_dir']
    object_keys = read_index(data_dir)
    objects = []
    masks = []
    masked_objects = []

    # get the mode
    mode = config['contact_point_mode']
    logging.info('Using contact point mode %d' %(mode))

    # load all of the objects
    for object_key in object_keys:
        logging.info('Loading object %s' %(object_key))
        
        # first load the clean mesh
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
            
    # loop through the objects and compute coverage for each
    for graspable, masked_graspables, mask in zip(objects, masked_objects, masks):
        obj_name = graspable.key
        logging.info('Computing coverage for object %s' %(obj_name))

        # compute stable poses (for later down the pipeline, maybe we should remove this...)
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=config['min_prob'])
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)

        """
        T_obj_world = stf.SimilarityTransform3D(from_frame='world', to_frame='obj')
        mv.MayaviVisualizer.plot_mesh(graspable.mesh, T_obj_world)
        mlab.axes()
        mlab.show()

        for stable_pose in stable_poses:
            logging.info('Displaying stable pose with p = %.3f' %(stable_pose.p))
            T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')
            mv.MayaviVisualizer.plot_stable_pose(graspable.mesh, stable_pose, T_table_world, d=0.15)
            mlab.axes()
            mlab.show()
        """

        privacy_metric = graspable.mesh.surface_area(mask) / graspable.mesh.surface_area()
        logging.info('Privacy metric: %f' %(privacy_metric))

        # antipodal pairs
        logging.info('Computing grasps')
        all_contacts, all_vertices, all_normals = load_vertex_contacts(graspable, mode=mode)
        valid_indices, valid_contacts = filter_contacts(all_contacts, max_vertices=config['max_vertices'])  # for all grasps

        """
        T_obj_world = stf.SimilarityTransform3D(from_frame='world', to_frame='obj')
        m = graspable.mesh
        #m = m.mask(np.array(valid_indices[2000:]))
        mv.MayaviVisualizer.plot_mesh(m, T_obj_world, style='dexnet')
        mlab.view(azimuth=180)
        for index, contact in zip(valid_indices[2000:2001], valid_contacts[2000:2001]):
            logging.info('Index %d', index)
            l = np.array([contact.point, contact.point + 0.01*contact.normal])
            mlab.points3d(contact.point[0], contact.point[1], contact.point[2], scale_factor=0.01, color=(0,0,1))
            mlab.plot3d(l[:,0], l[:,1], l[:,2], tube_radius=0.002, color=(0,1,0))
        #mlab.show()
        """

        logging.info('Found %d candidate contacts' %(len(valid_contacts)))
        logging.info('Computing qualities')
        grasps, index_pairs, qualities = compute_qualities(graspable, valid_indices, valid_contacts, config,
                                                           deterministic_metrics=config['quality_metrics'],
                                                           robust_metrics=config['robust_quality_metrics'],
                                                           num_samples=config['num_grasps'])
        num_unmasked_grasps = len(grasps)
        mlab.show()

        """
        gripper = gr.RobotGripper.load(config['gripper'])
        qualities = [q['robust_force_closure'] for q in qualities]
        vis_coverage(graspable, grasps, qualities, gripper, None, config)
        mlab.show()
        """

        # save expensive computation
        logging.info('Saving data')
        grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
        indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
        quality_filename = os.path.join(config['out_dir'], '{}_qualities.pkl'.format(obj_name))
        f = open(grasps_filename, 'w')
        pkl.dump(grasps, f)
        f.close()
        np.save(indices_filename, np.array(index_pairs))
        f = open(quality_filename, 'w')
        pkl.dump(qualities, f)
        f.close()

        # loop through all of the masked versions
        # TODO: re-enable, but for the sake of analyses it's basically irrelevant
        """
        for masked_graspable in masked_graspables:
            obj_name = masked_graspable.key
            logging.info('Computing coverage for object %s' %(obj_name))

            # antipodal pairs for this object
            logging.info('Computing grasps')
            all_contacts, all_vertices, all_normals = load_vertex_contacts(masked_graspable, mode=mode)
            valid_indices, valid_contacts = filter_contacts(all_contacts, max_vertices=config['max_vertices'])  # for all grasps
            logging.info('Computing qualities')
            num_grasps = num_unmasked_grasps * (1.0 - privacy_metric)
            grasps, index_pairs, qualities = compute_qualities(masked_graspable, valid_indices, valid_contacts, config,
                                                               deterministic_metrics=config['quality_metrics'],
                                                               robust_metrics=config['robust_quality_metrics'],
                                                               num_samples=num_grasps)

            # save expensive computation
            logging.info('Saving data') 
            grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
            indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
            quality_filename = os.path.join(config['out_dir'], '{}_qualities.json'.format(obj_name))
            f = open(grasps_filename, 'w')
            pkl.dump(grasps, f)
            np.save(indices_filename, np.array(index_pairs))
            f = open(quality_filename, 'w')
            json.dump(qualities, f)

         """
