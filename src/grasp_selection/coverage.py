"""
Script to compute the Ferrari-Canny metric for all pairs of contact points.
Author: Brian Hou
"""

from __future__ import print_function

import colorsys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import IPython

from mayavi import mlab

import os
import logging
import math
import pickle as pkl
import random
import sys
import time

import similarity_tf as stf
import tfx

import contacts
import database as db
import experiment_config as ec
import grasp as g
import graspable_object as go
import obj_file
import quality as q
import sdf_file

import stp_file as stp
import random

FRICTION_COEF = 0.5
MAX_WIDTH = 0.1

CONTACT_MODE_SDF = 0
CONTACT_MODE_VERTICES = 1
CONTACT_MODE_TRI_CENTERS = 2

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
        if success:
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

def grasp_from_contacts(c1, c2, width=MAX_WIDTH):
    """Constructs a ParallelJawPtGrasp3D object from two contact points.
    Default width is MAX_WIDTH, approx the PR2 gripper width.
    """
    grasp_center = 0.5 * (c1.point + c2.point)
    grasp_axis = c2.point - c1.point
    grasp_configuration = g.ParallelJawPtGrasp3D.configuration_from_params(
        grasp_center, grasp_axis, width=width)
    return g.ParallelJawPtGrasp3D(grasp_configuration)

def compute_qualities(graspable, indices, contacts,
                      metric='ferrari_canny_L1', friction_coef=0.5):
    n = len(contacts)
    grasps = []
    index_pairs = []
    qualities = []
    valid_contacts = []

    start_quality = time.time()
    for i in range(n):
        x = indices[i]
        c1 = contacts[i]
        start_quality_iter = time.time()
        logging.info('Computing qualities for index %d', i)

        for j in range(i, n):
            y = indices[j]
            c2 = contacts[j]

            grasp_valid = True
            if not in_force_closure(c1, c2, friction_coef):
                grasp_valid = False
                
            if grasp_valid:
                grasp = grasp_from_contacts(c1, c2)
                grasps.append(grasp)
                index_pairs.append((x,y))
                qualities.append(q.PointGraspMetrics3D.grasp_quality(grasp, graspable, metric, True, friction_coef))

                if c1 not in valid_contacts:
                    valid_contacts.append(c1)
                if c2 not in valid_contacts:
                    valid_contacts.append(c2)

        logging.info('Took %f seconds to compute qualities for index %d',
                     time.time() - start_quality_iter, i)
    logging.info('Took %f seconds to compute all qualities',
                 time.time() - start_quality)

    """
    all_points = np.array([c.point for c in valid_contacts])
    ax = plt.gca(projection = '3d')
    ax.scatter(all_points[:,0], all_points[:,1], all_points[:,2])
    plt.show()
    """

    return grasps, index_pairs, qualities

# Functions for plotting
def plot_mesh_with_qualities(graspable, contacts, fixed_index, valid_contacts,
                             mayavi=True, mayavi_rays=True,
                             meshlab=False, meshlab_path='{}.obj'):
    """Plots a mesh with vertices colored according to grasp quality. Fixes one
    contact and computes grasp quality with all other contacts. Only considers
    antipodal contacts that are in valid_contacts.

    Returns the computed qualities.

    graspable -- GraspableObject3D
    contacts -- list of Contact3D (that may slip!)
    fixed_index -- index of the contact to be fixed, used to index into contacts
    valid_contacts -- list of valid Contact3D objects (not in keep-out zone)

    mayavi -- boolean flag to visualize object in Mayavi
    mayavi_rays -- boolean flag to visualize grasp quality by plotting grasp axes

    meshlab -- boolean flag to export colored object into a Meshlab-compatible .obj file
    meshlab_path -- string to export .obj file to
    """
    qualities = np.zeros(len(contacts))

    c1 = contacts[fixed_index]
    start_quality = time.time()
    for i in range(len(contacts)):
        logging.debug('%d / %d grasps evaluated', i+1, len(contacts))
        c2 = contacts[i]

        # c2 is in the keep out zone
        if c2 not in valid_contacts:
            continue

        # c2 would slip; don't need to check antipodal or compute quality
        success, _, _ = c2.friction_cone()
        if not success: 
            continue

        # c1 and c2 aren't antipodal; don't need to compute quality
        if not is_antipodal(c1, c2):
            continue

        grasp = grasp_from_contacts(c1, c2)
        quality = q.PointGraspMetrics3D.grasp_quality(
            grasp, graspable, 'ferrari_canny_L1', soft_fingers=True,
            friction_coef=FRICTION_COEF)

        # the quality isn't positive; bad grasp
        if quality <= 0:
            continue

        # if it's a good grasp, save its quality
        qualities[i] = quality

    logging.info('Took %f seconds to compute qualities', time.time() - start_quality)

    if mayavi:
        obj_to_grid = graspable.sdf.transform_pt_obj_to_grid
        mlab.figure(bgcolor=(1, 1, 1)) # white background
        points = np.array([obj_to_grid(np.array(v)) for v in graspable.mesh.vertices()])
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        surf = mlab.triangular_mesh(x, y, z, graspable.mesh.triangles(), color=(0.5, 0.5, 0.5))
        if mayavi_rays:
            lines = []
            for quality, c2 in zip(qualities, contacts):
                if quality > 0:
                    lines.append((obj_to_grid(c1.point), obj_to_grid(c2.point)))

            for start, end in lines:
                direction = end - start
                direction = 2 * (direction / np.linalg.norm(direction))
                start = start - direction
                end = end + direction
                mlab.plot3d(*zip(start, end), color=(0, 0, 0), tube_radius=0.05)
        mlab.draw()
        mlab.show()
    elif meshlab:
        # write vertex colors to new obj file to manually visualize later
        colors = np.zeros((len(contacts), 3))
        max_quality = max(qualities)
        for i in range(len(qualities)):
            quality = qualities[i]
            if i == fixed_index: # cyan
                color = np.array([0, 1, 1])
            elif quality <= 0: # white
                color = np.array([1, 1, 1])
            else: # some shade of green
                color = np.array([0, quality / max_quality, 0])
            colors[i, :] = color # set to some RGB color

        with open(meshlab_path.format(fixed_index), 'w') as f:
            for i, (vertex, vertex_normal) in enumerate(zip(graspable.mesh.vertices(), graspable.mesh.normals())):
                # write normals
                f.write('vn {} {} {}\n'.format(*vertex_normal))

                # write vertex
                f.write('v {} {} {} {} {} {}\n'.format(vertex[0], vertex[1], vertex[2], colors[i, 0], colors[i, 1], colors[i, 2]))

            for a, b, c in graspable.mesh.triangles():
                f.write('f {0}//{0} {1}//{1} {2}//{2}\n'.format(a+1, b+1, c+1))
    return qualities

def plot_rays(graspable, grasps, qualities,
              max_rays=1000, max_width=float('inf'),
              table_vertices=None, table_tris=None, low=0.000, high=0.006):
    """
    Visualizes grasp quality by plotting colored grasp axes.

    graspable -- GraspableObject3D
    grasps -- list of pairs of ParallelPtGrasp3D objects
    qualities -- list of grasp qualities

    max_rays -- maximum number of rays to plot
    max_width -- maximum gripper width

    If table_vertices and table_tris are specified, plots a table surface for a
    stable pose.
    """
    if len(grasps) != len(qualities):
        raise ValueError('Must supply grasp and quality lists of same length')
    if len(qualities) == 0:
        logging.warning('No valid lines to plot')
        return

    # construct line objects
    lines = []
    line_qualities = []
    for grasp, quality in zip(grasps, qualities):
        g1, g2 = grasp.endpoints()
        if quality != 0 and np.linalg.norm(g1 - g2) <= max_width:
            lines.append((g1, g2))
            line_qualities.append(quality)
    if len(lines) == 0:
        logging.warning('No valid lines to plot')
        return

    if low is None or high is None:
        low, high = min(line_qualities), max(line_qualities)
    logging.info('Qualities are between %f and %f', low, high)

    # subsample rays
    logging.info('%d grasps, will sample below %d', len(lines), max_rays)
    indices = np.random.permutation(len(lines))[:min(len(lines), max_rays)]
    lines = np.array(lines)[indices]
    line_qualities = np.array(line_qualities)[indices]

    # plot rays
    f = lambda quality: 0.3 * (quality - low) / (high - low)
    for quality, line in zip(line_qualities, lines):
        points = np.c_[np.array(line[0]), np.array(line[1])].T
        if quality < 0:
            color = (1, 0, 0)
        else:
            # interpolate colors from red to green
            color = plt.get_cmap('hsv')(f(quality))[:-1]
        tube_radius = 0.002
        mlab.plot3d(points[:,0], points[:,1], points[:,2], color=color, tube_radius=tube_radius)

# Convenient helpers for useful plots
def vis_fixed(fixed_ind, **kwargs):
    qual_fixed = list(qualities)
    for pair in antipodal_pairs:
        index = ind_map[pair]
        if fixed_ind not in pair:
            qual_fixed[index] = 0
    plot_rays(graspable, contacts, qual_fixed, **kwargs)

def vis_top(n=500, **kwargs):
    cutoff = sorted(qualities, reverse=True)[n]
    top_quals = list(qualities)
    for i, q in enumerate(qualities):
        if q <= cutoff:
            top_quals[i] = 0
    plot_rays(graspable, contacts, top_quals, **kwargs)

def vis_stable(graspable, grasps, qualities,
               stp, vis_transform=True, eps=0.1, **kwargs):
    if len(grasps) != len(qualities):
        raise ValueError('Must supply grasp and quality lists of same length')
    n = len(grasps)

    # extract stable pose info
    tf = stf.SimilarityTransform3D(pose=tfx.pose(stp.r))
    n = stp.r[2, :]
    x0 = stp.x0
    tfinv = tf.inverse()

    # render mesh in stable pose
    logging.info('Transforming mesh')
    m = graspable.mesh.transform(tf)
    mn, mx = m.bounding_box()
    d = max(mx[1] - mn[1], mx[0] - mn[0]) / 2 + eps
    z = mn[2]
    table_vertices = np.array([[d, d, z],
                               [d, -d, z],
                               [-d, d, z],
                               [-d, -d, z]])
    table_vertices = np.array([tfinv.apply(e) for e in table_vertices])
    table_tris = np.array([[0, 1, 2], [1, 2, 3]])
    if vis_transform:
        mlab.figure()
        graspable.mesh.visualize()
        mlab.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2],
                             table_tris, representation='surface', color=(0,0,0), opacity=0.5)

    # prune grasps in collision with the surface
    logging.info('Pruning grasps')
    valid_grasps = []
    valid_qualities = []
    for grasp, quality in zip(grasps, qualities):
        g1, g2 = grasp.endpoints()
        if n.dot(g2 - x0) > 0 and n.dot(g1 - x0) > 0:
            valid_grasps.append(grasp)
            valid_qualities.append(quality)

    # plot rays
    logging.info('Plotting rays')
    plot_rays(graspable, valid_grasps, valid_qualities, **kwargs)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    dataset = database[config['datasets'].keys()[0]]

    mode = CONTACT_MODE_TRI_CENTERS

    for graspable in dataset:
        obj_name = graspable.key

        # stable poses
        stable_pose_filename = os.path.join(config['out_dir'], '{}_stable_poses.stp'.format(obj_name))
        stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_filename, min_prob=0.01)
        stable_poses = stp.StablePoseFile().read(stable_pose_filename)

        # antipodal pairs
        all_contacts, all_vertices, all_normals = load_vertex_contacts(graspable, mode=mode)
        valid_indices, valid_contacts = filter_contacts(all_contacts, max_vertices=config['max_vertices'])  # for all grasps
        grasps, index_pairs, qualities = compute_qualities(graspable, valid_indices, valid_contacts)

        # Save expensive computation
        grasps_filename = os.path.join(config['out_dir'], '{}_grasps.pkl'.format(obj_name))
        indices_filename = os.path.join(config['out_dir'], '{}_indices.npy'.format(obj_name))
        quality_filename = os.path.join(config['out_dir'], '{}_qualities.npy'.format(obj_name))
        f = open(grasps_filename, 'w')
        pkl.dump(grasps, f)
        np.save(indices_filename, np.array(index_pairs))
        np.save(quality_filename, np.array(qualities))

        # organize grasps by segment
        if config['use_segments']:
            seg_filename = os.path.join(config['shape_data_dir'], config['datasets'].keys()[0], '%s_0.seg' %(obj_name))

            # load segments
            f = open(seg_filename, 'r')
            triangle_labels = []
            for line in f:
                triangle_labels.append(int(line))
            triangle_labels = np.array(triangle_labels[:-1])

            # organize grasps by segment
            vertex_labels = [[] for i in range(len(all_vertices))]

            for i, tri in enumerate(graspable.mesh.triangles()):
                if mode == CONTACT_MODE_TRI_CENTERS:
                    vertex_labels[i].append(triangle_labels[i])
                else:
                    vertex_labels[tri[0]].append(triangle_labels[i])
                    vertex_labels[tri[1]].append(triangle_labels[i])
                    vertex_labels[tri[2]].append(triangle_labels[i])
                
            grasp_labels = []
            for grasp, index_pair in zip(grasps, index_pairs):
                labels_1 = set(vertex_labels[index_pair[0]])
                labels_2 = set(vertex_labels[index_pair[1]])

                g_labels = []
                for l1 in labels_1:
                    if l1 in labels_2:
                        g_labels.append(l1)
                grasp_labels.append(set(g_labels))

            # plot joint histogram for segments
            plt.figure(12)
            qualities_arr = np.array(qualities)
            hist, bins, _ = plt.hist(qualities_arr[qualities_arr > 0], bins=config['hist_num_bins'], range=(0.0, config['max_tau']))
            plt.title('Object %s Quality' %(obj_name), fontsize=15)
            plt.xlabel('Quality', fontsize=15)
            plt.ylabel('Count', fontsize=15)
            figname = 'obj_%s_hist.pdf' %(obj_name)
            plt.savefig(os.path.join(config['out_dir'], figname), dpi=400)

            hist = hist / np.sum(hist)
            filename = 'obj_%s_hist.npy' %(obj_name)
            np.save(os.path.join(config['out_dir'], filename), hist)
            filename = 'obj_%s_hist_bins.npy' %(obj_name)
            np.save(os.path.join(config['out_dir'], filename), bins)

            # go through the human labelled segments
            color_delta = 1.0 / (np.max(triangle_labels) + 1)
            segment_labels = []
            for label in set(triangle_labels):
                grasps_label = []
                qualities_label = []
                color = colorsys.hsv_to_rgb(label*color_delta, 0.9, 0.9)
                segment_labels.append('Label %d' %(label))

                for grasp, quality, grasp_ls in zip(grasps, qualities, grasp_labels):
                    if label in grasp_ls:
                        grasps_label.append(grasp)
                        qualities_label.append(quality)
                        
                # plot joint histogram for segments
                plt.figure(10)
                qualities_label = np.array(qualities_label)
                plt.hist(qualities_label[qualities_label > 0], bins=config['hist_num_bins'], color=color, range=(0.0, config['max_tau']))

                # plot single histogram of quality
                plt.figure(11)
                plt.clf()
                hist, bins, _ = plt.hist(qualities_label[qualities_label > 0], bins=config['hist_num_bins'], range=(0.0, config['max_tau']))
                plt.title('Label %d Quality' %(label), fontsize=15)
                plt.xlabel('Quality', fontsize=15)
                plt.ylabel('Count', fontsize=15)
                figname = 'obj_%s_label_%d_hist.pdf' %(obj_name, label)
                plt.savefig(os.path.join(config['out_dir'], figname), dpi=400)

                # normalize hist for saving
                hist = hist / np.sum(hist)
                filename = 'obj_%s_label_%d_hist.npy' %(obj_name, label)
                np.save(os.path.join(config['out_dir'], filename), hist)
                filename = 'obj_%s_label_%d_hist_bins.npy' %(obj_name, label)
                np.save(os.path.join(config['out_dir'], filename), bins)

                # plot segmented region
                one_vs_all_labels = np.zeros(triangle_labels.shape)
                one_vs_all_labels[triangle_labels == label] = 0 
                one_vs_all_labels[triangle_labels != label] = 1
                mlab.clf()
                graspable.mesh.visualize_segments(one_vs_all_labels.astype(np.int16))
                figname = 'obj_%s_label_%d_mesh.png' %(obj_name, label)                
                mlab.savefig(os.path.join(config['out_dir'], figname))

                # plot grasps on segment
                mlab.clf()
                most_stable_pose = stable_poses[-1]
                logging.info('About to plot rays with most stable pose.')
                graspable.mesh.visualize_segments(one_vs_all_labels.astype(np.int16))
                plot_rays(graspable, grasps_label, qualities_label, max_rays=500, low=0.0, high=config['max_tau'])
                figname = 'obj_%s_label_%d_grasps.png' %(obj_name, label)                
                mlab.savefig(os.path.join(config['out_dir'], figname))

            # histogram of all grasp qualities
            plt.figure(10)
            plt.title('All Label Qualities', fontsize=15)
            plt.xlabel('Quality', fontsize=15)
            plt.ylabel('Count', fontsize=15)
            plt.legend(segment_labels, loc='best')
            figname = 'obj_%s_all_hist.pdf' %(obj_name)                
            plt.savefig(os.path.join(config['out_dir'], figname), dpi=400)

            # filter grasps by quality threshold
            max_tau = config['max_tau']
            tau_res = config['tau_res']
            n_tau = int(math.ceil(max_tau / tau_res))
            tau_vals = [j * tau_res for j in range(n_tau)]
            for tau in tau_vals:
                vertex_labels = [0 for j in range(len(all_vertices))]
                triangle_labels = [0 for j in range(len(graspable.mesh.triangles()))]
                for quality, index_pair in zip(qualities, index_pairs):
                    if quality > tau:
                        vertex_labels[index_pair[0]] = 1
                        vertex_labels[index_pair[1]] = 1
                for j, tri in enumerate(graspable.mesh.triangles()):
                    if mode == CONTACT_MODE_TRI_CENTERS and vertex_labels[j] == 1:
                        triangle_labels[j] = 1
                    elif vertex_labels[tri[0]] == 1 or vertex_labels[tri[1]] == 1 or vertex_labels[tri[2]] == 1:
                        triangle_labels[j] = 1
                        
                # visualize
                logging.info('Segmentation with tau = %f' %(tau))
                mlab.clf()
                graspable.mesh.visualize_segments(np.array(triangle_labels))
                figname = 'obj_%s_tau_%f_mesh.png' %(obj_name, tau)                
                mlab.savefig(os.path.join(config['out_dir'], figname))

        #IPython.embed()

        # sort by quality
        """
        grasps_and_qualities = zip(grasps, qualities)
        grasps_and_qualities.sort(key = lambda x: x[1], reverse = True)
        grasps = [g[0] for g in grasps_and_qualities]
        qualities = [g[1] for g in grasps_and_qualities]

        # ray visualization
        most_stable_pose = stable_poses[-1]
        logging.info('About to plot rays with most stable pose.')
        vis_stable(graspable, grasps[:100], qualities[:100],
                   most_stable_pose, vis_transform=True, max_rays=500)
        mlab.draw()
        mlab.show()

        IPython.embed()
        exit(0)
        """
