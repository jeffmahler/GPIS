"""
Script to compute the Ferrari-Canny metric for all pairs of contact points.
Author: Brian Hou
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import IPython

from mayavi import mlab

import os
import logging
import random
import time

import similarity_tf as stf
import tfx

import contacts
import database as db
import grasp as g
import graspable_object as go
import obj_file
import quality as q
import sdf_file

import stp_file as stp

FRICTION_COEF = 0.5
MAX_WIDTH = 0.1

def load_vertex_contacts(graspable, vis=False):
    """Create contact points from the mesh vertices and normals.

    graspable -- GraspableObject3D
    vis       -- whether to draw the sdf and mesh (and the planes to determine if points are on the handle)
    """
    if vis:
        graspable.plot_sdf_vs_mesh()

    start_loading_contacts = time.time()

    vertex_contacts = []
    on_surface_count = 0
    for i, vertex in graspable.mesh.vertices():
        if graspable.mesh.normals() is None:
            contact = contacts.Contact3D(graspable, np.array(vertex))
        else:
            normal = graspable.mesh.normals()[i] # outward facing normal
            contact = contacts.Contact3D(graspable, np.array(vertex), -normal)
            contact.normal = normal
        contact.friction_cone() # friction cone is cached for later
        vertex_contacts.append(contact)

    # loading ~4100 contacts and computing their friction cones takes ~20 seconds
    logging.info('Loading contacts took %f seconds', time.time() - start_loading_contacts)
    return vertex_contacts

def filter_slip_contacts(vertex_contacts, friction_coef=FRICTION_COEF):
    """Return the contacts that won't slip."""
    start_filtering_contacts = time.time()
    no_slip_indices, no_slip_contacts = [], []
    for i, contact in enumerate(vertex_contacts):
        success, _, _ = contact.friction_cone(friction_coef=friction_coef)
        if success:
            no_slip_indices.append(i)
            no_slip_contacts.append(contact)
    logging.info('Filtering contacts took %f seconds', time.time() - start_filtering_contacts)
    return no_slip_indices, no_slip_contacts

def is_antipodal(c1, c2, friction_coef=FRICTION_COEF):
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

def compute_antipodal_qualities(graspable, contacts, indices,
                                metric='ferrari_canny_L1', friction_coef=FRICTION_COEF):
    n = len(contacts)
    quality_matrix = np.zeros((n, n))

    antipodal_pairs = []
    antipodal_grasps = []
    start_quality = time.time()
    for i, x in enumerate(indices):
        c1 = contacts[x]
        start_quality_iter = time.time()
        logging.info('Computing qualities for index %d', x)

        for j, y in enumerate(indices[i+1:], i+1):
            c2 = contacts[y]

            if is_antipodal(c1, c2):
                antipodal_pairs.append((x, y))

                grasp = grasp_from_contacts(c1, c2)
                antipodal_grasps.append(grasp)

                quality_matrix[x, y] = quality_matrix[y, x] = \
                    q.PointGraspMetrics3D.grasp_quality(grasp, graspable, metric, True, friction_coef)

        logging.info('Took %f seconds to compute qualities for index %d',
                     time.time() - start_quality_iter, x)
    logging.info('Took %f seconds to compute all qualities',
                 time.time() - start_quality)
    return antipodal_pairs, antipodal_grasps, quality_matrix

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

def plot_rays(graspable, grasp_pairs, qualities,
              max_rays=1000, max_width=float('inf'),
              table_vertices=None, table_tris=None):
    """Plots a mesh to visualize grasp quality by plotting colored grasp axes.

    graspable -- GraspableObject3D
    grasp_pairs -- list of pairs of Contact3D
    qualities -- list of grasp qualities

    max_rays -- maximum number of rays to plot
    max_width -- maximum gripper width

    If table_vertices and table_tris are specified, plots a table surface for a
    stable pose.
    """
    obj_to_grid = graspable.sdf.transform_pt_obj_to_grid
    mlab.figure(bgcolor=(1, 1, 1))
    if table_vertices is not None and table_tris is not None:
        table_vertices = np.array([obj_to_grid(np.array(v)) for v in table_vertices])
        mlab.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2],
                             table_tris, representation='surface', color=(0,0,0), opacity=0.5)
    points = np.array([obj_to_grid(np.array(v)) for v in graspable.mesh.vertices()])
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    surf = mlab.triangular_mesh(x, y, z, graspable.mesh.triangles(), color=(0.5, 0.5, 0.5))

    lines = []
    line_qualities = []
    for quality, (c1, c2) in zip(qualities, grasp_pairs):
        if quality != 0 and np.linalg.norm(c1.point - c2.point) <= max_width:
            lines.append((obj_to_grid(c1.point), obj_to_grid(c2.point)))
            line_qualities.append(quality)
    low, high = min(line_qualities), max(line_qualities)
    logging.info('Qualities are between %f and %f', low, high)

    logging.info('%d grasps, will sample below %d', len(lines), max_rays)
    indices = np.random.permutation(len(lines))[:min(len(lines), max_rays)]
    lines = np.array(lines)[indices]
    line_qualities = np.array(line_qualities)[indices]

    f = lambda quality: 0.3 * (quality - low) / (high - low)
    for quality, (start, end) in zip(line_qualities, lines):
        direction = end - start
        direction = 2 * (direction / np.linalg.norm(direction))
        start = start - direction
        end = end + direction
        if quality < 0:
            color = (1, 0, 0)
        else:
            # interpolate colors from red to green
            color = plt.get_cmap('hsv')(f(quality))[:-1]
        tube_radius = 0.04
        mlab.plot3d(*zip(start, end), color=color, tube_radius=tube_radius)
        if quality < 0:
            mlab.plot3d(*zip(start, start-direction), color=(0, 0, 1), tube_radius=tube_radius)
            mlab.plot3d(*zip(end, end+direction), color=(0, 0, 1), tube_radius=tube_radius)

    mlab.draw()
    mlab.show()

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

def vis_stable(graspable, antipodal_pairs, antipodal_grasps,
               all_contacts, qualities,
               stp, vis_transform=True, **kwargs):
    qualities = np.copy(qualities)

    tf = stf.SimilarityTransform3D(pose=tfx.pose(stp.r))
    n = stp.r[2, :]
    x0 = stp.x0

    m = graspable.mesh.transform(tf)
    mn, mx = m.bounding_box()
    d = max(mx[1] - mn[1], mx[0] - mn[0]) / 2
    z = mn[2]
    table_vertices = np.array([[d, d, z],
                               [d, -d, z],
                               [-d, d, z],
                               [-d, -d, z]])
    table_tris = np.array([[0, 1, 2], [1, 2, 3]])
    if vis_transform:
        mlab.figure()
        m.visualize()
        mlab.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2],
                             table_tris, representation='surface', color=(0,0,0))

    for pair, grasp in zip(antipodal_pairs, antipodal_grasps):
        x, y = pair
        g1, g2 = grasp.endpoints()
        if n.dot(g2 - x0) > 0 and n.dot(g1 - x0) > 0:
            continue
        else:
            qualities[x, y] = qualities[y, x] = 0

    contacts = []
    stable_quals = []
    for x in range(qualities.shape[0]):
        for y in range(qualities.shape[1]):
            if qualities[x, y] > 0:
                contacts.append((all_contacts[x], all_contacts[y]))
                stable_quals.append(qualities[x, y])

    tfinv = tf.inverse()
    table_vertices = np.array([tfinv.apply(e) for e in table_vertices])
    plot_rays(graspable, contacts, stable_quals,
              table_vertices=table_vertices, table_tris=table_tris, **kwargs)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    config = {
        'database_dir': '/home/jmahler/brian/uniform_meshes/test_db',
        'database_name': 'google_test_db.hdf5',
        'database_cache_dir': '.',
        'datasets': {
            'google': {'start_index': 0, 'end_index': 1}
        }
    }
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    dataset = database['google']
    obj_name = 'actual-google-drill'
    graspable = dataset[obj_name]

    stable_pose_fname = 'temp.stp'
    stp.StablePoseFile().write_mesh_stable_poses(graspable.mesh, stable_pose_fname, min_prob=0.01)
    stable_poses = stp.StablePoseFile().read(stable_pose_fname)

    all_contacts = load_vertex_contacts(graspable)
    vertex_indices, vertex_contacts = filter_slip_contacts(all_contacts)  # for all grasps
    antipodal_pairs, antipodal_grasps, quality_matrix = compute_antipodal_qualities(graspable, vertex_contacts, vertex_indices)

    # Save expensive computation
    with open('{}-antipodal.txt'.format(obj_name), 'w') as f:
        f.write(str(antipodal_pairs))
    np.save('{}-quality', quality_matrix)
    # dataset.store_grasps(obj_name, antipodal_grasps)

    # ray visualization
    most_stable_pose = stable_poses[-1]
    logging.info('About to plot rays with most stable pose.')
    vis_stable(graspable, antipodal_pairs, antipodal_grasps, all_contacts,
               most_stable_pose, vis_transform=False, max_rays=500)
