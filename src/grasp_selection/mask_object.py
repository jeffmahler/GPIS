"""
Script for protecting meshes from triangle masks.

Usage: python src/grasp_selection/mask_object.py masks/turbine_housing.obj

Author: Brian Hou
"""

import argparse
import logging
import os
import time

import numpy as np
import scipy.spatial as ss

import obj_file as of
import mesh as m

def remove_intersections(unmasked, masked):
    # Remove vertices from unmasked region that would be inside masked region
    logging.info('Removing vertices that are inside masked hull.')
    hull_start = time.time()
    hull = ss.Delaunay(np.array(masked.vertices()))
    in_hull = hull.find_simplex(np.array(unmasked.vertices())) >= 0
    in_hull_indices = set(np.where(in_hull)[0])
    unmasked_no_hull_tris = []
    for tri in unmasked.triangles():
        intersect = set(tri).intersection(in_hull_indices)
        if not intersect or (intersect and len(intersect) == 3):
            unmasked_no_hull_tris.append(tri)
    logging.info('Went from %d triangles to %d',
                  len(unmasked_tris_new), len(unmasked_no_hull_tris))
    logging.info('Took %f seconds.', time.time() - hull_start)
    unmasked.set_triangles(unmasked_no_hull_tris)
    unmasked.remove_unreferenced_vertices()


def load_mask_indices(mask_fname):
    return np.load(mask_fname)


def color_vertices(mesh, vert_bitmask, fname_out, color=np.array([1, 0, 0])):
    """Create a new mesh with the vertices in the vert_bitmask colored and write the
    mesh to fname_out.

    mesh -- mesh.Mesh3D
    vert_bitmask -- bitarray with the same length as mesh.vertices()
    fname_out -- string path to output file
    color -- np 3-array color (RGB)
    """
    color_start = time.time()
    colored_mesh = mesh.copy()
    colors = colored_mesh.colors()
    colors[vert_bitmask] = color
    colored_mesh.set_colors(colors)
    of.ObjFile(fname_out).write(colored_mesh)
    logging.info('Took %f seconds.', time.time() - color_start)


def filter_faces(mesh, face_bitmask):
    """Creates a new mesh that only keeps the faces in the face_bitmask and removes
    unreferenced vertices.

    mesh -- mesh.Mesh3D
    face_bitmask -- bitarray with the same length as mesh.triangles()
    """
    filter_start = time.time()
    filtered = mesh.copy()
    filtered_tris = np.array(filtered.triangles())
    filtered_tris_new = filtered_tris[face_bitmask].tolist()
    filtered.set_triangles(filtered_tris_new)
    filtered.remove_unreferenced_vertices()
    logging.info('Took %f seconds.', time.time() - filter_start)
    return filtered


def zipper(unmasked, masked, masked_vert_bitmask, fname_out):
    """Create a new mesh with the vertices and triangles of both unmasked and masked
    regions, and write the mesh to fname_out.

    Also, keep track of which vertex indices are referenced by new triangles.

    unmasked, masked -- mesh.Mesh3D
    masked_vert_bitmask -- zero bitarray that *will be mutated*
    fname_out -- string path to output file
    """
    zipper_start = time.time()
    offset = len(unmasked.vertices())

    new_verts = unmasked.vertices() + masked.vertices()
    new_tris = list(unmasked.triangles())

    for tri in masked.triangles():
        new_tri = [v + offset for v in tri]
        masked_vert_bitmask[new_tri] = 1
        new_tris.append(new_tri)

    merged = m.Mesh3D(new_verts, new_tris)
    logging.info('Took %f seconds.', time.time() - zipper_start)
    of.ObjFile(fname_out).write(merged)
    return merged


def mask(mesh, mask_indices, output_paths):
    """Return a mesh where the masked faces are replaced with their convex hull.

    mesh -- mesh.Mesh3D
    mask_indices -- np.array of face indices (0-indexed)
    output_paths -- list of string paths to files
    """
    bitmask = np.zeros(len(mesh.triangles()), np.bool)
    bitmask[mask_indices] = 1

    # Color mask on original mesh
    logging.info('Coloring original mesh.')
    mask_indices = set(mask_indices)
    vert_bitmask = np.zeros(len(mesh.vertices()), np.bool)
    for i, tri in enumerate(mesh.triangles()):
        if i in mask_indices:
            vert_bitmask[tri] = 1
    color_vertices(mesh, vert_bitmask, output_paths[0])

    # Compute masked region
    logging.info('Filtering masked faces.')
    masked_only = filter_faces(mesh, bitmask)

    logging.info('Computing convex hull of masked region.')
    masked_hull = masked_only.convex_hull()
    of.ObjFile(output_paths[1]).write(masked_hull)

    logging.info('Computing bounding box of masked region.')
    masked_bbox = masked_only.bounding_box_mesh()
    of.ObjFile(output_paths[2]).write(masked_bbox)

    # Compute unmasked region
    logging.info('Filtering unmasked faces.')
    unmasked_only = filter_faces(mesh, ~bitmask)
    of.ObjFile(output_paths[3]).write(unmasked_only)

    # Combine masked and unmasked region
    logging.info('Combining masked and unmasked regions.')
    new_num_verts = len(unmasked_only.vertices()) + len(masked_hull.vertices())
    masked_vert_bitmask = np.zeros(new_num_verts, np.bool)
    masked_hull = zipper(unmasked_only, masked_hull,
                         masked_vert_bitmask, output_paths[4])

    # Color hull region on combined mesh
    color_vertices(masked_hull, masked_vert_bitmask, output_paths[5])

    # Combine masked and unmasked region
    logging.info('Combining masked and unmasked regions.')
    new_num_verts = len(unmasked_only.vertices()) + len(masked_bbox.vertices())
    masked_vert_bitmask = np.zeros(new_num_verts, np.bool)
    masked_bbox = zipper(unmasked_only, masked_bbox,
                         masked_vert_bitmask, output_paths[6])

    # Color hull region on combined mesh
    color_vertices(masked_bbox, masked_vert_bitmask, output_paths[7])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_file')
    args = parser.parse_args()

    # Read mesh to be masked
    mesh = of.ObjFile(args.obj_file).read()

    obj_dir, obj_fname = os.path.split(args.obj_file)
    obj_name = obj_fname.split('.')[0]

    # Load mask indices
    mask_fname = '{}_mask.npy'.format(obj_name)
    mask_indices = load_mask_indices(os.path.join(obj_dir, mask_fname))

    # Apply mask to create new object, write masked object to *_masked.obj
    output_files = [
        os.path.join(obj_dir, 'output', s.format(obj_name)) for s in [
            '{}_mask_colored.obj',   # color masked faces (i.e. Sherdil's viewer)
            '{}_mask_hull.obj',      # hull only (no color)
            '{}_mask_bbox.obj',      # bbox only (no color)
            '{}_no_mask.obj',        # unmasked region only (no color)

            '{}_masked_hull.obj',         # masked (hull) obj (no color)
            '{}_masked_hull_colored.obj', # masked (hull) obj
            '{}_masked_bbox.obj',         # masked (bbox) obj (no color)
            '{}_masked_bbox_colored.obj', # masked (bbox) obj
         ]
    ]
    mask(mesh, mask_indices, output_files)

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()
