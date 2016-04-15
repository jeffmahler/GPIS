"""
Save a bunch of mesh segments
"""
import argparse
import copy
import logging
import pickle as pkl
import os
import random
import string
import time

import colorsys
import IPython
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')

import database as db
import experiment_config as ec
import mesh
import obj_file

if __name__ == '__main__':
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('object_key')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    dataset_name = 'MeshSegBenchmark'
    out_dir = os.path.join('results', 'segments', args.object_key)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # read config file and object
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    ds = database[dataset_name]
    obj = ds[args.object_key]

    # read in mesh segments
    shape_data_dir = config['shape_data_dir']
    seg_dir = os.path.join(shape_data_dir, dataset_name, 'segments')
    seg_filename = os.path.join(seg_dir, '%s/%s_0.seg' %(args.object_key, args.object_key))
    
    f = open(seg_filename, 'r')
    triangle_labels = []
    for line in f:
        triangle_labels.append(int(line))
        
    # setup bufffers
    vertices = obj.mesh.vertices()
    triangles = obj.mesh.triangles()
    label_array = np.array(triangle_labels)
    unique_labels = set(triangle_labels)
    num_colors = len(unique_labels)

    # get colors for plotting
    color_delta = 1.0 / num_colors
    colors = []
    for j in range(len(unique_labels)):
        colors.append(colorsys.hsv_to_rgb(j*color_delta, 0.9, 0.9))

    # form mesh from segments
    mv.figure()
    for label, color in zip(unique_labels, colors):
        logging.info('Saving segment %d' %label)
        label_indices = np.where(label_array == label)[0]
        label_tris = [triangles[i] for i in label_indices.tolist()]

        vertex_indices = []
        for t in label_tris:
            vertex_indices.extend(t)
        vertex_indices = set(vertex_indices)

        label_vertex_index_pairs = [(i, vertices[i]) for i in vertex_indices]
        label_vertices = [v[1] for v in label_vertex_index_pairs]
        vertex_mappings = -1 * np.ones(len(vertices))
        vertex_mappings = vertex_mappings.tolist()
        for i, v in enumerate(label_vertex_index_pairs):
            vertex_mappings[v[0]] = i

        label_tris = [[vertex_mappings[t[0]], vertex_mappings[t[1]], vertex_mappings[t[2]]] for t in label_tris]

        m = mesh.Mesh3D(label_vertices, label_tris)
        of = obj_file.ObjFile(os.path.join(out_dir, 'segment_%d.obj' %(label)))
        of.write(m)

        m.visualize(color=color)
    mv.show()
