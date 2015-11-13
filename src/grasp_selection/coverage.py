"""
Script to compute the Ferrari-Canny metric for all pairs of contact points.
Author: Brian Hou
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import IPython

import logging
import time

import contacts
import database as db
import graspable_object as go
import obj_file
import sdf_file

def load_contacts(graspable, plot=False):
    sdf = graspable.sdf
    mesh = graspable.mesh

    if plot:
        sdf_surface_points, _ = sdf.surface_points()
        vertex_points = np.array([sdf.transform_pt_obj_to_grid(np.array(v)) for v in mesh.vertices()])
        plot_sdf_vs_mesh(sdf_surface_points, vertex_points, max(sdf.dimensions))

    # Loading contacts takes ~16 seconds
    start_loading_contacts = time.time()
    vertex_contacts = []
    on_surface_count = 0
    IPython.embed()
    for vertex, normal in zip(mesh.vertices(), mesh.normals()):
        contact = contacts.Contact3D(graspable, np.array(vertex), -np.array(normal))
        vertex_contacts.append(contact)

        as_grid = contact.graspable.sdf.transform_pt_obj_to_grid(contact.point)
        on_surface, sdf_val = contact.graspable.sdf.on_surface(as_grid)
        if on_surface:
            on_surface_count += 1
        else:
            print(sdf_val)
    print(on_surface_count)
    print('Loading contacts took', time.time() - start_loading_contacts, 'seconds')

    return graspable, vertex_contacts

def plot_sdf_vs_mesh(sdf_surface_points, mesh_surface_points, dim=50):
    def plot_surface(points, color):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        ax.scatter(x, y, z, c=color)

    ax = plt.gca(projection = '3d')
    plot_surface(sdf_surface_points, 'b')
    plot_surface(mesh_surface_points, 'r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(0, dim)
    ax.set_ylim3d(0, dim)
    ax.set_zlim3d(0, dim)

    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    database_filename = 'google_test_db.hdf5'
    config = {
        'database_dir': '.',
        'database_cache_dir': '.',
        'datasets': {
            'google': {'start_index': 0, 'end_index': 1}
        }
    }
    database = db.Hdf5Database(database_filename, config)
    graspable = database['google']['normalized-google-drill']
    vertex_contacts = load_contacts(graspable, plot=True)
    for contact in vertex_contacts:
        contact.plot_friction_cone()
        IPython.embed()
        plt.show()
