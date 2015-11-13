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
import graspable_object as go
import obj_file
import sdf_file

def load_graspable_and_contacts(mesh_file_name, sdf_file_name):
    # Loading GraspableObject takes ~2 seconds
    start_loading = time.time()
    sdf = sdf_file.SdfFile(sdf_file_name).read()
    mesh = obj_file.ObjFile(mesh_file_name).read()
    graspable = go.GraspableObject3D(sdf, mesh=mesh)
    print('Loading GraspableObject took', time.time() - start_loading, 'seconds')

    # Loading contacts takes ~16 seconds
    start_loading_contacts = time.time()
    vertex_contacts = []
    on_surface_count = 0
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

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    mesh_file_name = '../normalized-google-drill.obj'
    sdf_file_name = '../normalized-google-drill.sdf'
    graspable, vertex_contacts = \
        load_graspable_and_contacts(mesh_file_name, sdf_file_name)
    for contact in vertex_contacts:
        contact.plot_friction_cone()
        IPython.embed()
        plt.show()
