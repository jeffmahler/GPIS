import numpy as np
from mayavi import mlab
import time
import wx
import pdb
from tvtk.api import tvtk
import pdb
import mlab_3D_to_2D
import os
import sys
import pickle

other_code_dir = "../src/grasp_selection"
other_code_dir = os.getcwd() + "/" + other_code_dir
sys.path.insert(0, other_code_dir)
sys.stdout = open("mylog.txt", "w")

import obj_file as of
import mesh
import json
import yaml
import glob

#End Imports

figure = mlab.gcf()

mesh_names = ["part1", "gearbox", "printer_feeder"]
mesh_files = ["thingiverse/" + i + ".obj" for i in mesh_names]
camera_views = [-20, -55, 45]
for mesh_name, mesh_file, camera_view in zip(mesh_names, mesh_files, camera_views):
    mlab.clf()
    ofile = of.ObjFile(mesh_file)
    msh = ofile.read()
    mlab.yaw(camera_view)
    msh.visualize((0.0, 0.0, 1.0))
   

    mask_files = glob.glob(os.path.join("partial_masks/", mesh_name) + '_partial_mask*.npy')
    #print(zip(mask_files, range(len(mask_files))))

    progressive_mask = np.array([])


    for mask_file, index in zip(mask_files, range(len(mask_files))):

        try:
            read_file = open(mask_file, 'r')
            triangle_index_list = np.load(read_file)
            read_file.close()
        except Exception:
            pass

        progressive_mask = np.append(progressive_mask, triangle_index_list)


        masked_triangles = [msh.triangles()[int(i)] for i in progressive_mask]
        non_masked_triangles = [msh.triangles()[int(i)] for i in range(len(msh.triangles())) if i not in progressive_mask]
        blue_msh = mesh.Mesh3D(msh.vertices(), non_masked_triangles, msh.normals())
        red_msh = mesh.Mesh3D(msh.vertices(), masked_triangles, msh.normals())
        mlab.clf()
        blue_msh.visualize((0.0, 0.0, 1.0))
        red_msh.visualize((1.0, 0.0, 0.0))
        mlab.savefig("views/" + mesh_name +  "_" + str(index) + ".png")










