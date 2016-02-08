import numpy as np
import mesh
import obj_file as of
import stable_poses as stp

import similarity_tf as stf
import tfx

import pyhull
import pyhull.convex_hull as cvh
import os
import sys

import mayavi.mlab as mv

import IPython

if __name__ == '__main__':
    filename = sys.argv[1]
    ofile = of.ObjFile(filename)
    msh = ofile.read()
    cvh_msh = msh.convex_hull()

    mv.figure(bgcolor=(1,1,1))
    msh.visualize()
    mv.show()
    exit(0)

    min_prob = 0.01
    prob_mapping = stp.compute_stable_poses(msh)
    R_list = []
    for face, p in prob_mapping.items():
        if p >= min_prob:# and face[0] == 161 and face[1] == 16 and face[2] == 67:
            R_list.append([p, stp.compute_basis([cvh_msh.vertices()[i] for i in face], msh), face])

    #    m.num_connected_components()
    for R in R_list:
        print 'Prob', R[0]

        tf = stf.SimilarityTransform3D(pose=tfx.pose(R[1]))
        m = msh.transform(tf)
            
        mv.figure()
        m.visualize()
        #mv.points3d(m.center_of_mass[0], m.center_of_mass[1], m.center_of_mass[2], scale_factor=0.01)
        #mv.points3d(m.vertex_mean_[0], m.vertex_mean_[1], m.vertex_mean_[2], scale_factor=0.01, color=(1,0,0))

        mn, mx = m.bounding_box()
        d = max(mx[1] - mn[1], mx[0] - mn[0]) / 2
        z = mn[2]
        table_vertices = np.array([[d, d, z],
                                   [d, -d, z],
                                   [-d, d, z],
                                   [-d, -d, z]])
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        mv.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2], table_tris, representation='surface', color=(0,0,0))

        #mv.axes()
#    mv.points3d(cvh_m.center_of_mass_[0], cvh_m.center_of_mass_[1], cvh_m.center_of_mass_[2], scale_factor=0.01)
#    mv.points3d(cvh_m.vertex_mean_[0], cvh_m.vertex_mean_[1], cvh_m.vertex_mean_[2], scale_factor=0.01, color=(1,0,0))
        """
        cvh_m = cvh_msh.transform(tf)
        mv.figure()
        cvh_m.visualize()
        mv.points3d(cvh_m.center_of_mass_[0], cvh_m.center_of_mass_[1], cvh_m.center_of_mass_[2], scale_factor=0.01)
        mv.points3d(cvh_m.vertex_mean_[0], cvh_m.vertex_mean_[1], cvh_m.vertex_mean_[2], scale_factor=0.01, color=(1,0,0))

        vertices = np.array(cvh_m.vertices())
        v = np.array([vertices[i] for i in R[2]])
        mv.points3d(v[:,0], v[:,1], v[:,2], scale_factor=0.01, color=(0,1,0))

        mv.axes()
        """
        mv.show()

        #IPython.embed()
 
