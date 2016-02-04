import logging
import IPython
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import database as db
import experiment_config as ec
import similarity_tf as stf
import tfx

"""
NAMING CONVENTION:
T_{a}_{b}
pose of a frame wrt b frame
"""

def plot_pose(T, alpha=1.0, line_width=2.0):
    """ Provide rotation R and translation t of frame wrt world """
    T_inv = T.inverse()
    R = T_inv.rotation
    t = T_inv.translation

    x_axis_3d_line = np.array([t, t + alpha * R[:,0]])
    y_axis_3d_line = np.array([t, t + alpha * R[:,1]])
    z_axis_3d_line = np.array([t, t + alpha * R[:,2]])

    ax = plt.gca(projection = '3d')
    ax.plot(x_axis_3d_line[:,0], x_axis_3d_line[:,1], x_axis_3d_line[:,2], c='r', linewidth=line_width)
    ax.plot(y_axis_3d_line[:,0], y_axis_3d_line[:,1], y_axis_3d_line[:,2], c='g', linewidth=line_width)
    ax.plot(z_axis_3d_line[:,0], z_axis_3d_line[:,1], z_axis_3d_line[:,2], c='b', linewidth=line_width)

    ax.scatter(t[0], t[1], t[2], c='k', s=150)
    ax.text(t[0], t[1], t[2], T.to_frame.upper())

def plot_mesh(mesh, T, color='m', size=20):
    """ Plots a mesh object in pose T """
    ax = plt.gca(projection = '3d')
    mesh_tf = mesh.transform(T.inverse())
    mesh_tf_vertices = np.array(mesh_tf.vertices())
    ax.scatter(mesh_tf_vertices[:,0], mesh_tf_vertices[:,1], mesh_tf_vertices[:,2], c=color, s=size)

def plot_grasp(grasp, T, color='c', size=1, n=25):
    """ Plots a grasp object in pose T """
    ax = plt.gca(projection = '3d')
    g1, g2 = best_grasp.endpoints()
    grasp_points = np.c_[np.array(g1), np.array(g2)]
    grasp_points_tf = T.inverse().apply(grasp_points)
    grasp_points_tf = grasp_points_tf.T
    g1_tf = grasp_points_tf[0,:]
    g2_tf = grasp_points_tf[1,:]

    interp_grasp_points = []
    for m in range(1,n-1):
        t = m * 1.0 / n
        interp_grasp_points.append(t * g1_tf + (1 - t) * g2_tf)
    interp_grasp_points = np.array(interp_grasp_points)

    ax.scatter(g1_tf[0], g1_tf[1], g1_tf[2], c=color, s=size*125)
    ax.scatter(g2_tf[0], g2_tf[1], g2_tf[2], c=color, s=size*125)
    ax.scatter(interp_grasp_points[:,0], interp_grasp_points[:,1], interp_grasp_points[:,2], c=color, s=size*25)    

    T_grasp_obj = grasp.gripper_pose_stf()
    approach_line = np.c_[grasp.center, grasp.center + 0.1*T_grasp_obj.rotation[:,0]]
    approach_line_tf = T.inverse().apply(approach_line)
    approach_line_tf = approach_line_tf.T
    ax.plot(approach_line_tf[:,0], approach_line_tf[:,1], approach_line_tf[:,2], c='y', linewidth=10)

if __name__ == '__main__':
    stp_filename = sys.argv[1]
    delta_y = float(sys.argv[2])
    delta_z = float(sys.argv[3])

    line_width = 6.0
    alpha = 0.05
    dim = 0.3
    test = True

    # read in stable pose rotation
    R_stp_obj = np.load(stp_filename)

    R_stp_cb = np.eye(3)
    t_stp_cb = np.array([0.0, delta_y, delta_z])
    T_stp_cb = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_cb, -t_stp_cb), from_frame='cb', to_frame='stp')
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')
    T_obj_cb = T_obj_stp.dot(T_stp_cb)
    T_cb_obj = T_obj_cb.inverse()

    # TEST (pretend theres a chessboard) 
    if test:
        # load the object
        object_key = 'spray'
        config_filename = 'cfg/test_cnn_database_indexer.yaml'
        config = ec.ExperimentConfig(config_filename)
        database_name = os.path.join(config['database_dir'], config['database_name'])
        database = db.Hdf5Database(database_name, config)
        dataset = database.dataset(config['datasets'].keys()[0])
        obj = dataset.graspable(object_key)
        mesh = obj.mesh

        # get the best grasp
        metric = 'pfc_f_0.200000_tg_0.020000_rg_0.020000_to_0.020000_ro_0.020000'
        sorted_grasps, sorted_metrics = dataset.sorted_grasps(object_key, metric)
        best_grasp = sorted_grasps[0]

        # plot all of the poses
        R_cb_world = np.eye(3)
        t_cb_world = np.zeros(3)
        T_cb_world = stf.SimilarityTransform3D(pose=tfx.pose(R_cb_world, t_cb_world), to_frame='cb')
        
        plot_pose(T_cb_world, alpha=alpha)
        plot_pose(T_stp_cb, alpha=alpha)
        plot_pose(T_obj_cb, alpha=alpha)

        plot_mesh(mesh, T_obj_cb)

        plot_grasp(best_grasp.align_with_stable_pose(R_stp_obj[2,:]), T_obj_cb)

        ax = plt.gca(projection = '3d')
        ax.set_xlim3d(-dim, dim)
        ax.set_ylim3d(-dim, dim)
        ax.set_zlim3d(-dim, dim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
