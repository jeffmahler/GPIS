import IPython
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mayavi.mlab as mv
from random import choice
import os
import sys

import database as db
import experiment_config as ec
import similarity_tf as stf
import tfx

# MAYAVI PLOTTING FUNCTIONS
def mv_plot_table(T_table_world, d=0.5):
        """ Plots a table in pose T """
        table_vertices = np.array([[d, d, 0],
                                   [d, -d, 0],
                                   [-d, d, 0],
                                   [-d, -d, 0]])
        table_vertices_tf = T_table_world.apply(table_vertices.T).T
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        mv.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2], table_tris, representation='surface', color=(0,0,0))

def mv_plot_pose(T_frame_world, alpha=0.5, tube_radius=0.005, center_scale=0.025):
        T_world_frame = T_frame_world.inverse()
        R = T_world_frame.rotation
        t = T_world_frame.translation

        x_axis_tf = np.array([t, t + alpha * R[:,0]])
        y_axis_tf = np.array([t, t + alpha * R[:,1]])
        z_axis_tf = np.array([t, t + alpha * R[:,2]])
        
        mv.points3d(t[0], t[1], t[2], color=(1,1,1), scale_factor=center_scale)

        mv.plot3d(x_axis_tf[:,0], x_axis_tf[:,1], x_axis_tf[:,2], color=(1,0,0), tube_radius=tube_radius)
        mv.plot3d(y_axis_tf[:,0], y_axis_tf[:,1], y_axis_tf[:,2], color=(0,1,0), tube_radius=tube_radius)
        mv.plot3d(z_axis_tf[:,0], z_axis_tf[:,1], z_axis_tf[:,2], color=(0,0,1), tube_radius=tube_radius)

        mv.text3d(t[0], t[1], t[2], ' %s' %T_frame_world.to_frame.upper(), scale=0.01)

def mv_plot_mesh(mesh, T_mesh_world, style='wireframe', color=(0.5,0.5,0.5)):
        mesh_tf = mesh.transform(T_mesh_world.inverse())
        mesh_tf.visualize(style=style, color=color)

def mv_plot_point_cloud(point_cloud, T_points_world, color=(0,1,0), scale=0.01):
        point_cloud_tf = T_points_world.apply(point_cloud).T
        mv.points3d(point_cloud_tf[:,0], point_cloud_tf[:,1], point_cloud_tf[:,2], color=color, scale_factor=scale)

config_file = "../../cfg/test_hdf5_label_grasps_gce_jacky.yaml"
database_filename = "/mnt/terastation/shape_data/MASTER_DB_v3/aselab_db.hdf5"
dataset_name = "aselab"
item_name = "spray"

alpha = 0.05
center_scale = 0.0075
tube_radius = 0.0025
table_extent = 0.5

config = ec.ExperimentConfig(config_file)
database = db.Hdf5Database(database_filename, config)

# read the grasp metrics and features
ds = database.dataset(dataset_name)
graspable = ds.graspable(item_name)
grasps = ds.grasps(item_name)
grasp_features = ds.grasp_features(item_name, grasps)
grasp_metrics = ds.grasp_metrics(item_name, grasps)
stable_poses = ds.stable_poses(item_name)

p = len(stable_poses)
n = len(grasps)

def get_aligned_grasp_sp_pair():
	grasp = choice(grasps)
	stable_pose = stable_poses[2]

	return grasp.grasp_aligned_with_stable_pose(stable_pose), stable_pose

#test stable pose aligned grasps
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)  

def add_arrow(ax, origin, dest, c="r"):
	ax.add_artist(Arrow3D([origin[0], dest[0]], [origin[1], dest[1]], [origin[2], dest[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color=c))

def test_align():
	grasp, stable_pose = get_aligned_grasp_sp_pair()
	grasp_axis = grasp.rotated_full_axis.T

	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')

	#add_arrow(ax, stable_pose.x0, stable_pose.r[0], "r")
	#add_arrow(ax, stable_pose.x0, stable_pose.r[1], "g")
	add_arrow(ax, stable_pose.x0, stable_pose.r[2], "b")

	add_arrow(ax, grasp.center, np.ravel(grasp_axis[0]), "orange")
	#add_arrow(ax, grasp.center, np.ravel(grasp_axis[1]), "gray")
	#add_arrow(ax, grasp.center, np.ravel(grasp_axis[2]), "brown")
	
	ax.set_xlabel('x_values')
	ax.set_ylabel('y_values')
	ax.set_zlabel('z_values')

	plt.draw()
	plt.show()

#test collides with stable pose
def test_collide():
	grasp, stable_pose = get_aligned_grasp_sp_pair()
	grasp_axes = grasp.rotated_full_axis
	debug_output = []
	does_collide = grasp.collides_with_stable_pose(stable_pose, debug_output)
	print "Collide? {0}".format(does_collide)

	if does_collide:
		# transform the mesh
		T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
		object_mesh = graspable.mesh
		object_mesh_tf = object_mesh.transform(T_obj_stp)
		mn, mx = object_mesh_tf.bounding_box()
		z = mn[2]

                # define poses
                T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')

                R_stp_obj = stable_pose.r
                T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')

                t_stp_table = np.array([0, 0, z])
                T_stp_table = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), t_stp_table), from_frame='table', to_frame='stp')

                T_obj_world = T_obj_stp.dot(T_stp_table).dot(T_table_world)

                T_gripper_obj = grasp.gripper_transform(gripper='zeke')
                
                T_gripper_world = T_gripper_obj.dot(T_obj_world)

		mv.figure()
                mv_plot_table(T_table_world, d=table_extent)
                mv_plot_pose(T_table_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
                mv_plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
                mv_plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
                mv_plot_mesh(object_mesh, T_obj_world)
                mv_plot_point_cloud(collision_box_vertices, T_obj_world)
                mv.show()

	return does_collide

def test_grasp_alignment(graspable, grasp, stable_pose):
        grasp = grasp.grasp_aligned_with_stable_pose(stable_pose)
	debug_output = []
	does_collide = grasp.collides_with_stable_pose(stable_pose, debug_output)
	print "Collide? {0}".format(does_collide)

        # transform the mesh
        T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
        object_mesh = graspable.mesh
        object_mesh_tf = object_mesh.transform(T_obj_stp)
        mn, mx = object_mesh_tf.bounding_box()
        z = mn[2]

        # define poses
#        R_camera_table = 
        T_camera_table = stf.SimilarityTransform3D(from_frame='table', to_frame='camera')

        T_table_world = stf.SimilarityTransform3D(from_frame='world', to_frame='table')
        
        R_stp_obj = stable_pose.r
        T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(R_stp_obj.T, np.zeros(3)), from_frame='stp', to_frame='obj')
        
        t_stp_table = np.array([0, 0, z])
        T_stp_table = stf.SimilarityTransform3D(pose=tfx.pose(np.eye(3), t_stp_table), from_frame='table', to_frame='stp')
        
        T_obj_world = T_obj_stp.dot(T_stp_table).dot(T_table_world)
        
        T_gripper_obj = grasp.gripper_transform(gripper='zeke')
        
        T_gripper_world = T_gripper_obj.dot(T_obj_world)
        
        mv.figure()
        mv_plot_table(T_table_world, d=table_extent)
        mv_plot_pose(T_table_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_pose(T_obj_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_pose(T_gripper_world, alpha=alpha, tube_radius=tube_radius, center_scale=center_scale)
        mv_plot_mesh(object_mesh, T_obj_world)
        mv_plot_point_cloud(collision_box_vertices, T_obj_world)
        mv.show()

for i in range(1000):
	print 'Iteration', i
	test_collide()
