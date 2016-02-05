import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from random import choice

import database as db
import experiment_config as ec

config_file = "../../cfg/test_hdf5_label_grasps_gce_jacky.yaml"
database_filename = "/mnt/terastation/shape_data/MASTER_DB_v3/dexnet_db3_01_22_16.hdf5"
dataset_name = "YCB"
item_name = "frenchs_classic_yellow_mustard_14oz"

config = ec.ExperimentConfig(config_file)
database = db.Hdf5Database(database_filename, config)

# read the grasp metrics and features
ds = database.dataset(dataset_name)
grasps = ds.grasps(item_name)
grasp_features = ds.grasp_features(item_name, grasps)
grasp_metrics = ds.grasp_metrics(item_name, grasps)
stable_poses = ds.stable_poses(item_name)

p = len(stable_poses)
n = len(grasps)

def get_aligned_grasp_sp_pair():
	grasp = choice(grasps)
	stable_pose = choice(stable_poses)

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
	add_arrow(ax, grasp.center, np.ravel(grasp_axis[2]), "brown")
	
	ax.set_xlabel('x_values')
	ax.set_ylabel('y_values')
	ax.set_zlabel('z_values')

	plt.draw()
	plt.show()

#test collides with stable pose
def test_collide():
	grasp, stable_pose = get_aligned_grasp_sp_pair()
	does_collide = grasp.collides_with_stable_pose(stable_pose)
	print "Collide? {0}".format(does_collide)



