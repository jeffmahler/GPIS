import os
from tfx import pose, point, transform, rotation
from numpy import array, matrix, dot, cos, sin, pi, cross
from numpy.linalg import norm, inv
import database as db
import experiment_config as ec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

#constants
config_dir = 'f:/Dev/ws/GPIS/cfg/test_hdf5_label_grasps_gce_jacky.yaml'
dataset_name = "YCB"
obj_name = "stanley_13oz_hammer"

#db connection
config = ec.ExperimentConfig(config_dir)
database_filename = os.path.join(config['database_dir'], config['database_name'])
database = db.Hdf5Database(database_filename, config)

#getting the grasps
ds = database.dataset(dataset_name)
grasps = ds.grasps(obj_name)
stable_poses = ds.stable_poses(obj_name)

#choosing one sample object and grasp
grasp = grasps[0]
grasp_axis_y = grasp.axis
grasp_axis_x = array([grasp_axis_y[1], -grasp_axis_y[0], 0])
grasp_axis_x = grasp_axis_x / norm(grasp_axis_x)
grasp_axis_z = cross(grasp_axis_x, grasp_axis_y)
grasp_axis_o = matrix([grasp_axis_x, grasp_axis_y, grasp_axis_z]).T

grasp_center = grasp.center
stable_pose = stable_poses[0]
table_normal = stable_pose.r[2]

#solving angle of y axis rotation so grasp is parallel to table surface
class DexGraspYAngleSolver:

    DEFAULT_SAMPLE_SIZE = 1000

    @staticmethod
    def solve(grasp_axis, table_normal, n = DEFAULT_SAMPLE_SIZE):
        return DexGraspYAngleSolver._argmin(DexGraspYAngleSolver._getMatrixProductXAxis(array(grasp_axis), array(table_normal)), 0, 2*pi, n)

    @staticmethod
    def _getMatrixProductXAxis(grasp_axis, table_normal):
        def matrixProduct(theta):
            R = matrix([[cos(theta), 0, -sin(theta)], [0, 1, 0], [sin(theta), 0, cos(theta)]])
            grasp_axis_rotated = dot(R, grasp_axis)
            grasp_axis_rotated_vector = array([grasp_axis_rotated[0, i] for i in range(3)])
            return abs(dot(table_normal, grasp_axis_rotated_vector))
        return matrixProduct
        
    @staticmethod
    def _argmin(f, a, b, n):
        #finds the argmax x of f(x) in the range [a, b) with n samples
        delta = (b - a) / n
        min_y = f(a)
        min_x = a
        for i in range(1, n):
            x = i * delta
            y = f(x)
            if y <= min_y:
                min_y = y
                min_x = x

        return min_x

def col_from_mat(mat, i):
    return array([mat[j, i] for j in range(3)])
        
theta = DexGraspYAngleSolver.solve([1,0,0], dot(inv(grasp_axis_o), table_normal))
cos_t = cos(theta)
sin_t = sin(theta)
R = matrix([[cos_t, 0, -sin_t], [0, 1, 0], [sin_t, 0, cos_t]])

#new rotated grasp axis that's parallel to stable pose plane
grasp_axis_r = dot(grasp_axis_o, R)

#draw a vector
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
def plot_rotated_grasp_axis():
    def get_arrow(origin, dest, color = "k"):
        return Arrow3D([origin[0], dest[0]], [origin[1], dest[1]], [origin[2], dest[2]], mutation_scale=20, lw=1, arrowstyle="-|>", color = color)
    grasp_axis_x_r = col_from_mat(grasp_axis_r, 0)
    grasp_axis_y_r = col_from_mat(grasp_axis_r, 1)
    grasp_axis_z_r = col_from_mat(grasp_axis_r, 2)
    fig = plt.figure()  
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    vec_n = get_arrow(grasp_center, table_normal)
    vec_x = get_arrow(grasp_center, grasp_axis_x_r, "r")
    vec_y = get_arrow(grasp_center, grasp_axis_y_r, "g")
    vec_z = get_arrow(grasp_center, grasp_axis_z_r, "b")

    vec_x_o = get_arrow(grasp_center, grasp_axis_x, "yellow")
    vec_y_o = get_arrow(grasp_center, grasp_axis_y, "gray")
    vec_z_o = get_arrow(grasp_center, grasp_axis_z, "brown")
    ax.add_artist(vec_n)
    ax.add_artist(vec_x)
    ax.add_artist(vec_y)
    ax.add_artist(vec_z)

    ax.add_artist(vec_x_o)
    ax.add_artist(vec_y_o)
    ax.add_artist(vec_z_o)
    plt.show()

#grasp pose in object frame:
grasp_obj = pose(point(grasp_center), grasp_axis_r, frame="OBJECT")