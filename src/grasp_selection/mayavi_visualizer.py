import numpy as np
import mayavi.mlab as mv

class MayaviVisualizer:

    # MAYAVI VISUALIZER
    @staticmethod
    def plot_table(T_table_world, d=0.5):
        """ Plots a table in pose T """
        table_vertices = np.array([[d, d, 0],
                                   [d, -d, 0],
                                   [-d, d, 0],
                                   [-d, -d, 0]])
        table_vertices_tf = T_table_world.apply(table_vertices.T).T
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        mv.triangular_mesh(table_vertices[:,0], table_vertices[:,1], table_vertices[:,2], table_tris, representation='surface', color=(0,0,0))

    @staticmethod
    def plot_pose(T_frame_world, alpha=0.5, tube_radius=0.005, center_scale=0.025):
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

    @staticmethod
    def plot_mesh(mesh, T_mesh_world, style='wireframe', color=(0.5,0.5,0.5)):
        mesh_tf = mesh.transform(T_mesh_world.inverse())
        mesh_tf.visualize(style=style, color=color)

    @staticmethod
    def plot_point_cloud(point_cloud, T_points_world, color=(0,1,0), scale=0.01):
        point_cloud_tf = T_points_world.apply(point_cloud).T
        mv.points3d(point_cloud_tf[:,0], point_cloud_tf[:,1], point_cloud_tf[:,2], color=color, scale_factor=scale)

    @staticmethod
    def plot_grasp(grasp, T_obj_world, alpha=0.5, tube_radius=0.005, endpoint_color=(0,1,0), endpoint_scale=0.01, grasp_axis_color=(0,1,0), palm_axis_color=(0,0,1)):
        g1, g2 = grasp.endpoints()
        center = grasp.center
        g1_tf = T_obj_world.inverse().apply(g1)
        g2_tf = T_obj_world.inverse().apply(g2)
        center_tf = T_obj_world.inverse().apply(center)
        grasp_axis_tf = np.array([g1_tf, g2_tf])

        T_gripper_obj = grasp.gripper_transform(gripper='zeke')
        palm_axis = T_gripper_obj.rotation[:,1]

        axis_tf = np.array([g1_tf, g2_tf])
        palm_axis_tf = T_obj_world.inverse().apply(palm_axis, direction=True)
        palm_axis_tf = np.array([center_tf, center_tf + alpha * palm_axis_tf])

        mv.points3d(g1_tf[0], g1_tf[1], g1_tf[2], color=endpoint_color, scale_factor=endpoint_scale)
        mv.points3d(g2_tf[0], g2_tf[1], g2_tf[2], color=endpoint_color, scale_factor=endpoint_scale)

        mv.plot3d(grasp_axis_tf[:,0], grasp_axis_tf[:,1], grasp_axis_tf[:,2], color=grasp_axis_color, tube_radius=tube_radius)
        mv.plot3d(palm_axis_tf[:,0], palm_axis_tf[:,1], palm_axis_tf[:,2], color=palm_axis_color, tube_radius=tube_radius)
        
