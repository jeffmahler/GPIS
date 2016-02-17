
                return

		object_mesh_tf.visualize(style='wireframe')

		grasp_center_tf = T_obj_stp.apply(grasp.center)

		#R_pr2_zeke = np.eye(3)
		R_pr2_zeke = np.array([[0, -1, 0],
                              [1, 0, 0],
                              [0, 0, 1]])
		grasp_axes = R_pr2_zeke.dot(grasp_axes)
		grasp_axes_tf = T_obj_stp.apply(grasp_axes, direction=True)
		grasp_x_axis_tf = np.array([grasp_center_tf, grasp_center_tf + alpha * grasp_axes_tf[:,0]])
		grasp_y_axis_tf = np.array([grasp_center_tf, grasp_center_tf + alpha * grasp_axes_tf[:,1]])
		grasp_z_axis_tf = np.array([grasp_center_tf, grasp_center_tf + alpha * grasp_axes_tf[:,2]])

		mv.points3d(grasp_center_tf[0], grasp_center_tf[1], grasp_center_tf[2], color=(0,0,0), scale_factor=0.025)
		for vertex in debug[0]:
			vertex_tf = T_obj_stp.apply(vertex)
			mv.points3d(vertex_tf[0], vertex_tf[1], vertex_tf[2], color=(1,0,0), scale_factor=0.01)
		mv.plot3d(grasp_x_axis_tf[:,0], grasp_x_axis_tf[:,1], grasp_x_axis_tf[:,2], color=(1,0,0), tube_radius=tube_radius)
		mv.plot3d(grasp_y_axis_tf[:,0], grasp_y_axis_tf[:,1], grasp_y_axis_tf[:,2], color=(0,1,0), tube_radius=tube_radius)
		mv.plot3d(grasp_z_axis_tf[:,0], grasp_z_axis_tf[:,1], grasp_z_axis_tf[:,2], color=(0,0,1), tube_radius=tube_radius)
		mv.show()
