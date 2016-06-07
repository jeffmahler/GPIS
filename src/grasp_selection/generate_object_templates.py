import logging
import IPython
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import mayavi_visualizer as mv
import numpy as np
import os
import sys

import database as db
import experiment_config as ec
import similarity_tf as stf
import tfx

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    output_dir = sys.argv[2]

    dpi = 100
    line_width = 6.0
    alpha = 10.0

    # open up the config and database
    config = ec.ExperimentConfig(config_filename)
    database_name = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_name, config)
    dataset = database.dataset(config['datasets'].keys()[0])

    # get obj
    for obj in dataset:
        object_key = obj.key
        logging.info('Generating templates for object %s' %(object_key))
        sdf = obj.sdf
        stable_poses = dataset.stable_poses(object_key)

        for stable_pose in stable_poses:
            # transform the SDF and point on the plane
            tf = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r))
            sdf_tf = sdf.transform(tf, detailed=False)
            x0_grid = sdf_tf.transform_pt_obj_to_grid(tf.apply(stable_pose.x0))

            n = stable_pose.r[2,:]
            x0 = stable_pose.x0
            com = obj.mesh.center_of_mass
            t = n.dot(x0 - com) / np.linalg.norm(n)
            com_grid = sdf_tf.transform_pt_obj_to_grid(tf.apply(com))
            com_proj = com + t * n
            com_proj_grid = sdf_tf.transform_pt_obj_to_grid(tf.apply(com_proj))

            x_axis_line = np.array([com_proj_grid[:2], com_proj_grid[:2] + alpha * np.array([1,0])])
            y_axis_line = np.array([com_proj_grid[:2], com_proj_grid[:2] + alpha * np.array([0,1])])
            x_axis_3d_line = np.array([com_grid, com_grid + alpha * np.array([1,0,0])])
            y_axis_3d_line = np.array([com_grid, com_grid + alpha * np.array([0,1,0])])

            # visualize
            logging.info('Stable pose %s' %(stable_pose.id))
            if False:
                ax = plt.gca(projection = '3d')
                ax.scatter(x0_grid[0], x0_grid[1], x0_grid[2], c='m', s=120)
                ax.scatter(com_grid[0], com_grid[1], com_grid[2], c='y', s=150)
                ax.plot(x_axis_3d_line[:,0], x_axis_3d_line[:,1], x_axis_3d_line[:,2], c='r', linewidth=line_width)
                ax.plot(y_axis_3d_line[:,0], y_axis_3d_line[:,1], y_axis_3d_line[:,2], c='g', linewidth=line_width)
                sdf_tf.scatter()
                plt.show()

            # slice the sdf
            z_slice = int(np.ceil(x0_grid[2])+1)
            sdf_slice = sdf_tf.data_[:,:,z_slice]
            surface_points = np.where(np.abs(sdf_slice) < sdf_tf.surface_thresh_)
            
            sdf_binary_slice = 255.0 * np.ones(sdf_slice.shape)
            sdf_binary_slice[surface_points[0], surface_points[1]] = 0.0

            # plot and save the slice with center of mass, axis
            plt.figure()
            plt.imshow(sdf_binary_slice, cmap=plt.cm.Greys_r, interpolation='none')
            plt.scatter(com_proj_grid[1], com_proj_grid[0], c='b', s=150)
            plt.plot(x_axis_line[:,1], x_axis_line[:,0], c='r', linewidth=line_width, linestyle=':')
            plt.plot(y_axis_line[:,1], y_axis_line[:,0], c='g', linewidth=line_width, linestyle='--')
            plt.text(0, 0, 'z=%.3f'%(abs(t)))
            plt.axis('off')
            figname = '%s_template_%s.svg' %(object_key, stable_pose.id)
            plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

            # save the stable pose rotation matrix
            filename = '%s_template_%s.npy' %(object_key, stable_pose.id)
            np.save(os.path.join(output_dir, filename), stable_pose.r)

