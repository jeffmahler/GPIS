"""
Script to evaluate the probability of success for a few grasps on Izzy, logging the target states and the predicted quality in simulation
Authors: Jeff Mahler and Jacky Liang
"""
import logging
import IPython
import numpy as np
import mayavi.mlab as mv

import os
import shutil
import sys
sys.path.append("src/grasp_selection/control/DexControls")

from DexController import DexController
from DexRobotIzzy import DexRobotIzzy
from ZekeState import ZekeState
from IzzyState import IzzyState
import discrete_adaptive_samplers as das
import termination_conditions as tc

from MayaviVisualizer import MayaviVisualizer
from mab_single_object_objective import MABSingleObjectObjective

import rgbd_sensor as rs
import database as db
import experiment_config as ec
import tfx
import similarity_tf as stf

# Experiment tag generator for saving output
def gen_experiment_id(n=10):
    """ Random string for naming """
    chrs = 'abcdefghijklmnopqrstuvwxyz'
    inds = np.random.randint(0,len(chrs), size=n)
    return ''.join([chrs[i] for i in inds])

def compute_grasp_set(dataset, object_name, stable_pose, num_grasps, metric='pfc_f_0.200000_tg_0.020000_rg_0.020000_to_0.020000_ro_0.020000'):
    """ Add the best grasp according to PFC as well as num_grasps-1 uniformly at random from the remaining set """
    grasp_set = []

    # get sorted list of grasps to ensure that we get the top grasp
    sorted_grasps, sorted_metrics = dataset.sorted_grasps(object_name, metric)

    num_total_grasps = len(sorted_grasps)
    best_grasp = sorted_grasps[0]
    grasp_set.append(best_grasp)

    # get random indices
    indices = np.arange(1, num_total_grasps)
    np.random.shuffle(indices)

    i = 0
    while len(grasp_set) < num_grasps:
        grasp_candidate = sorted_grasps[indices[i]]
        grasp_candidate = grasp_candidate.grasp_aligned_with_stable_pose(stable_pose)
        in_collision = grasp_candidate.collides_with_stable_pose(stable_pose)
        center_rel_table = grasp_candidate.center - stable_pose.x0
        dist_to_table = center_rel_table.dot(stable_pose.r[2,:])

        # make sure not in collision and above z
        if not in_collision and dist_to_table > IzzyState.DELTA_Z:
            grasp_set.append(grasp_candidate)
        i = i+1
    
    return grasp_set
    
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    output_dir = sys.argv[2]

    # open config and read params
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    object_name = config['object_name']
    num_grasp_views = config['num_grasp_views']
    max_iter = config['max_iter']
    snapshot_rate = config['snapshot_rate']

    # open database and dataset
    database = db.Hdf5Database(database_filename, config)
    ds = database.dataset(dataset_name)

    # setup output directories and logging (TODO: make experiment wrapper class in future)
    experiment_id = 'single_grasp_experiment_%s' %(gen_experiment_id())
    experiment_dir = os.path.join(output_dir, experiment_id)
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    config['experiment_dir'] = experiment_dir
    experiment_log = os.path.join(experiment_dir, experiment_id +'.log')
    hdlr = logging.FileHandler(experiment_log)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logging.getLogger().addHandler(hdlr) 

    # copy over config file
    config_path, config_fileroot = os.path.split(config_filename)
    shutil.copyfile(config_filename, os.path.join(experiment_dir, config_fileroot))
    logging.info('RUNNING EXPERIMENT %s' %(experiment_id))

    # read the grasp metrics and features
    graspable = ds.graspable(object_name)
    grasps = ds.grasps(object_name)
    grasp_features = ds.grasp_features(object_name, grasps)
    grasp_metrics = ds.grasp_metrics(object_name, grasps)
    stable_poses = ds.stable_poses(object_name)
    stable_pose = stable_poses[config['stable_pose_index']]

    # compute the list of grasps to execute (TODO: update this section)
    grasps_to_execute = compute_grasp_set(ds, object_name, stable_pose, config['num_grasps_to_sample'], metric=config['grasp_metric'])

    # plot grasps
    T_obj_stp = stf.SimilarityTransform3D(pose=tfx.pose(stable_pose.r)) 
    object_mesh = graspable.mesh
    object_mesh_tf = object_mesh.transform(T_obj_stp)
    delta_view = 360.0 / num_grasp_views

    for i, grasp in enumerate(grasps_to_execute):
        mv.clf()
        object_mesh_tf.visualize(style='wireframe')
        MayaviVisualizer.mv_plot_grasp(grasp, T_obj_stp, alpha=1.5*config['alpha'], tube_radius=config['tube_radius'])

        for j in range(num_grasp_views):
            az = j * delta_view
            mv.view(az)
            figname = 'grasp_%d_view_%d.png' %(i, j)                
            #mv.savefig(os.path.join(experiment_dir, figname))

    # init hardware
    logging.info('Initializing hardware')
    camera = rs.RgbdSensor()
    ctrl = DexController()
    
    camera.reset()

    # Thompson sampling
    objective = MABSingleObjectObjective(graspable, stable_pose, ds, ctrl, camera, config)
    ts = das.ThompsonSampling(objective, grasps_to_execute)
    logging.info('Running Thompson sampling.')

    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        ]
    ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)
    ctrl.stop()
