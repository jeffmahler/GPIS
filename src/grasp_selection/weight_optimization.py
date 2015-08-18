from os.path import exists

import IPython
import numpy as np

import database as db
import feature_functions as ff
import iterative_local_optimizers as ilo
from objectives import MinimizationObjective, StochasticGraspWeightObjective
import termination_conditions as tc

config = {
    'database_dir': '/mnt/terastation/shape_data/MASTER_DB_v1',
    'dataset': 'PriorsTrain',

    'window_steps': 13,

    'weight_proj_win': 1.0,
    'weight_grad_x': 0.0,
    'weight_grad_y': 0.0,
    'weight_curvature': 0.0,
    'weight_grasp_center': 0.0,
    'weight_grasp_axis': 0.0,
    'weight_grasp_angle': 0.0,
    'weight_gravity': 0.0,
}

def load_data(path, config):
    if exists(path):
        return np.load(path)

    training = db.Dataset(config['dataset'], config)
    all_grasps = []
    all_features = []

    for obj in training:
        feature_loader = ff.GraspableFeatureLoader(obj, training.name, config)
        obj_grasps = training.load_grasps(obj.key)
        obj_features = feature_loader.load_all_features(obj_grasps)
        all_grasps.extend(obj_grasps)
        all_features.extend(obj_features)

    if exists(path):
        return all_grasps, np.load(path)

    num_grasps = len(all_grasps)
    design_matrix = np.zeros(num_grasps, 2 * config['window_steps']**2)

    i = 0
    for grasp, feature in zip(all_grasps, all_features):
        w1 = feature.extractors_[0]
        w2 = feature.extractors_[1]

        proj1 = w1.extractors_[0]
        proj2 = w2.extractors_[0]

        design_matrix[i, :] = np.concat([proj1.phi, proj2.phi])
        i += 1

    np.save(path, design_matrix)
    return all_grasps, design_matrix

if __name__ == '__main__':
    np.random.seed(100)

    grasps, data = load_data('projection_window.npy', config)
    successes = [g.successes for g in grasps]
    failures = [g.failures for g in grasps]

    objective = objectives.MinimizationObjective(
        objectives.StochasticGraspWeightObjective(data, successes, failures))
    step_policy = ilo.DecayingStepPolicy(1)
    optimizer = ilo.UnconstrainedGradientAscent(objective, step_policy)
    start = np.random.rand(2 * config['window_steps']**2)
    result = optimizer.solve(termination_condition=tc.MaxIterTerminationCondition(100),
                             snapshot_rate=1, start_x=start)
    IPython.embed()
