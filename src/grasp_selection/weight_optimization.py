import logging
from os.path import exists

import h5py
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

    'kernel_sigma': 1.0,
    'kernel_l': 1.0,
}

def load_data(path, config):
    precomputed = False # exists(path)

    training = db.Dataset(config['dataset'], config)
    all_grasps = []
    all_features = []

    for obj in training:
        obj_grasps = training.load_grasps(obj.key)
        all_grasps.extend(obj_grasps)

        if not precomputed:
            feature_loader = ff.GraspableFeatureLoader(obj, training.name, config)
            obj_features = feature_loader.load_all_features(obj_grasps)
            all_features.extend(obj_features)
            # break

    if precomputed:
        logging.info('Loading from %s', path)
        with h5py.File(path, 'r') as f:
            design_matrix = f['projection_window'][()]
        logging.info('Loaded.')
        return all_grasps, design_matrix

    num_grasps = len(all_grasps)
    design_matrix = np.zeros((num_grasps, 2 * config['window_steps']**2))

    i = 0
    for grasp, feature in zip(all_grasps, all_features):
        w1 = feature.extractors_[0]
        w2 = feature.extractors_[1]

        proj1 = w1.extractors_[0]
        proj2 = w2.extractors_[0]

        design_matrix[i, :] = np.concatenate([proj1.phi, proj2.phi])
        i += 1

    # logging.info('Saving to %s', path)
    # with h5py.File(path, 'w') as f:
    #     f['projection_window'] = design_matrix
    # logging.info('Saved.')
    return all_grasps, design_matrix

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    np.random.seed(100)

    grasps, data = load_data('grasp_features.hdf5', config)
    successes = np.array([g.successes for g in grasps])
    failures = np.array([g.failures for g in grasps])

    objective = MinimizationObjective(
        StochasticGraspWeightObjective(data, successes, failures, config))
    step_policy = ilo.DecayingStepPolicy(10, 1000)
    def positive_constraint(x):
        x[x < 0] = 0
        return x
    # optimizer = ilo.UnconstrainedGradientAscent(objective, step_policy)
    optimizer = ilo.ConstrainedGradientAscent(objective, step_policy,
                                              [positive_constraint])
    start = np.random.rand(2 * config['window_steps']**2)

    logging.info('Starting optimization.')
    result = optimizer.solve(termination_condition=tc.MaxIterTerminationCondition(5000),
                             snapshot_rate=100, start_x=start, true_x=None)
    IPython.embed()
