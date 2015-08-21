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
    'partial_gradient_size': 200,

    'plot': True,
}

def load_data(path, config):
    precomputed = exists(path)

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

    logging.info('Saving to %s', path)
    with h5py.File(path, 'w') as f:
        f['projection_window'] = design_matrix
    logging.info('Saved.')
    return all_grasps, design_matrix

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    np.random.seed(100)

    grasps, data = load_data('grasp_features.hdf5', config)
    successes = np.array([g.successes for g in grasps]) - 1 # subtract alpha0
    failures = np.array([g.failures for g in grasps]) - 1 # subtract beta0
    weight_range = (25, 55)

    loss = StochasticGraspWeightObjective(data, successes, failures, config)
    objective = MinimizationObjective(loss)
    step_policy = ilo.LogStepPolicy(1000, 150)
    def positive_constraint(x):
        x[x < 0] = 0
        return x
    # optimizer = ilo.UnconstrainedGradientAscent(objective, step_policy)
    optimizer = ilo.ConstrainedGradientAscent(objective, step_policy,
                                              [positive_constraint])
    start = np.random.uniform(*weight_range, size=2 * config['window_steps']**2)
    start = 1e2 * np.ones(2 * config['window_steps']**2)

    logging.info('Starting optimization.')
    result = optimizer.solve(termination_condition=tc.MaxIterTerminationCondition(5000),
                             snapshot_rate=1000, start_x=start, true_x=None)

    proj_win_weight = result.best_x
    max_weight = np.max(proj_win_weight)
    opt_weights = proj_win_weight.reshape((2, config['window_steps'], config['window_steps']))
    rand_weights = start.reshape((2, config['window_steps'], config['window_steps']))

    logging.info('Loss: %f to %f, delta=%f', loss(start), loss(result.best_x), np.linalg.norm(start - result.best_x))

    # debugging stuff

    def min_and_max(arr):
        return np.min(arr), np.max(arr)

    ground_truth = loss.mu_
    logging.info('Actual pfc range: %s', min_and_max(ground_truth))

    random_kernel = loss.kernel(start)
    random_kernel_matrix = random_kernel.matrix(loss.X_)
    random_alpha = 1 + np.dot(random_kernel_matrix, loss.S_) - loss.S_
    random_beta = 1 + np.dot(random_kernel_matrix, loss.F_) - loss.F_
    random_predicted = random_alpha / (random_alpha + random_beta)
    logging.info('Initial random pfc range: %s', min_and_max(random_predicted))

    kernel = loss.kernel(proj_win_weight)
    kernel_matrix = kernel.matrix(loss.X_)
    alpha = 1 + np.dot(kernel_matrix, loss.S_) - loss.S_
    beta = 1 + np.dot(kernel_matrix, loss.F_) - loss.F_
    predicted = alpha / (alpha + beta)
    logging.info('Predicted pfc range: %s', min_and_max(predicted))

    if config['plot']:
        import matplotlib.pyplot as plt

        # plot weight vectors
        for weight in (rand_weights, opt_weights):
            fig, axes = plt.subplots(nrows=1, ncols=2)
            for ax, w in zip(axes.flat, weight):
                im = ax.imshow(w, interpolation='none',
                               vmin=0, vmax=100,
                               cmap=plt.cm.binary)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

        # plot error over time (should be decreasing)
        plt.figure()
        # negate bc vals_f is from minimization objective, normalize by num_grasps
        loss_over_time = -np.array(result.vals_f) / len(grasps)
        plt.plot(result.iters, loss_over_time, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Error')

        plt.figure()
        r1 = plt.bar(np.arange(250)+0.00, random_predicted, 0.25, color='r')
        r2 = plt.bar(np.arange(250)+0.25, predicted, 0.25, color='y')
        r3 = plt.bar(np.arange(250)+0.50, ground_truth, 0.25, color='g')
        plt.legend((r1[0], r2[0], r3[0]), ('Random', 'Predicted', 'Actual'))
        plt.xlim((0, 20))

        plt.show(block=False)

    IPython.embed()
