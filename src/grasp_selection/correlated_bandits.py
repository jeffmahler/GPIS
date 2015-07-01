"""
Main file for correlated bandit experiments.

Author: Brian Hou
"""
import json
import logging
import os
import time

import IPython
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import kernels
import objectives
import pfc
import pr2_grasp_checker as pgc
import termination_conditions as tc

def best_values_over_time(result, plot=False):
    """Computes the expected values for the best arms of a BetaBernoulliModel at
    each time step.
    Params:
        result - AdaptiveSamplingResult instance, from a BetaBernoulliModel
    Returns:
        best_values - list of floats, expected values over time
    """
    best_values = [(m.alphas / m.betas)[m.best_pred_ind] for m in result.models]

    if plot:
        plt.figure()
        plt.plot(result.iters, best_values, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('P(Success)')

    return best_values

def save_results(results, file='bandit_results.npy'):
    """Saves results to a file."""
    asarray = np.array(results)
    np.save(file, asarray)

def label_correlated(obj, dest, config):
    """Label an object with grasps according to probability of force closure,
    using correlated bandits."""
    bandit_start = time.clock()

    # sample initial antipodal grasps
    sampler = ags.AntipodalGraspSampler(config)
    antipodal_start = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(obj, vis=False)
    antipodal_end = time.clock()
    antipodal_duration = antipodal_end - antipodal_start
    logging.info('Antipodal grasp candidate generation took %f sec' %(antipodal_duration))

    if not grasps:
        logging.info('Skipping %s' %(obj.key))
        return

    # bandit params
    max_iter = config['bandit_max_iter']
    confidence = config['bandit_confidence']
    snapshot_rate = config['bandit_snapshot_rate']
    tc_list = [
        tc.MaxIterTerminationCondition(max_iter),
        tc.ConfidenceTerminationCondition(confidence)
    ]

    # run bandits!
    graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

    candidates = []
    for grasp in grasps:
        grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
        candidates.append(pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config))

    def phi(rv):
        windows = rv.grasp.surface_window(obj, config['window_width'], config['window_steps'])
        clean_windows = windows # align and filter window
        window_vec = np.ravel(clean_windows)
        return window_vec
    nn = kernels.KDTree(phi=phi)
    kernel = kernels.SquaredExponentialKernel(
        sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)

    objective = objectives.RandomBinaryObjective()
    ts = das.CorrelatedThompsonSampling(
        objective, candidates, nn, kernel, tolerance=config['kernel_tolerance'])

    logging.info('Solving bandits!')
    ts_result = ts.solve(termination_condition=tc.OrTerminationCondition(tc_list), snapshot_rate=snapshot_rate)

    object_grasps = [c.grasp for c in ts_result.best_candidates]
    grasp_qualities = list(ts_result.best_pred_means)

    bandit_stop = time.clock()
    logging.info('Bandits took %f sec' %(bandit_stop - bandit_start))

    # get rotated, translated versions of grasps
    delay = 0
    pr2_grasps = []
    pr2_grasp_qualities = []
    theta_res = config['grasp_theta_res'] * np.pi
    grasp_checker = pgc.OpenRaveGraspChecker(view=config['vis_grasps'])
    i = 0
    if config['vis_grasps']:
        delay = config['vis_delay']

    for grasp in object_grasps:
        rotated_grasps = grasp.transform(obj.tf, theta_res)
        rotated_grasps = grasp_checker.prune_grasps_in_collision(obj, rotated_grasps, auto_step=True, close_fingers=False, delay=delay)
        pr2_grasps.extend(rotated_grasps)
        pr2_grasp_qualities.extend([grasp_qualities[i]] * len(rotated_grasps))
        i = i+1

    logging.info('Num grasps: %d' %(len(pr2_grasps)))

    grasp_filename = os.path.join(dest, obj.key + '.json')
    with open(grasp_filename, 'w') as f:
        json.dump([g.to_json(quality=q) for g, q in zip(pr2_grasps, pr2_grasp_qualities)], f,
                  sort_keys=True, indent=4, separators=(',', ': '))

    return best_values_over_time(ts_result)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', default='cfg/correlated.yaml')
    parser.add_argument('output_dest', default='out/')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    chunk = db.Chunk(config)

    # make output directory
    dest = os.path.join(args.output_dest, chunk.name)
    try:
        os.makedirs(dest)
    except os.error:
        pass

    # loop through objects, labelling each
    results = []
    for obj in chunk:
        logging.info('Labelling object {}'.format(obj.key))
        values_over_time = label_correlated(obj, dest, config)
        if values_over_time is None:
            continue # no grasps to run bandits on for this object
        results.append(values_over_time)

    save_results(results)
