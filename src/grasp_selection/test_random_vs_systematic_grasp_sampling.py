"""
Main file for testing systematic vs random sampling in grasp evaluation
Author: Jeff Mahler
"""
import argparse
import logging
import pickle as pkl
import os
import random
import string
import time

import IPython
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import numpy as np
import scipy.stats

import antipodal_grasp_sampler as ags
import database as db
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import grasp_sampler as gs
import json_serialization as jsons
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import quality
import random_variables as rvs
import robust_grasp_quality as rgq
import termination_conditions as tc

pfc_mean_tag = 'pfc_mean'
pfc_true_tag = 'pfc_true'
pfc_rnd_tag = 'pfc_rnd'
pfc_sig_tag = 'pfc_sig'
pfc_tlr_tag = 'pfc_tlr'

def plot_window_2d(window, num_steps, title='', save=False):
    """Plot a 2D image of a window."""
    if num_steps % 2 == 1: # center window at contact
        indices = np.array(range(-(num_steps // 2), (num_steps // 2) + 1))
    else: # arbitrarily cut off upper bound to preserve window size
        indices = np.array(range(-(num_steps // 2), (num_steps // 2)))
    indices = np.array(range(num_steps)) # for easier debugging

    fig = plt.figure()
    plt.title(title)
    imgplot = plt.imshow(window, extent=[indices[0], indices[-1], indices[-1], indices[0]],
                         interpolation='none', cmap=plt.cm.binary)
    plt.colorbar()
    plt.clim(-0.004, 0.004) # fixing color range for visual comparisons

    if save and title:
        plt.tight_layout()
        plt.savefig(title.replace(' ', '-'), bbox_inches='tight')
        plt.close()

def plot_disc(disc):
    ax = plt.gca(projection = '3d')    
    num_radial = disc.shape[0]
    num_angles = disc.shape[1]
    for i in range(num_angles):
        for r in range(num_radial):
            theta = 2.0 * np.pi * float(i) / num_angles
            x = r * np.cos(theta) 
            y = r * np.sin(theta) 
            ax.scatter(x, y, disc[r,i], s=100) 

def plot_window_and_disc(disc, proj_window):
    num_radial = disc.shape[0]
    num_angles = disc.shape[1]
    plt.figure()
    plt.imshow(proj_window, interpolation='none', cmap=plt.cm.binary)
    for i in range(num_angles):
        for r in range(num_radial):
            theta = 2.0 * np.pi * float(i) / num_angles
            x = r * np.cos(theta) + 6
            y = r * np.sin(theta) + 6
            plt.scatter(x, y, c=u'g') 

    plt.show()

def prune_grasps_intersecting_table(grasps, obj, stp):
    coll_free_grasps = []
    coll_grasps = []
    n = stp.r[2,:]
    x0 = stp.x0
    for i, grasp in enumerate(grasps):
        g1, g2 = grasp.endpoints()
        t_max = n.dot(x0 - g1) / n.dot(g2 - g1)
        if (n.dot(g2 - g1) > 0 and t_max < 0) or (n.dot(g2 - g1) < 0 and t_max > 0):
            print 'Adding grasp', i
            coll_free_grasps.append(grasp)
        else:
            coll_grasps.append(grasp)
    return coll_free_grasps

def vis_grasp(graspable, grasp):
    points = np.array(graspable.mesh.vertices())
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    surf = mlab.triangular_mesh(x, y, z, graspable.mesh.triangles(), color=(0.5, 0.5, 0.5))#, scalars=qualities)

    color = (0, 0, 0)
    tube_radius = 0.005
    line_width = 1.0
    g1, g2 = grasp.endpoints()
    mlab.plot3d(*zip(g1, g2), color=color, line_width=line_width, tube_radius=tube_radius)

    mlab.draw()
    mlab.show()

def plot_grasp_metrics(grasp_metrics, line_width=4.0, font_size=15):
    plt.figure()

    plot_objs = []
    labels = []
    samples = sorted(grasp_metrics[pfc_rnd_tag].keys())
    num_colors = 1 + len(grasp_metrics[pfc_sig_tag].keys()) + len(grasp_metrics[pfc_tlr_tag].keys()) + 1
    colors = plt.get_cmap('hsv')(np.linspace(1.0/num_colors, 1.0, num_colors))

    p, = plt.plot([samples[0], samples[-1]], [grasp_metrics[pfc_true_tag], grasp_metrics[pfc_true_tag]],
                  linewidth=line_width, c=colors[0])
    label = 'True'
    plot_objs.append(p)
    labels.append(label)

    i = 0
    for scale, metric in grasp_metrics[pfc_sig_tag].iteritems():
        p, = plt.plot([samples[0], samples[-1]], [grasp_metrics[pfc_sig_tag][scale][0], grasp_metrics[pfc_sig_tag][scale][0]],
                      linewidth=line_width, c=colors[i+1])
        label = 'Sigma Point Scale = %d' %(scale)
        plot_objs.append(p)
        labels.append(label)
        i += 1

    i = 0
    for scale, metric in grasp_metrics[pfc_tlr_tag].iteritems():
        p, = plt.plot([samples[0], samples[-1]], [grasp_metrics[pfc_tlr_tag][scale][0], grasp_metrics[pfc_tlr_tag][scale][0]],
                      linewidth=line_width, c=colors[i+1+len(grasp_metrics[pfc_sig_tag].keys())])
        label = 'Taylor Scale = %d' %(scale)
        plot_objs.append(p)
        labels.append(label)
        i += 1

    pfc_rnd_mean = []
    pfc_rnd_std = []
    i = 0
    for num_samples, metric_list in grasp_metrics[pfc_rnd_tag].iteritems():
        metric_arr = np.array([m[0] for m in metric_list])
        pfc_rnd_mean.append((num_samples, np.mean(metric_arr)))
        pfc_rnd_std.append((num_samples, np.std(metric_arr)))
        i += 1

    pfc_rnd_mean.sort(key=lambda x:x[0])
    pfc_rnd_std.sort(key=lambda x:x[0])
    pfc_rnd_samples = np.array([a[0] for a in pfc_rnd_mean])
    pfc_rnd_mean = np.array([a[1] for a in pfc_rnd_mean])
    pfc_rnd_std = np.array([a[1] for a in pfc_rnd_std])
    p, _, _ = plt.errorbar(pfc_rnd_samples, pfc_rnd_mean, yerr=pfc_rnd_std, mew=line_width, capsize=10,
                           linewidth=line_width, c=colors[1+len(grasp_metrics[pfc_sig_tag].keys())+len(grasp_metrics[pfc_tlr_tag].keys())])
    label = 'IID Sampling'
    plot_objs.append(p)
    labels.append(label)

    plt.xlim(samples[0], samples[-1])
    plt.ylim(0,1)
    plt.xlabel('Num Samples', fontsize=font_size)
    plt.ylabel('Estimated PFC', fontsize=font_size)
    plt.title('Estimated PFC vs Samples', fontsize=font_size)

    params = {'legend.fontsize':5, 'legend.linewidth':1.0}
    plt.rcParams.update(params)
    plt.legend(plot_objs, labels, loc='best')
    #plt.setp(plt.gca().get_legend().get_texts(), fontsize='3')

def random_vs_systematic(obj, dataset, config):
    # sample grasps
    sample_start = time.clock()
    if config['grasp_sampler'] == 'antipodal':
        # antipodal sampling
        logging.info('Using antipodal grasp sampling')
        sampler = ags.AntipodalGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

        # pad with gaussian grasps
        num_grasps = len(grasps)
        min_num_grasps = config['min_num_grasps']
        if num_grasps < min_num_grasps:
            target_num_grasps = min_num_grasps - num_grasps
            gaussian_sampler = gs.GaussianGraspSampler(config)
            gaussian_grasps = gaussian_sampler.generate_grasps(
                obj, target_num_grasps=target_num_grasps, check_collisions=config['check_collisions'])
            grasps.extend(gaussian_grasps)
    else:
        # gaussian sampling
        logging.info('Using Gaussian grasp sampling')
        sampler = gs.GaussianGraspSampler(config)
        grasps = sampler.generate_grasps(
            obj, check_collisions=config['check_collisions'])

    sample_end = time.clock()
    sample_duration = sample_end - sample_start
    logging.info('Grasp candidate generation took %f sec' %(sample_duration))

    if not grasps or len(grasps) == 0:
        logging.info('Skipping %s' %(obj.key))
        return

    # compute quality
    grasp_metrics = {}
    graspable_rv = rvs.GraspableObjectPoseGaussianRV(obj, config)
    f_rv = rvs.FrictionGaussianRV(config['friction_coef'], config['sigma_mu'], config)#scipy.stats.norm(config['friction_coef'], config['sigma_mu'])

    for i, grasp in enumerate(grasps):
        logging.info('Evaluating quality for grasp %d' %(i))
        grasp_rv = rvs.ParallelJawGraspPoseGaussianRV(grasp, config)

        # setup storage
        grasp.grasp_id_ = i
        grasp_metrics[grasp.grasp_id] = {}
        grasp_metrics[grasp.grasp_id][pfc_rnd_tag] = {}
        grasp_metrics[grasp.grasp_id][pfc_sig_tag] = {}
        grasp_metrics[grasp.grasp_id][pfc_tlr_tag] = {}
        grasp_metrics[grasp.grasp_id][pfc_mean_tag] = quality.PointGraspMetrics3D.grasp_quality(grasp, obj, method='force_closure')
        grasp_metrics[grasp.grasp_id][pfc_true_tag] = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config,
                                                                                                 quality_metric='force_closure',
                                                                                                 num_samples=config['num_brute_force_samples'])

        # get pfc by sigma points
        for scale in config['pfc_sigma_scales']:
            sig_start = time.time()
            pfc = rgq.RobustGraspQuality.probability_success_sigma_pts(graspable_rv, grasp_rv, f_rv, config, scale=scale)
            sig_end = time.time()
            grasp_metrics[grasp.grasp_id][pfc_sig_tag][scale] = (pfc, sig_end - sig_start)

        # get pfc by sigma points
        for scale in config['pfc_taylor_scales']:
            tlr_start = time.time()
            pfc = rgq.RobustGraspQuality.probability_success_sigma_pts(graspable_rv, grasp_rv, f_rv, config, scale=scale)
            tlr_end = time.time()
            grasp_metrics[grasp.grasp_id][pfc_tlr_tag][scale] = (pfc, tlr_end - tlr_start)

        # get pfc by random sampling
        for num_samples in config['pfc_num_samples']:
            grasp_metrics[grasp.grasp_id][pfc_rnd_tag][num_samples] = []

            for j in range(config['pfc_num_trials']):
                rnd_start = time.time()
                pfc = rgq.RobustGraspQuality.probability_success(graspable_rv, grasp_rv, f_rv, config, quality_metric='force_closure',
                                                                 num_samples=num_samples)
                rnd_end = time.time()
                grasp_metrics[grasp.grasp_id][pfc_rnd_tag][num_samples].append((pfc, rnd_end - rnd_start))

        
    result_dir = 'results/pfc_estimation'
    dpi = 400
    for grasp in grasps:
        logging.info('Visualizing results for grasp %d' %(grasp.grasp_id))
        plot_grasp_metrics(grasp_metrics[grasp.grasp_id])
        figname = 'grasp_%d_pfc.png' %(grasp.grasp_id)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)

    IPython.embed()
    for grasp in grasps:
        vis_grasp(obj, grasp)

def redo_output(obj):
    grasp_filename = 'grasps.pkl'
    gm_filename = 'grasp_metrics.pkl'
    result_dir = 'results/pfc_estimation'
    dpi = 400

    import pickle as pkl
    f = open(os.path.join(result_dir, grasp_filename), 'r')
    grasps = pkl.load(f)
    f = open(os.path.join(result_dir, gm_filename), 'r')
    grasp_metrics = pkl.load(f)    

    IPython.embed()

    for grasp in grasps:
        logging.info('Visualizing results for grasp %d' %(grasp.grasp_id))
        plot_grasp_metrics(grasp_metrics[grasp.grasp_id])
        figname = 'grasp_%d_pfc.png' %(grasp.grasp_id)
        plt.savefig(os.path.join(result_dir, figname), dpi=dpi)

    for grasp in grasps:
        vis_grasp(obj, grasp)
    
if __name__ == '__main__':
    np.random.seed(100)
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    # read config file
    config = ec.ExperimentConfig(args.config)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)

    for dataset_name in config['datasets'].keys():
        dataset = database.dataset(dataset_name)

        # label each object in the dataset with grasps
        for obj in dataset:
            logging.info('Labelling object {} with grasps'.format(obj.key))
            if True:#try:
                #random_vs_systematic(obj, dataset, config)
                redo_output(obj)

            #except Exception as e:
            #    logging.warning('Failed to complete grasp labelling for object {}'.format(obj.key))
            #    logging.warning(str(e))

    database.close()
