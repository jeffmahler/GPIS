import csv
import logging
import IPython
import numpy as np
import os
import sys
import time

import database as db
import experiment_config as ec

from PIL import Image
import scipy.stats as ss

import matplotlib.pyplot as plt

OBJECT_TAG = 'object'
GRASP_ID_TAG = 'grasp_id'
TRIAL_TAG = 'trial'
LIFTED_COLOR_TAG = 'lifted_color'
HUMAN_LABEL_TAG = 'human_success'
HUMAN_PCT_TAG = 'human_robustness'

def repair_metrics(old_metrics):
    metrics = {}
    for metric_name, metric_val in old_metrics.iteritems():
        metric_key = metric_name
        if metric_name.find('vpc') != -1:
            continue
        if metric_name.find('vfc') != -1:
            continue
        if metric_name.find('ppc') != -1:
            metric_key = metric_name[:4] + metric_name[11:]
        if metric_name.find('lift_closure') != -1:
            metric_key = 'lift_closure'
        metrics[metric_key] = metric_val
    return metrics

if __name__ == '__main__':
    # read params
    logging.getLogger().setLevel(logging.INFO)
    config_filename = sys.argv[1]
    grasp_trial_csv_filename = sys.argv[2]
    output_dir = sys.argv[3]

    num_bins = 10
    font_size = 15
    dpi = 100
    max_len = 15

    # open csv file
    f = open(grasp_trial_csv_filename, 'r')
    csv_reader = csv.reader(f, delimiter=',')
    headers = csv_reader.next()

    # open database
    config = ec.ExperimentConfig(config_filename)
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    dataset_name = config['datasets'].keys()[0]
    database = db.Hdf5Database(database_filename, config)
    dataset = database.dataset(dataset_name) 
    gripper_name = config['gripper']
    
    obj_key = dataset.object_keys[0]
    grasps = dataset.grasps(obj_key, gripper=gripper_name)
    grasp_metrics = dataset.grasp_metrics(obj_key, grasps, gripper=gripper_name)
    metrics = repair_metrics(grasp_metrics[0])

    # open output csv file
    human_label_csv_filename = os.path.join(output_dir, 'analyzed_human_labeled_grasp_trial_results.csv')
    if os.path.exists(human_label_csv_filename):
        logging.warning('Output csv file already exists')

    human_label_headers = [HUMAN_PCT_TAG]
    human_label_headers.extend(headers)
    out_f = open(human_label_csv_filename, 'w')
    csv_writer = csv.DictWriter(out_f, human_label_headers)
    csv_writer.writeheader()

    # get indices of interesting tags
    object_ind = headers.index(OBJECT_TAG)
    grasp_id_ind = headers.index(GRASP_ID_TAG)
    trial_ind = headers.index(TRIAL_TAG)
    human_ind = headers.index(HUMAN_LABEL_TAG)

    # setup vars
    grasps_dict = {}
    grasp_metrics_dict = {}
    human_success_dict = {}
    human_count_dict = {}
    target_metrics = {}
    corr_metrics = {}
    obj_names = []
    grasp_ids = []

    for i, row in enumerate(csv_reader):
        # read ids
        obj = row[object_ind]
        grasp_id = int(row[grasp_id_ind])
        trial_num = int(row[trial_ind])
        human_label = int(row[human_ind])

        logging.info('Reading grasp %d trial %d from object %s' %(grasp_id, trial_num, obj))

        # read grasp metrics
        if obj not in grasps_dict.keys():
            grasps_dict[obj] = dataset.grasps(obj, gripper=gripper_name)
            grasp_metrics_dict[obj] = dataset.grasp_metrics(obj, grasps_dict[obj], gripper=gripper_name)
            human_success_dict[obj] = {}
            human_count_dict[obj] = {}
        metrics = grasp_metrics_dict[obj][grasp_id]
        metrics = repair_metrics(metrics)

        # aggregate successes and failures
        if grasp_id not in human_success_dict[obj]:
            human_success_dict[obj][grasp_id] = 0
            human_count_dict[obj][grasp_id] = 0
        human_success_dict[obj][grasp_id] += human_label
        human_count_dict[obj][grasp_id] += 1
        
        if human_count_dict[obj][grasp_id] == 10:
            metrics[HUMAN_PCT_TAG] = float(human_success_dict[obj][grasp_id]) / 10.0

            # log the metrics
            if HUMAN_PCT_TAG not in target_metrics.keys():
                target_metrics[HUMAN_PCT_TAG] = []
            target_metrics[HUMAN_PCT_TAG].append(metrics[HUMAN_PCT_TAG])
            obj_names.append(obj)
            grasp_ids.append(grasp_id)

            for metric_name, metric_val in metrics.iteritems():
                if metric_name.find('vfc') == -1 and metric_name.find(HUMAN_PCT_TAG) == -1:
                    if metric_name not in corr_metrics.keys():
                        corr_metrics[metric_name] = []
                    corr_metrics[metric_name].append(metric_val)

            # write to an output dictionary
            output_dict = {}
            output_dict.update(metrics)
            [output_dict.update({k:v}) for k, v in zip(headers, row)]
            csv_writer.writerow(output_dict)
            out_f.flush()

    obj_names = np.array(obj_names)
    grasp_ids = np.array(grasp_ids)

    # add target binary metrics
    thresh = 0.1
    while thresh <= 1.0:
        target_metrics['%s_thresh_%.2f'%(HUMAN_PCT_TAG, thresh)] = 1 * (np.array(target_metrics[HUMAN_PCT_TAG]) >= thresh)    
        thresh += 0.1

    # collect correlation statistics
    corr_coefs = {}
    for target_metric_name, target_metric_vals in target_metrics.iteritems():
        corr_coefs[target_metric_name] = {}
        corr_coefs[target_metric_name]['pr'] = []
        corr_coefs[target_metric_name]['sp'] = []

        for metric_name, corr_metric_vals in corr_metrics.iteritems():
            # compute correlation
            rho = np.corrcoef(target_metric_vals, corr_metric_vals)
            rho = rho[1,0]
            corr_coefs[target_metric_name]['pr'].append((metric_name, rho))
        
            nu, p = ss.spearmanr(target_metric_vals, corr_metric_vals)
            corr_coefs[target_metric_name]['sp'].append((metric_name, nu, p))

            # scatter plot
            if target_metric_name.find('thresh') == -1:
                plt.figure()
                plt.scatter(target_metric_vals, corr_metric_vals, s=50, c='b')
                plt.xlabel('%s' %(target_metric_name), fontsize=font_size)
                plt.ylabel('%s' %(metric_name[:max_len]), fontsize=font_size)
                plt.xlim(0,1)
                plt.ylim(0,np.max(corr_metric_vals))
                figname = os.path.join(output_dir, 'metric_%s_%s_scatter.pdf' %(metric_name[:max_len], target_metric_name))
                plt.savefig(figname, dpi=dpi)

        # sort
        corr_coefs[target_metric_name]['pr'].sort(key = lambda x: x[1], reverse=True)
        corr_coefs[target_metric_name]['sp'].sort(key = lambda x: x[1], reverse=True)

        logging.info('')
        logging.info('Metric %s' %(target_metric_name))

        logging.info('')
        logging.info('Ranking by Spearman Coefficient')
        for i, x in enumerate(corr_coefs[target_metric_name]['sp']):
            logging.info('Rank %d metric=%s rho=%.3f pvalue=%.3f' %(i, x[0], x[1], x[2]))

        logging.info('')
        logging.info('Ranking by Pearson Coefficient')
        for i, x in enumerate(corr_coefs[target_metric_name]['pr']):
            logging.info('Rank %d metric=%s rho=%.3f' %(i, x[0], x[1]))

    # collect example grasps
    target_metric_vals = np.array(target_metrics[HUMAN_PCT_TAG])
    physical_success_ind = np.where(target_metric_vals == np.max(target_metric_vals))[0]
    physical_failure_ind = np.where(target_metric_vals == np.min(target_metric_vals))[0]

    false_negative_examples = {}
    false_positive_examples = {}

    for metric_name, corr_metric_vals in corr_metrics.iteritems():
        false_negative_examples[metric_name] = []
        false_positive_examples[metric_name] = []

        corr_metric_vals = np.array(corr_metric_vals)
        corr_metric_successes = corr_metric_vals[physical_success_ind]
        corr_metric_failures = corr_metric_vals[physical_failure_ind]
        
        false_negative_inds = np.where(corr_metric_successes < np.percentile(corr_metric_successes, 25))[0]
        for false_negative_ind in false_negative_inds.tolist():
            false_negative_obj = obj_names[physical_success_ind][false_negative_ind]
            false_negative_grasp_id = grasp_ids[physical_success_ind][false_negative_ind]
            false_negative_metric = corr_metric_successes[false_negative_ind]
            false_negative_examples[metric_name].append((false_negative_obj, false_negative_grasp_id, false_negative_metric))

        false_positive_inds = np.where(corr_metric_failures > np.percentile(corr_metric_failures, 75))[0]
        for false_positive_ind in false_positive_inds.tolist():
            false_positive_obj = obj_names[physical_failure_ind][false_positive_ind]
            false_positive_grasp_id = grasp_ids[physical_failure_ind][false_positive_ind]
            false_positive_metric = corr_metric_failures[false_positive_ind]
            false_positive_examples[metric_name].append((false_positive_obj, false_positive_grasp_id, false_positive_metric))

    # plot histogram of grasp robustnesses
    logging.info('Saving grasp histogram')
    grasp_success_hist, grasp_success_bins = np.histogram(target_metrics[HUMAN_PCT_TAG],
                                                          bins=num_bins, range=(0,1))
    width = (grasp_success_bins[1] - grasp_success_bins[0])
    
    plt.figure()
    plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
    plt.title('Grasp Physical Robustness Histogram', fontsize=font_size)
    plt.xlabel('Physical Robustness', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    figname = os.path.join(output_dir, 'physical_robustness_histogram.pdf')
    plt.savefig(figname, dpi=dpi)

    # plot histograms by data split
    num_bins = 25
    for target_metric_name, target_values in target_metrics.iteritems():
        if target_metric_name.find('thresh') != -1:
            success_ind = np.where(target_values == 1)[0]
            failure_ind = np.where(target_values == 0)[0]

            for metric_name, corr_metric_vals in corr_metrics.iteritems():
                corr_metric_vals = np.array(corr_metric_vals)
                grasp_success_hist, grasp_success_bins = np.histogram(corr_metric_vals[success_ind],
                                                                      bins=num_bins, range=(0,np.max(corr_metric_vals)))
                grasp_failure_hist, grasp_failure_bins = np.histogram(corr_metric_vals[failure_ind],
                                                                      bins=num_bins, range=(0,np.max(corr_metric_vals)))
                width = (grasp_success_bins[1] - grasp_success_bins[0])

                plt.figure()
                
                plt.subplot(2,1,1)
                plt.bar(grasp_success_bins[:-1], grasp_success_hist, width=width, color='b')
                plt.gca().set_title('Metric %s Success Histogram' %(metric_name[:max_len]), fontsize=font_size)
                plt.ylabel('Num Grasps', fontsize=font_size)
                plt.xlim(0, np.max(corr_metric_vals))

                plt.subplot(2,1,2)
                plt.bar(grasp_failure_bins[:-1], grasp_failure_hist, width=width, color='r')
                plt.gca().set_title('Metric %s Failure Histogram' %(metric_name[:max_len]), fontsize=font_size)
                plt.xlabel('Metric', fontsize=font_size)
                plt.ylabel('Num Grasps', fontsize=font_size)
                plt.xlim(0, np.max(corr_metric_vals))
                
                figname = os.path.join(output_dir, 'metric_%s_%s_histogram.pdf' %(metric_name[:max_len], target_metric_name))
                plt.savefig(figname, dpi=dpi)        

    IPython.embed()
    database.close()
    exit(0)

