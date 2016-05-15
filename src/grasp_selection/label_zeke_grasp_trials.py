import csv
import logging
import IPython
import os
import sys
import time

import database as db
import experiment_config as ec

from PIL import Image

import matplotlib.pyplot as plt

OBJECT_TAG = 'object'
GRASP_ID_TAG = 'grasp_id'
TRIAL_TAG = 'trial'
LIFTED_COLOR_TAG = 'lifted_color'
HUMAN_LABEL_TAG = 'human_success'

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
    font_size = 15

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
    human_label_csv_filename = os.path.join(output_dir, 'human_labeled_grasp_trial_results.csv')
    if os.path.exists(human_label_csv_filename):
        logging.warning('Output csv file already exists')

    human_label_headers = [HUMAN_LABEL_TAG]
    human_label_headers.extend(metrics.keys())
    human_label_headers.extend(headers)
    out_f = open(human_label_csv_filename, 'w')
    csv_writer = csv.DictWriter(out_f, human_label_headers)
    csv_writer.writeheader()

    # get indices of interesting tags
    object_ind = headers.index(OBJECT_TAG)
    grasp_id_ind = headers.index(GRASP_ID_TAG)
    trial_ind = headers.index(TRIAL_TAG)
    image_ind = headers.index(LIFTED_COLOR_TAG)

    # setup vars
    grasps_dict = {}
    grasp_metrics_dict = {}
    grasp_success = 0
    plt.figure()

    for i, row in enumerate(csv_reader):
        # read ids
        obj = row[object_ind]
        grasp_id = int(row[grasp_id_ind])
        trial_num = int(row[trial_ind])

        # read grasp metrics
        if obj not in grasps_dict.keys():
            grasps_dict[obj] = dataset.grasps(obj, gripper=gripper_name)
            grasp_metrics_dict[obj] = dataset.grasp_metrics(obj, grasps_dict[obj], gripper=gripper_name)
        metrics = grasp_metrics_dict[obj][grasp_id]
        metrics = repair_metrics(metrics)
            
        # open image
        image_filename = row[image_ind]
        lift_image = Image.open(image_filename)

        plt.clf()
        plt.imshow(lift_image)
        plt.title('Object %s grasp %d trial %d' %(obj, grasp_id, trial_num))
        plt.axis('off')
        plt.ioff()
        plt.show(block=False)

        human_input = raw_input('Did the grasp succeed? [y/n] ')
        while human_input.lower() != 'n' and human_input.lower() != 'y':
            logging.info('Did not understand input. Please answer \'y\' or \'n\'')
            human_input = raw_input('Did the grasp succeed? [y/n] ')

        if human_input.lower() == 'y':
            grasp_success = 1
            logging.info('Recorded success')
        else:
            grasp_success = 0
            logging.info('Recorded failure')

        output_dict = {}
        output_dict.update(metrics)
        [output_dict.update({k:v}) for k, v in zip(headers, row)]
        output_dict[HUMAN_LABEL_TAG] = grasp_success
        csv_writer.writerow(output_dict)
        out_f.flush()

    database.close()
    exit(0)
