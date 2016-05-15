import csv
import logging
import IPython
import os
import sys
import time

from PIL import Image

import matplotlib.pyplot as plt

OBJECT_TAG = 'object'
GRASP_ID_TAG = 'grasp_id'
TRIAL_TAG = 'trial'
LIFTED_COLOR_TAG = 'lifted_color'
LIFTED_DEPTH_TAG = 'lifted_depth'
OBJECT_POSE_TAG = 'object_pose'

LOST_ENTRIES = {
    'endstop_holder': ['klnozulimi'],
    'gearbox': ['rktrjsywob'],
    'mount1': ['qvgkoczzdc'],
    'mount2': ['lvcjgaqgta', 'nevnreqzlx'],
    'nozzle': ['bcixxgqovu'],
    'part1': ['yzgsvcghkw'],
    'part3': ['temundtuzy', 'osavibntcn', 'deudlxsvok', 'sitwscscfv', 'zjhxtfaiyn']
}

if __name__ == '__main__':
    # read params
    logging.getLogger().setLevel(logging.INFO)
    grasp_trial_csv_filename = sys.argv[1]
    data_dir = sys.argv[2]
    font_size = 15

    # open csv file
    f = open(grasp_trial_csv_filename, 'r')
    csv_reader = csv.reader(f, delimiter=',')
    headers = csv_reader.next()

    filename, ext = os.path.splitext(grasp_trial_csv_filename)
    out_filename = os.path.join(filename+'_repaired.csv')

    out_f = open(out_filename, 'w')
    csv_writer = csv.DictWriter(out_f, headers)
    csv_writer.writeheader()
    
    # write original contents to file
    for row in csv_reader:
        output_dict = dict(zip(headers, row))
        csv_writer.writerow(output_dict)
        out_f.flush()

    for obj, experiment_ids in LOST_ENTRIES.iteritems():
        logging.info('Repairing object %s' %(obj))
        for experiment_id in experiment_ids:
            logging.info('Repairing experiment %s' %(experiment_id))
            experiment_dir = os.path.join(data_dir, 'single_grasp_experiment_%s' %(experiment_id))
            for grasp_dir in os.listdir(experiment_dir):
                if grasp_dir.find('grasp') == 0 and grasp_dir.find('view') == -1:
                    grasp_dir = os.path.join(experiment_dir, grasp_dir)
                    grasp_id = int(grasp_dir[grasp_dir.rfind('_')+1:])
                    
                    # count trials to ensure validity
                    num_trials = 0
                    for trial_dir in os.listdir(grasp_dir):
                        if trial_dir.find('trial') != -1:
                            num_trials += 1

                    # count num trials
                    if num_trials == 10:
                        for trial_dir in os.listdir(grasp_dir):
                            if trial_dir.find('trial') != -1:
                                trial_dir = os.path.join(grasp_dir, trial_dir)
                                trial_num = int(trial_dir[trial_dir.rfind('_')+1:])

                                output_dict = dict(zip(headers, [0 for i in range(len(row))]))
                                output_dict[OBJECT_TAG] = obj
                                output_dict[GRASP_ID_TAG] = grasp_id
                                output_dict[TRIAL_TAG] = trial_num
                                output_dict[LIFTED_COLOR_TAG] = os.path.join(trial_dir, 'lifted_color.png')
                                output_dict[LIFTED_DEPTH_TAG] = os.path.join(trial_dir, 'lifted_depth.png')
                                output_dict[OBJECT_POSE_TAG] = os.path.join(trial_dir, 'T_obj_world.stf')                                
                                
                                csv_writer.writerow(output_dict)
                                out_f.flush()

    f.close()
    out_f.close()
    
