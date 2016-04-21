import sys
import csv
import os
from fnmatch import fnmatch
import pickle
import IPython

def csv_to_dict_lists(csv_file_name, to_type=None):
    dict_lists = {}
    with open(csv_file_name, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key, val in row.items():
                if key not in dict_lists:
                    dict_lists[key] = []
                
                if to_type is not None:
                    val = to_type(val)
                dict_lists[key].append(val)
    
    return dict_lists

def parse_experiments(experiments_path, obj_mapping_file, output_file):
    #load obj mapping
    obj_ids = {}
    with open(obj_mapping_file, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            name, id = row['object'], row['id']
            if name in obj_ids:
                obj_ids[name].append(id)
            else:
                obj_ids[name] = [id]

    #data dict 
    #[object name][grasps][grasp id][grasp_output|actual_states|target_states][CSV HEADERS]
    #[object name][grasp_metric_results][CSV HEADERS]
    
    data_dict = {}
    for obj_name in obj_ids.keys():
        data_dict[obj_name] = {}
        
    #filling in data dict
    for obj_name in obj_ids.keys():
        for experiment_id in obj_ids[obj_name]:
            experiment_path = os.path.join(experiments_path,
                'single_grasp_experiment_{0}'.format(experiment_id))
            #fill in grasp_metric_results
            gmr_csv_file_name = os.path.join(experiment_path, 'grasp_metric_results.csv')
                
            gmr_dict_lists = csv_to_dict_lists(gmr_csv_file_name, to_type = float)

            if 'grasp_metric_results' in data_dict[obj_name]:
                for key in gmr_dict_lists.keys():
                    data_dict[obj_name]['grasp_metric_results'][key].extend(gmr_dict_lists[key])
            else:
                data_dict[obj_name]['grasp_metric_results'] = gmr_dict_lists
            
            #fill in data about trials of each grasp
            child_folders = next(os.walk(experiment_path))[1]
            grasp_folders = []
            for folder in child_folders:
                if fnmatch(folder, 'grasp_*'):
                    grasp_folders.append(folder)

            if 'grasps' not in data_dict[obj_name]:
                data_dict[obj_name]['grasps'] = {}
            for grasp_folder in grasp_folders:
                grasp_id = int(grasp_folder[6:])
                data_dict[obj_name]['grasps'][grasp_id] = {}
                data_dict[obj_name]['grasps'][grasp_id]['grasp_output'] = csv_to_dict_lists(os.path.join(
                    experiment_path, grasp_folder, 'grasp_output.csv'), to_type = float)
                data_dict[obj_name]['grasps'][grasp_id]['actual_states'] = csv_to_dict_lists(os.path.join(
                    experiment_path, grasp_folder, 'actual_states.csv'), to_type = float)
                data_dict[obj_name]['grasps'][grasp_id]['target_states'] = csv_to_dict_lists(os.path.join(
                    experiment_path, grasp_folder, 'target_states.csv'), to_type = float)

    #saving data dict
    with open(output_file, 'wb') as file: 
        pickle.dump(data_dict, file)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print "Invalid Parameters. Usage: [experiments folder] [obj to experiment csv] [path to output file]"
        exit(0)
        
    experiments_path = sys.argv[1]
    obj_mapping_file = sys.argv[2]
    output_file = sys.argv[3]
    
    parse_experiments(experiments_path, obj_mapping_file, output_file)