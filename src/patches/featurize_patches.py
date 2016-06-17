'''
Generate new features of patches data. 
Code partially borrowed from original analyze_patch_planirity.py
Author: Jacky Liang
'''

import os
import argparse
import numpy as np
import logging
import IPython
import yaml

from sklearn.preprocessing import normalize
from patches_data_loader import PatchesDataLoader
    
def _friction_cones(args, params):
    surface_normal_prefix = ["surface_normals_"]
    moment_arms_prefix = ["moment_arms_"]
    
    surface_normals_file_num_pairs = PatchesDataLoader.get_patch_files_and_nums(args.input_path, surface_normal_prefix)
    moment_arms_file_num_pairs = PatchesDataLoader.get_patch_files_and_nums(args.input_path, moment_arms_prefix)
    
    n_surface_normals_files = len(surface_normals_file_num_pairs)
    n_moment_arms_files = len(moment_arms_file_num_pairs)
    
    if n_surface_normals_files == 0 or n_moment_arms_files == 0:
        logging.error("Can't find sufficient surface normals and moment arms files necessary for featurizing in_out_cones!")
        return
    
    surface_normals_file_num_pairs.sort(key=lambda x: x[1])
    moment_arms_file_num_pairs.sort(key=lambda x: x[1])
    
    beta = np.arctan(params['friction_coef'])
    
    i, j = 0, 0
    while i < n_surface_normals_files and j < n_moment_arms_files:
        surface_normals_num = surface_normals_file_num_pairs[i][1]
        moment_arms_num = moment_arms_file_num_pairs[j][1]
        
        if surface_normals_num < moment_arms_num:
            i += 1
            logging.debug("No matching moment arm file found for num {0}".format(surface_normals_num))
            continue
        if moment_arms_num < surface_normals_num:
            j += 1
            logging.debug("No matching surface normal file found for num {0}".format(surface_normals_num))
            continue
            
        surface_normal_file = surface_normals_file_num_pairs[i][0]
        moment_arms_file = moment_arms_file_num_pairs[i][0]
        
        logging.info("Processing {0} and {1} for in out cones".format(surface_normal_file, moment_arms_file))
            
        surface_normal_data = np.load(os.path.join(args.input_path, surface_normal_file))['arr_0']
        contacts_data = np.load(os.path.join(args.input_path, moment_arms_file))['arr_0']

        # check lines within friction cone
        contact_lines = contacts_data[:,3:] - contacts_data[:,:3]
        ip_1 = abs(np.sum(surface_normal_data[:,:,0] * contact_lines, axis=1))
        ip_2 = abs(np.sum(surface_normal_data[:,:,1] * contact_lines, axis=1))
        cl_norms = np.linalg.norm(contact_lines, axis=1)

        # Dan: need this little hack, otherwise file 83 messes up for pfc 0.05.
        alphas_1 = np.arccos(np.clip(ip_1 / cl_norms, float('-inf'), 1-1e-9))
        alphas_2 = np.arccos(np.clip(ip_2 / cl_norms, float('-inf'), 1-1e-9))
        
        if params['type'] == "in_out":
            in_cone_inds = np.where((alphas_1 <= beta) & (alphas_2 <= beta)  & (np.isfinite(alphas_1)) & (np.isfinite(alphas_2)))[0]
            output_data = np.array([1 if k in in_cone_inds else 0 for k in range(len(contacts_data))])
        elif params["type"] == "angle":
            output_data = np.max(np.array([alphas_1, alphas_2]), axis=0)
        
        output_filename = "{0}_cones_{1}_{2}".format(params['type'], params['suffix'], moment_arms_num)
        logging.info("Saving {0}".format(output_filename))
        np.savez(os.path.join(args.output_path, output_filename), output_data)
        
        i += 1
        j += 1

def featurize_friction_cones_gen(params):
    
    def featurize_friction_cones(args):
        return _friction_cones(args, params)
        
    return featurize_friction_cones
        
def _approx_normals(wi, num, data, params=None):
    def get_approx_normal(img):
        l = int(np.sqrt(img.shape[0]))
        c = l // 2
        res = 0.05/15
        grad = np.gradient(img.reshape(l,l))
        normal = normalize(np.array([grad[0][c, c]/res, grad[1][c, c]/res, -1]).reshape(1,-1))
        return normal
        
    normals = []
    for img in data:
        normals.append(get_approx_normal(img))
        
    normals = np.array(normals)
    output_filename = wi + '_approx_normals_' + num
    return output_filename, normals

def _planarities(wi, num, data, params=None):
    dim = int(np.sqrt(data.shape[1]))
    x_ind, y_ind = np.meshgrid(np.arange(dim), np.arange(dim))
    A = np.c_[np.ravel(x_ind), np.ravel(y_ind), np.ones(dim**2)]
    b = data.T

    w, _, _, _ = np.linalg.lstsq(A, b)
    pred_w = A.dot(w)

    pred_w_error = pred_w - b
    mean_sq_dev_planarity_w = np.mean(pred_w_error**2, axis=0)
    
    output_filename = wi + '_sq_planarities_' + num
    return output_filename, mean_sq_dev_planarity_w

def _crop_windows(wi, num, data, params=None):
    def crop(img, dim):
        big_dim = int(np.sqrt(img.shape[0]))
        img = img.reshape(big_dim, big_dim)
        mid = big_dim // 2
        delta = dim // 2

        small_img = img[mid - delta : mid + delta + 1, mid - delta : mid + delta + 1]

        return small_img.reshape(dim*dim)

    dim = params['crop_dim']
    cropped = np.array([crop(data[i,:], dim) for i in range(len(data))])
    
    output_filename = "{0}_crop_{1}_{2}".format(wi, dim, num)
    return output_filename, cropped
    
def featurize_projections_gen(featurize, params=None):

    def featurize_projections(args):
        ws = ("w1", "w2")
        prefixes = [w + '_projection_window_' for w in ws]
        for filename, num in PatchesDataLoader.get_patch_files_and_nums(args.input_path, prefixes):
            logging.info("Processing " + filename)
            
            wi = filename[:2]
            data = np.load(os.path.join(args.input_path, filename))['arr_0']
            output_filename, output_data = featurize(wi, num, data, params=params)
            
            logging.info("Saving " + output_filename)
            np.savez(os.path.join(args.output_path, output_filename), output_data)
            
    return featurize_projections
    
FEATURIZER_MAP = {
    "planarity" : featurize_projections_gen(_planarities),
    "in_out_cone_5" : featurize_friction_cones_gen({'type':'in_out','friction_coef':0.5, 'suffix':'5'}),
    "angle_cone_5" : featurize_friction_cones_gen({'type':'angle','friction_coef':0.5, 'suffix':'5'}),
    "crop_3" : featurize_projections_gen(_crop_windows, {'crop_dim':3}),
    "crop_5" : featurize_projections_gen(_crop_windows, {'crop_dim':5}),
    "approx_normals": featurize_projections_gen(_approx_normals)
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(logging.INFO)
    
    with open(args.config) as config_file:
        to_featurize_config = yaml.safe_load(config_file)
        
    for feature_name, to_featurize in to_featurize_config.items():
        if to_featurize:
            FEATURIZER_MAP[feature_name](args)