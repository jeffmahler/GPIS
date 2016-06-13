import os
import argparse
import numpy as np
from sklearn.preprocessing import normalize
import IPython

def get_approx_normal(img):
    l = int(np.sqrt(img.shape[0]))
    c = l // 2
    res = 0.05/15
    grad = np.gradient(img.reshape(l,l))
    normal = normalize(np.array([grad[0][c, c]/res, grad[1][c, c]/res, -1]).reshape(1,-1))
    return normal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('projection_prefix', default='w1')
    parser.add_argument('input_path')
    parser.add_argument('output_path')
    args = parser.parse_args()

    for file in os.listdir(args.input_path):
        full_prefix = args.projection_prefix + '_projection_window_'
        if file.startswith(full_prefix):
            print "Processing " + file
            num = file[len(full_prefix):-4]
            data = np.load(os.path.join(args.input_path, file))['arr_0']
            
            normals = []
            for img in data:
                normals.append(get_approx_normal(img))
                
            normals = np.array(normals)
            np.savez(os.path.join(args.output_path, args.projection_prefix + '_approx_normals_' + num), normals)