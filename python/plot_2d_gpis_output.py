import math
import numpy as np
from scipy import linalg

import matplotlib as mp
import pylab as plt

import IPython

DEF_SIGMA = 2.7183
DEF_BETA = 0.1

def se_kernel(x, y, sigma = DEF_SIGMA):
    return np.exp(-np.linalg.norm(x - y) / 2*sigma)

def kernel_matrix(x, y, sigma = DEF_SIGMA):
    m = x.shape[0]
    n = y.shape[0]
    K = np.zeros([m, n])
    
    for i in range(m):
        for j in range(n):
            K[i,j] = se_kernel(x[i,:], y[j,:], sigma)

    return K

def grid_points(m, n):
    points = np.zeros([m*n, 2])
    index = 0
    for i in range(m):
        for j in range(n):
            points[index,:] = np.array([i, j])
            index = index + 1
    return points

def plot_tsdf(tsdf_filename):
    tsdf = np.genfromtxt(tsdf_filename, delimiter=',')
    plt.figure()
    plt.imshow(tsdf, cmap='Greys')
    plt.show()

def plot_active_set(tsdf_filename, active_inputs_filename):
    tsdf = np.genfromtxt(tsdf_filename, delimiter=',')
    inputs = np.genfromtxt(active_inputs_filename, delimiter=',')

    plt.figure()
    plt.imshow(tsdf, cmap='Greys')
    plt.scatter(inputs[:,0], inputs[:,1])
    plt.show()

def plot_predicted_tsdf(tsdf_filename, pred_filename, alpha_filename, active_inputs_filename):
    tsdf = np.genfromtxt(tsdf_filename, delimiter=',')
    pred_tsdf = np.genfromtxt(pred_filename, delimiter=',')
    alpha = np.genfromtxt(alpha_filename, delimiter=',')
    inputs = np.genfromtxt(active_inputs_filename, delimiter=',')
    inputs = inputs - np.ones(inputs.shape)

    (m, n) = tsdf.shape
    '''
    all_points = grid_points(m, n)
    
    K = kernel_matrix(all_points, inputs)
    pred_tsdf = K.dot(alpha)
    '''
    pred_tsdf_grid = np.reshape(pred_tsdf, [m, n])

    plt.figure()

    plt.subplot(1,2,1)
    plt.imshow(tsdf, cmap='Greys')
    plt.scatter(inputs[:,1], inputs[:,0])

    plt.subplot(1,2,2)
    plt.imshow(pred_tsdf_grid, cmap='Greys')

    plt.show()


if __name__ == '__main__':
#    plot_tsdf('data/plane.csv')
#    plot_active_set('data/plane.csv', 'inputs.csv')
    plot_predicted_tsdf('data/big_plane.csv', 'predictions.csv', 'alpha.csv', 'inputs.csv')
