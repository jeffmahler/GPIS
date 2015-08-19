import IPython
import matplotlib.pyplot as plt
import numpy as np

def plot_kernels(obj_key, k_mat, pfc_diff, all_neighbor_kernels=None, all_neighbor_pfc_diffs=None, neighbor_keys=None, font_size=10):
    """
    Generates a kernel plot for the given object grasp kernel matrix and PFC values, optionally including the plots for neighbors as well
    """
    max_len = 15
    labels = [obj_key[:max_len]]
    if neighbor_keys is not None:
        labels.extend([nk[:max_len] for nk in neighbor_keys])
    scatter_objs =[]
    plt.figure()
    colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(all_neighbor_pfc_diffs)))
    scatter_objs.append(plt.scatter(k_mat.ravel(), pfc_diff.ravel(), c='#eeeeff'))
    for i, (neighbor_pfc_diffs, neighbor_kernels) in enumerate(zip(all_neighbor_pfc_diffs, all_neighbor_kernels)):
        scatter_objs.append(plt.scatter(np.array(neighbor_kernels).ravel(), np.array(neighbor_pfc_diffs).ravel(), c=colors[i]))
    plt.xlabel('Kernel', fontsize=font_size)
    plt.ylabel('PFC Diff', fontsize=font_size)
    plt.title('Correlations', fontsize=font_size)
    plt.legend(scatter_objs, labels)

def plot_grasp_histogram(quality, num_bins=100, font_size=10):
    """
    Generates a plot of the histograms of grasps by probability of force closure
    """
    bin_edges = np.linspace(0, 1, num_bins+1)
    plt.figure()
    n, bins, patches = plt.hist(quality, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)

