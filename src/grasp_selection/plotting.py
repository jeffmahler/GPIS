import matplotlib.pyplot as plt

def plot_kernels(k_mat, pfc_diff, all_neighbor_kernels=None, all_neighbor_pfc_diffs=None):
    """
    Generates a kernel plot for the given object grasp kernel matrix and PFC values, optionally including the plots for neighbors as well
    """
    labels = [obj.key[:15]] + map(lambda x: x[:15], neighbor_keys)
    scatter_objs =[]
    plt.figure()
    colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(neighbor_pfc_diffs)))
    scatter_objs.append(plt.scatter(k_vec.ravel(), pfc_vec.ravel(), c='#eeeeff'))
    for i, (neighbor_pfc_diffs, neighbor_kernels) in enumerate(zip(all_neighbor_pfc_diffs, all_neighbor_kernels)):
        scatter_objs.append(plt.scatter(neighbor_kernels.ravel(), neighbor_pfc_diffs.ravel(), c=colors[i]))
    plt.xlabel('Kernel', fontsize=15)
    plt.ylabel('PFC Diff', fontsize=15)
    plt.title('Correlations', fontsize=15)
    plt.legend(scatter_objs, labels)

def plot_grasp_histogram(quality, num_bins=100):
    """
    Generates a plot of the histograms of grasps by probability of force closure
    """
    bin_edges = np.linspace(0, 1, num_bins+1)
    plt.figure()
    n, bins, patches = plt.hist(quality, bin_edges)
    plt.xlabel('Probability of Success', fontsize=font_size)
    plt.ylabel('Num Grasps', fontsize=font_size)
    plt.title('Histogram of Grasps by Probability of Success', fontsize=font_size)

