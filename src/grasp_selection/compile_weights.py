import argparse
import h5py
import IPython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

import experiment_config as ec

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('result_dir')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    config = ec.ExperimentConfig(args.config)
    result_dir = args.result_dir

    results = []
    for _, dirs, _ in os.walk(result_dir):
        for d in dirs:
            for root, _, files in os.walk(os.path.join(result_dir, d)):
                for f in files:
                    if f.endswith('.hdf5'):
                        results.append((root, f))

    for root, name in results:
        image_dir = os.path.split(root)[0]
        fname = os.path.join(root, name)
        logging.info('Loading %s', fname)
        with h5py.File(fname, 'r') as f:
            opt_weights = f['opt_weights'][()]
            orig_weights = f['orig_weights'][()]
            opt_kernel = f['opt_kernel'][()]
            orig_kernel = f['orig_kernel'][()]
            true_pfc = f['true_pfc'][()]
            opt_pfc = f['opt_pfc'][()]
            orig_pfc = f['orig_pfc'][()]
            loss = f['loss'][()]
            iters = f['iters'][()]
        logging.info('Loaded.')

        # plot weight vectors
        for weight, title in zip((orig_weights, opt_weights),
                                 ('Original', 'Optimized')):
            logging.info('Plotting %s', title)
            weight = weight.reshape((2, config['window_steps'], config['window_steps']))
            fig, axes = plt.subplots(nrows=1, ncols=2)
            for ax, w in zip(axes.flat, weight):
                im = ax.imshow(w, interpolation='none',
                               vmin=0, vmax=config['weight_initial'],
                               cmap=plt.cm.binary)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)
            ax.set_title(title)

            plt.savefig(os.path.join(image_dir, title.lower()))

        # plot error over time (should be decreasing)
        logging.info('Plotting loss')
        plt.figure()
        plt.plot(iters, loss, color='blue', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Error')
        plt.savefig(os.path.join(image_dir, 'loss'))

        # plot grasp qualities
        logging.info('Plotting grasp qualities')
        plt.figure()
        np.random.seed(100)
        total_num_grasps = opt_kernel.shape[0]
        view_num_grasps = 20
        some_grasps = np.random.choice(total_num_grasps, size=view_num_grasps, replace=False)

        ground_truth = true_pfc[some_grasps]
        predicted = opt_pfc[some_grasps]
        random_predicted = orig_pfc[some_grasps]

        r1 = plt.bar(np.arange(view_num_grasps)+0.00, random_predicted, 0.25, color='r')
        r2 = plt.bar(np.arange(view_num_grasps)+0.25, predicted, 0.25, color='y')
        r3 = plt.bar(np.arange(view_num_grasps)+0.50, ground_truth, 0.25, color='g')
        plt.legend((r1[0], r2[0], r3[0]), ('Random', 'Predicted', 'Actual'))
        plt.savefig(os.path.join(image_dir, 'pfc'))

        # IPython.embed()
