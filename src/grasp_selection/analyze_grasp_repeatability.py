import IPython
import logging
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import scipy.stats as ss

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    result_dir = sys.argv[1]
    output_dir = sys.argv[2]

    success_metric_key = 'p_grasp_success'
    lift_metric_key = 'p_lift_success'
    grasp_id_name = 'grasp_id'
    grasp_result_file_root = 'grasp_metric_results.csv'
    font_size = 15
    marker_size = 100
    eps = 0
    dpi = 400
    a = 13

    all_grasp_metric_data = {}

    # extract all grasp metric data
    for experiment_dir in os.listdir(result_dir):
        grasp_result_filename = os.path.join(result_dir, experiment_dir, grasp_result_file_root)
        logging.info('Reading result %s' %(grasp_result_filename))
        try:
            grasp_metric_data = np.genfromtxt(grasp_result_filename, dtype=float, delimiter=',', names=True)
        except:
            grasp_metric_data = np.genfromtxt(grasp_result_filename, dtype=float, delimiter=';', names=True)            

        column_names = grasp_metric_data.dtype.names
        for metric_name in column_names:
            if metric_name != grasp_id_name and metric_name != 'ferrari_canny_l1' and \
                    metric_name != 'force_closure' and metric_name.find('vfc') == -1 and \
                    metric_name.find('vpc') == -1:
                metric_key = metric_name
                if metric_name.find('ppc') != -1:
                    metric_key = metric_name[:4] + metric_name[11:]
                if metric_name.find('lift_closure') != -1:
                    metric_key = 'lift_closure'

                # add to dict if nonexistent
                if metric_key not in all_grasp_metric_data.keys():
                    all_grasp_metric_data[metric_key] = []

                # add metrics
                all_grasp_metric_data[metric_key].extend(grasp_metric_data[metric_name].tolist())

    # analyze correlations
    target_metrics = [success_metric_key, lift_metric_key]
    for target_metric in target_metrics:
        target_values = all_grasp_metric_data[target_metric]
        #target_values = [1 * (v > 0.5) for v in target_values]

        # check against all other metrics
        pr_corr_coefs = []
        sp_corr_coefs = []
        for metric_key in all_grasp_metric_data.keys():
            if metric_key != target_metric:
                corr_values = all_grasp_metric_data[metric_key]

                # compute correlation
                rho = np.corrcoef(target_values, corr_values)
                rho = rho[1,0]
                pr_corr_coefs.append((metric_key, rho))

                nu = ss.spearmanr(target_values, corr_values)
                sp_corr_coefs.append((metric_key, nu[0]))

                # scatter data
                plt.figure()
                plt.scatter(corr_values, target_values, color='b', s=marker_size)
                plt.xlim(-eps, np.max(np.array(corr_values))+eps)
                plt.ylim(-eps, 1+eps)
                plt.xlabel(metric_key[:a], fontsize=font_size)                
                plt.ylabel(target_metric, fontsize=font_size)
                plt.title('Correlation = %.3f' %(nu[0]), fontsize=font_size)

                figname = os.path.join(output_dir, '%s_vs_%s.pdf' %(target_metric, metric_key[:a]))
                plt.savefig(figname, dpi=dpi)

        # sort corr coefs and store
        pr_corr_coefs.sort(key = lambda x: x[1], reverse=True)
        sp_corr_coefs.sort(key = lambda x: x[1], reverse=True)
        IPython.embed()

                
