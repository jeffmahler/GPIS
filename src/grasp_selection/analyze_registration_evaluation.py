import csv
import IPython
import numpy as np
import os
import sys
import tfx

import matplotlib.pyplot as plt

INVALID_OBJECTS = []

class ErrorStats:
    def __init__(self, error, tag):
        self.mean_error = np.mean(error, axis=0)
        self.med_error = np.median(error, axis=0)
        self.max_error = np.max(error, axis=0)
        self.std_error = np.std(error, axis=0)
        self.tag = tag # the name

    @staticmethod
    def stats_to_csv(stats_list, filename):
        root, ext = os.path.splitext(filename)
        if ext != '.csv':
            raise ValueError('Must save as a .csv file')
        f = open(filename, 'w')

        f.write('Tag,Mean,Median,Max,Std\n')
        for stats in stats_list:
            f.write('%s,%f,%f,%f,%f\n' %(stats.tag, stats.mean_error, stats.med_error,
                                         stats.max_error, stats.std_error))
        f.close()

if __name__ == '__main__':
    results_csv_filename = sys.argv[1]
    output_dir = sys.argv[2]
    f = open(results_csv_filename, 'r')
    csv_reader = csv.reader(f, delimiter=',', quotechar='|')

    headers = csv_reader.next()
    header_mappings = {}
    for i, header in enumerate(headers):
        header_mappings[header] = i

    translation_error = np.zeros([0,3])
    rotation_error = np.zeros([0,4])
    costs = np.zeros(0)
    durations = np.zeros(0)
    theta_error = np.zeros(0)

    for row in csv_reader:
        t = np.array([float(row[header_mappings['tx']]),
                      float(row[header_mappings['ty']]),
                      float(row[header_mappings['tz']])]).reshape([1,3])
        translation_error = np.r_[translation_error, t]

        q = np.array([float(row[header_mappings['qx']]),
                      float(row[header_mappings['qy']]),
                      float(row[header_mappings['qz']]),
                      float(row[header_mappings['qw']])]).reshape([1,4])
        rotation_error = np.r_[rotation_error, q]

        costs = np.r_[costs, float(row[header_mappings['cost']])]
        durations = np.r_[durations, float(row[header_mappings['duration']])]
        
        d_R = tfx.rotation(q).matrix
        d_theta = np.arccos(d_R[0,0])
        theta_error = np.r_[theta_error, d_theta]

    # histograms of translation error
    min_trans = -0.02
    max_trans = 0.02
    min_theta = 0.0
    max_theta = np.pi
    num_bins = 100
    font_size = 15
    line_width = 3.0
    dpi = 200

    max_cost = 0.04
    min_cost = 0.02
    delta_cost = 0.0005
    cost = max_cost
    all_errors = []
    eval_costs = []
    pct_data_vals = []
    med_trans_x_errors = []
    med_trans_y_errors = []
    med_trans_norm_errors = []
    med_theta_errors = []

    while cost > np.min(costs):
        print 'Processing cost', cost

        valid_indices = np.where(costs < cost)[0]

        tx_error_hist, tx_error_bins = np.histogram(translation_error[valid_indices,0], bins=num_bins, range=(min_trans, max_trans))
        ty_error_hist, ty_error_bins = np.histogram(translation_error[valid_indices,1], bins=num_bins, range=(min_trans, max_trans))
        tz_error_hist, tz_error_bins = np.histogram(translation_error[valid_indices,2], bins=num_bins, range=(min_trans, max_trans))
        theta_error_hist, theta_error_bins = np.histogram(theta_error[valid_indices], bins=num_bins, range=(min_theta, max_theta))
        width = (tx_error_bins[1] - tx_error_bins[0])
        theta_width = (theta_error_bins[1] - theta_error_bins[0])

        plt.figure()
        plt.bar(tx_error_bins[:-1], tx_error_hist, width=width, color='b')
        plt.xlabel('Error (m)', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)
        plt.xlim(min_trans, max_trans)
        plt.title('X Translation Error for Cost = %.4f' %(cost), fontsize=font_size)
        figname = os.path.join(output_dir, 'tx_error_hist_cost_%.4f.pdf' %(cost))
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.bar(ty_error_bins[:-1], ty_error_hist, width=width, color='b')
        plt.xlabel('Error (m)', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)
        plt.xlim(min_trans, max_trans)
        plt.title('Y Translation Error for Cost = %.4f' %(cost), fontsize=font_size)
        figname = os.path.join(output_dir, 'ty_error_hist_cost_%.4f.pdf' %(cost))
        plt.savefig(figname, dpi=dpi)

        plt.figure()
        plt.bar(theta_error_bins[:-1], theta_error_hist, width=theta_width, color='b')
        plt.xlabel('Error (radians)', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)
        plt.title('Rotation Error for Cost = %.4f' %(cost), fontsize=font_size)
        plt.xlim(min_theta, max_theta)
        figname = os.path.join(output_dir, 'theta_error_hist_cost_%.4f.pdf' %(cost))
        plt.savefig(figname, dpi=dpi)

        # compute statistics
        raw_trans_x_error = ErrorStats(translation_error[valid_indices,0], 'trans_x_raw_error_cost_%.4f' %(cost))
        sq_trans_x_error = ErrorStats(translation_error[valid_indices,0]**2, 'trans_x_sq_error_cost_%.4f' %(cost))
        abs_trans_x_error = ErrorStats(np.abs(translation_error[valid_indices,0]), 'trans_x_abs_error_cost_%.4f' %(cost))
        med_trans_x_errors.append(abs_trans_x_error.med_error)

        raw_trans_y_error = ErrorStats(translation_error[valid_indices,1], 'trans_y_raw_error_cost_%.4f' %(cost))
        sq_trans_y_error = ErrorStats(translation_error[valid_indices,1]**2, 'trans_y_sq_error_cost_%.4f' %(cost))
        abs_trans_y_error = ErrorStats(np.abs(translation_error[valid_indices,1]), 'trans_y_abs_error_cost_%.4f' %(cost))
        med_trans_y_errors.append(abs_trans_y_error.med_error)

        med_trans_norm_errors.append(np.median(np.linalg.norm(translation_error[valid_indices,:], axis=1)))

        raw_theta_error = ErrorStats(theta_error[valid_indices], 'theta_raw_error_cost_%.4f' %(cost))
        sq_theta_error = ErrorStats(theta_error[valid_indices]**2, 'theta_sq_error_cost_%.4f' %(cost))
        abs_theta_error = ErrorStats(np.abs(theta_error[valid_indices]), 'theta_abs_error_cost_%.4f' %(cost))
        med_theta_errors.append(abs_theta_error.med_error)

        all_errors.extend([raw_trans_x_error, sq_trans_x_error, abs_trans_x_error,
                           raw_trans_y_error, sq_trans_y_error, abs_trans_y_error,
                           raw_theta_error, sq_theta_error, abs_theta_error])

        # compute the percentage of data represented
        pct_data = float(valid_indices.shape[0]) / float(translation_error.shape[0])
        pct_data_vals.append(pct_data)
        eval_costs.append(cost)

        # update cost
        cost = cost - delta_cost

    # write all errors to a csv
    errors_filename = os.path.join(output_dir, 'registration_stats.csv')
    ErrorStats.stats_to_csv(all_errors, errors_filename)

    # plot percent data versus cost
    plt.figure()
    plt.plot(eval_costs, pct_data_vals, linewidth=line_width, color='b')
    plt.xlabel('Costs', fontsize=font_size)
    plt.ylabel('% Data', fontsize=font_size)
    plt.title('Cost vs Percent Data', fontsize=font_size)
    figname = os.path.join(output_dir, 'cost_vs_percent_data.pdf')
    plt.savefig(figname, dpi=dpi)

    # plot percent data versus cost
    plt.figure()
    plt.plot(eval_costs, med_trans_x_errors, linewidth=line_width, color='b')
    plt.xlabel('Costs', fontsize=font_size)
    plt.ylabel('Median Trans X Error (m)', fontsize=font_size)
    plt.title('Cost vs Median Error', fontsize=font_size)
    figname = os.path.join(output_dir, 'cost_vs_med_trans_x_error.pdf')
    plt.savefig(figname, dpi=dpi)

    plt.figure()
    plt.plot(eval_costs, med_trans_y_errors, linewidth=line_width, color='b')
    plt.xlabel('Costs', fontsize=font_size)
    plt.ylabel('Median Trans Y Error (m)', fontsize=font_size)
    plt.title('Cost vs Median Error', fontsize=font_size)
    figname = os.path.join(output_dir, 'cost_vs_med_trans_y_error.pdf')
    plt.savefig(figname, dpi=dpi)

    plt.figure()
    plt.plot(eval_costs, med_trans_norm_errors, linewidth=line_width, color='b')
    plt.xlabel('Costs', fontsize=font_size)
    plt.ylabel('Median Trans Norm Error (m)', fontsize=font_size)
    plt.title('Cost vs Median Error', fontsize=font_size)
    figname = os.path.join(output_dir, 'cost_vs_med_trans_norm_error.pdf')
    plt.savefig(figname, dpi=dpi)

    plt.figure()
    plt.plot(eval_costs, med_theta_errors, linewidth=line_width, color='b')
    plt.xlabel('Costs', fontsize=font_size)
    plt.ylabel('Median Theta Error (rad)', fontsize=font_size)
    plt.title('Cost vs Median Error', fontsize=font_size)
    figname = os.path.join(output_dir, 'cost_vs_med_theta_error.pdf')
    plt.savefig(figname, dpi=dpi)

    # compute worst case error in qx, qy for a sanity check
    print 'Worst qx error:', np.max(rotation_error[:,0])
    print 'Worst qy error:', np.max(rotation_error[:,1])
