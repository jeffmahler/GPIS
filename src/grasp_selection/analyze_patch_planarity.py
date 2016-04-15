"""
Script to analyze the distribution of PFC versus parameters such as planarity, angle with the friction cone, etc
"""
import IPython
import logging
import numpy as np
import os
import sys

import matplotlib.pyplot as plt

class ErrorStats:
    def __init__(self, pred_error, tag):
        self.mse = np.mean(pred_error**2)
        self.mae = np.mean(np.abs(pred_error))
        self.tag = tag # the name

    @staticmethod
    def stats_to_csv(stats_list, filename):
        root, ext = os.path.splitext(filename)
        if ext != '.csv':
            raise ValueError('Must save as a .csv file')
        f = open(filename, 'w')

        f.write('Tag,MSE,MAE\n')
        for stats in stats_list:
            f.write('%s,%f,%f\n' %(stats.tag, stats.mse, stats.mae))
        f.close()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    patch_data_dir = sys.argv[1]
    num_batches = int(sys.argv[2])
    output_dir = sys.argv[3]

    # read in the pfc values, surface normals values, and grasp parameters
    friction_coef = 0.5
    beta = np.arctan(friction_coef)
    num_bins = 100
    line_width = 2
    font_size = 20
    dpi = 400
    crop_dim = 3
    eps = 0.0

    pfc_tag = 'pfc'
    vfc_tag = 'vfc'

    com_tag = 'com'
    contact_tag = 'moment_arms'
    grasp_axis_tag = 'patch_orientation'
    grasp_center_tag = 'center'
    surface_normal_tag = 'surface_normals'
    projection_tag = 'projection_window'
    
    metric_filename_map = {}
    feature_filename_map = {}
    all_error_stats = []

    data_filenames = os.listdir(patch_data_dir)

    # read in the relevant filenames for each
    for filename in data_filenames:
        # parse file root
        filename_root, filename_ext = os.path.splitext(filename)
        filename_root = filename_root[:-3]

        # switch filename category based on name
        if filename.find(pfc_tag) != -1:# or filename.find(vfc_tag) != -1:
            if filename_root not in metric_filename_map.keys():
                metric_filename_map[filename_root] = []
            metric_filename_map[filename_root].append(filename)

        # switch filename category based on name
        if filename.find(com_tag) != -1 or filename.find(contact_tag) != -1 or \
                filename.find(grasp_axis_tag) != -1 or filename.find(grasp_center_tag) != -1 or \
                filename.find(projection_tag) != -1 or filename.find(surface_normal_tag) != -1:
            if filename_root not in feature_filename_map.keys():
                feature_filename_map[filename_root] = []
            feature_filename_map[filename_root].append(filename)
    
    # sort each list
    for metric, filename_list in metric_filename_map.iteritems():
        filename_list.sort()
    for feature, filename_list in feature_filename_map.iteritems():
        filename_list.sort()

    # histogram of quality for mean force closure versus mean not force closure
    for metric in metric_filename_map.keys():
        all_in_cone_hist = np.zeros(num_bins)
        all_out_of_cone_hist = np.zeros(num_bins)
        all_metrics = np.zeros(0)
        all_in_cone_metrics = np.zeros(0)
        all_out_of_cone_metrics = np.zeros(0)
        all_alphas = np.zeros(0)
        all_mean_sq_dev_planarity_w1 = np.zeros(0)
        all_mean_sq_dev_planarity_w2 = np.zeros(0)
        all_mean_sq_dev_planarity_w1_crop = np.zeros(0)
        all_mean_sq_dev_planarity_w2_crop = np.zeros(0)

        for i, metric_filename in enumerate(metric_filename_map[metric]):
            if i >= num_batches:
                break

            grasp_axis_filename = feature_filename_map[grasp_axis_tag][i]
            surface_normal_filename = feature_filename_map[surface_normal_tag][i]
            contacts_filename = feature_filename_map[contact_tag][i]
            proj_window_1_filename = feature_filename_map['w1_'+projection_tag][i]
            proj_window_2_filename = feature_filename_map['w2_'+projection_tag][i]

            # load in the quality metric data
            logging.info('Loading data batch %d for metric %s' %(i, metric))
            metric_data = np.load(os.path.join(patch_data_dir, metric_filename))['arr_0']
            grasp_axis_data = np.load(os.path.join(patch_data_dir, grasp_axis_filename))['arr_0']
            surface_normal_data = np.load(os.path.join(patch_data_dir, surface_normal_filename))['arr_0']
            contacts_data = np.load(os.path.join(patch_data_dir, contacts_filename))['arr_0']
            proj_window_1_data = np.load(os.path.join(patch_data_dir, proj_window_1_filename))['arr_0']
            proj_window_2_data = np.load(os.path.join(patch_data_dir, proj_window_2_filename))['arr_0']

            all_metrics = np.r_[all_metrics, metric_data]

            # check lines within friction cone
            contact_lines = contacts_data[:,3:] - contacts_data[:,:3]
            ip_1 = abs(np.sum(surface_normal_data[:,:,0] * contact_lines, axis=1))
            ip_2 = abs(np.sum(surface_normal_data[:,:,1] * contact_lines, axis=1))
            cl_norms = np.linalg.norm(contact_lines, axis=1)
            alphas_1 = np.arccos(ip_1 / cl_norms)
            alphas_2 = np.arccos(ip_2 / cl_norms)
            out_of_cone_inds = np.where(((alphas_1 > beta) | (alphas_2 > beta) | (np.isnan(alphas_1)) | (np.isnan(alphas_2))) & (metric_data != 0.5))[0]
            in_cone_inds = np.where((alphas_1 <= beta) & (alphas_2 <= beta)  & (np.isfinite(alphas_1)) & (np.isfinite(alphas_2)) & (metric_data != 0.5))[0]

            all_alphas = np.r_[all_alphas, np.max(np.array([alphas_1, alphas_2]), axis=0)]

            # get in force closure
            in_cone_metrics = metric_data[in_cone_inds]
            out_of_cone_metrics = metric_data[out_of_cone_inds]

            all_in_cone_metrics = np.r_[all_in_cone_metrics, in_cone_metrics]
            all_out_of_cone_metrics = np.r_[all_out_of_cone_metrics, out_of_cone_metrics]

            in_cone_hist, in_cone_bins = np.histogram(in_cone_metrics, bins=num_bins, range=(0,1))
            out_of_cone_hist, out_of_cone_bins = np.histogram(out_of_cone_metrics, bins=num_bins, range=(0,1))
            all_in_cone_hist = all_in_cone_hist + in_cone_hist
            all_out_of_cone_hist = all_out_of_cone_hist + out_of_cone_hist

            # evaluate the planarity of each projection window
            dim = int(np.sqrt(proj_window_1_data.shape[1]))
            x_ind, y_ind = np.meshgrid(np.arange(dim), np.arange(dim))
            A = np.c_[np.ravel(x_ind), np.ravel(y_ind), np.ones(dim**2)]
            b1 = proj_window_1_data.T
            b2 = proj_window_2_data.T

            w1, se1, rank1, sing1 = np.linalg.lstsq(A, b1)
            w2, se2, rank2, sing2 = np.linalg.lstsq(A, b2)
            pred_w1 = A.dot(w1)
            pred_w2 = A.dot(w2)
            pred_w1_error = pred_w1 - b1
            mean_sq_dev_planarity_w1 = np.mean(pred_w1_error**2, axis=0)
            mean_abs_dev_planarity_w1 = np.sum(np.abs(pred_w1_error), axis=0)

            pred_w2_error = pred_w2 - b2
            mean_sq_dev_planarity_w2 = np.mean(pred_w2_error**2, axis=0)
            mean_abs_dev_planarity_w2 = np.sum(np.abs(pred_w2_error), axis=0)

            all_mean_sq_dev_planarity_w1 = np.r_[all_mean_sq_dev_planarity_w1, mean_sq_dev_planarity_w1]
            all_mean_sq_dev_planarity_w2 = np.r_[all_mean_sq_dev_planarity_w2, mean_sq_dev_planarity_w2]

            # evaluate the planarity of each projection window with cropped center            
            dim = int(np.sqrt(proj_window_1_data.shape[1]))
            center = dim / 2
            proj_window_1_data = proj_window_1_data.reshape(proj_window_1_data.shape[0], dim, dim)
            proj_window_2_data = proj_window_2_data.reshape(proj_window_2_data.shape[0], dim, dim)
            proj_window_1_data = proj_window_1_data[:,center-crop_dim/2:center+crop_dim/2+1,center-crop_dim/2:center+crop_dim/2+1]
            proj_window_2_data = proj_window_2_data[:,center-crop_dim/2:center+crop_dim/2+1,center-crop_dim/2:center+crop_dim/2+1]
            proj_window_1_data = proj_window_1_data.reshape(proj_window_1_data.shape[0], crop_dim**2)
            proj_window_2_data = proj_window_2_data.reshape(proj_window_2_data.shape[0], crop_dim**2)

            x_ind, y_ind = np.meshgrid(np.arange(crop_dim), np.arange(crop_dim))
            A = np.c_[np.ravel(x_ind), np.ravel(y_ind), np.ones(crop_dim**2)]
            b1 = proj_window_1_data.T
            b2 = proj_window_2_data.T

            w1, se1, rank1, sing1 = np.linalg.lstsq(A, b1)
            w2, se2, rank2, sing2 = np.linalg.lstsq(A, b2)
            pred_w1 = A.dot(w1)
            pred_w2 = A.dot(w2)
            pred_w1_error = pred_w1 - b1
            mean_sq_dev_planarity_w1_crop = np.mean(pred_w1_error**2, axis=0)
            mean_abs_dev_planarity_w1_crop = np.sum(np.abs(pred_w1_error), axis=0)

            pred_w2_error = pred_w2 - b2
            mean_sq_dev_planarity_w2_crop = np.mean(pred_w2_error**2, axis=0)
            mean_abs_dev_planarity_w2_crop = np.sum(np.abs(pred_w2_error), axis=0)

            all_mean_sq_dev_planarity_w1_crop = np.r_[all_mean_sq_dev_planarity_w1_crop, mean_sq_dev_planarity_w1_crop]
            all_mean_sq_dev_planarity_w2_crop = np.r_[all_mean_sq_dev_planarity_w2_crop, mean_sq_dev_planarity_w2_crop]

        # plot histogram of in cone and out of cone
        all_in_cone_norm_hist = all_in_cone_hist / np.sum(all_in_cone_hist)
        all_out_of_cone_norm_hist = all_out_of_cone_hist / np.sum(all_out_of_cone_hist)
        width = (out_of_cone_bins[1] - out_of_cone_bins[0])

        plt.figure()
        plt.bar(out_of_cone_bins[:-1], all_out_of_cone_norm_hist, width=width, color='r')
        plt.bar(in_cone_bins[:-1], all_in_cone_norm_hist, width=width, color='b')
        plt.title('Metric %s Density vs Friction Cone' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Quality', fontsize=font_size)
        plt.ylabel('Normalized Density', fontsize=font_size)
        plt.legend(('Out of Cone', 'In Cone'), loc='best')
        
        figname = 'metric_%s_histogram.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # evaluate accuracy of mean and median for each bin
        mean_in_cone_metric = np.mean(all_in_cone_metrics)
        mean_out_of_cone_metric = np.mean(all_out_of_cone_metrics)

        pred_in_cone_metric_error = all_in_cone_metrics - mean_in_cone_metric
        pred_out_of_cone_metric_error = all_out_of_cone_metrics - mean_out_of_cone_metric
        pred_error = np.r_[pred_in_cone_metric_error, pred_out_of_cone_metric_error]
        all_error_stats.append(ErrorStats(pred_error, tag = metric + '_mean_cone_bins'))
        logging.info('MSE for Mean Bin Predictor: %f' %(all_error_stats[-1].mse))
        logging.info('MAE for Mean Bin Predictor: %f' %(all_error_stats[-1].mae))
        
        median_in_cone_metric = np.median(all_in_cone_metrics)
        median_out_of_cone_metric = np.median(all_out_of_cone_metrics)

        pred_in_cone_metric_error = all_in_cone_metrics - median_in_cone_metric
        pred_out_of_cone_metric_error = all_out_of_cone_metrics - median_out_of_cone_metric
        pred_error = np.r_[pred_in_cone_metric_error, pred_out_of_cone_metric_error]
        all_error_stats.append(ErrorStats(pred_error, tag = metric + '_median_cone_bins'))
        logging.info('MSE for Median Bin Predictor: %f' %(all_error_stats[-1].mse))
        logging.info('MAE for Median Bin Predictor: %f' %(all_error_stats[-1].mae))

        # evaluate accuracy of friction cone angle predictor
        valid_inds = np.where(np.isfinite(all_alphas))[0]
        all_valid_alphas = all_alphas[valid_inds]
        all_valid_metrics = all_metrics[valid_inds]
        subsample_inds = np.arange(all_valid_metrics.shape[0])[::1000]
        rho = np.corrcoef(np.c_[all_valid_alphas, all_valid_metrics].T)[1,0]

        A = np.c_[all_valid_alphas, np.ones(all_valid_alphas.shape[0])]
        b = all_valid_metrics
        w, se, rank, s = np.linalg.lstsq(A, b)
        pred_metrics = A.dot(w)
        pred_error = pred_metrics - all_valid_metrics
        all_error_stats.append(ErrorStats(pred_error, tag = metric + '_friction_cone_angle'))
        logging.info('MSE for Friction Angle Regressor: %f' %(all_error_stats[-1].mse))
        logging.info('MAE for Friction Angle Regressor: %f' %(all_error_stats[-1].mae))

        plt.figure()
        plt.hist(pred_error, bins=num_bins)
        plt.title('Metric %s Friction Cone Angle Predictor Error' %(metric[:10]), fontsize=font_size)
        plt.xlim(-1,1)
        plt.xlabel('Pred Quality - True Quality', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)

        figname = 'metric_%s_cone_angle_pred_error_hist.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # scatter the PFC vs friction cone angle
        min_alpha = np.min(all_valid_alphas)
        max_alpha = np.max(all_valid_alphas)
        x_vals = [min_alpha * 180.0 / np.pi, max_alpha * 180.0 / np.pi]
        y_vals = [w[1] + w[0] * min_alpha, w[1] + w[0] * max_alpha]

        plt.figure()
        plt.scatter(all_valid_alphas[subsample_inds] * 180.0 / np.pi, all_valid_metrics[subsample_inds], c='b', s=50)
        plt.plot(x_vals, y_vals, c='r', linewidth=line_width)
        plt.xlim(x_vals[0]-eps, x_vals[1]+eps)
        plt.ylim(-eps,1+eps)
        plt.title('Metric %s vs Friction Cone Angle' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Friction Cone Angle (Degrees)', fontsize=font_size)
        plt.ylabel('Quality', fontsize=font_size)
        plt.legend(('Best Fit Line (rho=%.2f)' %(rho), 'Datapoint'), loc='best')

        figname = 'metric_%s_cone_angle_scatter.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # predict PFC by window planarity
        A = np.c_[all_mean_sq_dev_planarity_w1, all_mean_sq_dev_planarity_w2, np.ones(all_mean_sq_dev_planarity_w1.shape[0])]
        b = all_metrics
        w, se, rank, s = np.linalg.lstsq(A, b)
        pred_metrics = A.dot(w)
        pred_error = pred_metrics - all_metrics
        all_error_stats.append(ErrorStats(pred_error, tag = metric + '_planarity'))
        logging.info('MSE for Planarity Regressor: %f' %(all_error_stats[-1].mse))
        logging.info('MAE for Planarity Regressor: %f' %(all_error_stats[-1].mae))        

        plt.figure()
        plt.hist(pred_error, bins=num_bins)
        plt.xlim(-1,1)
        plt.title('Metric %s Planarity Predictor Error' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Pred Quality - True Quality', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)

        figname = 'metric_%s_planarity_pred_error_hist.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # evaluate to correlation between planarity and PFC
        subsample_inds = np.arange(all_metrics.shape[0])[::1000]
        rho = np.corrcoef(np.c_[all_mean_sq_dev_planarity_w1, all_mean_sq_dev_planarity_w2, all_metrics].T)

        min_p1 = np.min(all_mean_sq_dev_planarity_w1)
        max_p1 = np.max(all_mean_sq_dev_planarity_w1)
        min_p2 = np.min(all_mean_sq_dev_planarity_w2)
        max_p2 = np.max(all_mean_sq_dev_planarity_w2)
        x_vals = [min_p1, max_p1]
        y_vals = [w[2] + w[0] * min_p1 + w[1] * min_p2, w[2] + w[0] * max_p1 + w[1] * max_p2]

        plt.figure()
        plt.scatter(all_mean_sq_dev_planarity_w1[subsample_inds], all_metrics[subsample_inds], c='b', s=50)
        plt.scatter(all_mean_sq_dev_planarity_w2[subsample_inds], all_metrics[subsample_inds], c='g', s=50)
        plt.plot(x_vals, y_vals, c='r', linewidth=line_width)
        plt.xlim(min_p1-eps, max_p1+eps)
        plt.ylim(-eps,1+eps)
        plt.title('Metric %s vs Projection Window Planarity' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Deviation from Planarity', fontsize=font_size)
        plt.ylabel('Quality', fontsize=font_size)
        plt.legend(('Best Fit Line (rho=%.2f)' %(rho[2,0]), 'W1', 'W2'), loc='best')

        figname = 'metric_%s_planarity_scatter.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # predict PFC by cropped window planarity
        A = np.c_[all_mean_sq_dev_planarity_w1_crop, all_mean_sq_dev_planarity_w2_crop, np.ones(all_mean_sq_dev_planarity_w1_crop.shape[0])]
        b = all_metrics
        w, se, rank, s = np.linalg.lstsq(A, b)
        pred_metrics = A.dot(w)
        pred_error = pred_metrics - all_metrics
        all_error_stats.append(ErrorStats(pred_error, tag = metric + '_planarity_cropped'))
        logging.info('MSE for Cropped Planarity Regressor: %f' %(all_error_stats[-1].mse))
        logging.info('MAE for Cropped Planarity Regressor: %f' %(all_error_stats[-1].mae))        

        plt.figure()
        plt.hist(pred_error, bins=num_bins)
        plt.xlim(-1,1)
        plt.title('Metric %s Cropped Planarity Predictor Error' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Pred Quality - True Quality', fontsize=font_size)
        plt.ylabel('Count', fontsize=font_size)

        figname = 'metric_%s_cropped_planarity_pred_error_hist.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

        # evaluate to correlation between cropped planarity and PFC
        subsample_inds = np.arange(all_metrics.shape[0])[::1000]
        rho = np.corrcoef(np.c_[all_mean_sq_dev_planarity_w1_crop, all_mean_sq_dev_planarity_w2_crop, all_metrics].T)

        min_p1 = np.min(all_mean_sq_dev_planarity_w1_crop)
        max_p1 = np.max(all_mean_sq_dev_planarity_w1_crop)
        min_p2 = np.min(all_mean_sq_dev_planarity_w2_crop)
        max_p2 = np.max(all_mean_sq_dev_planarity_w2_crop)
        x_vals = [min_p1, max_p1]
        y_vals = [w[2] + w[0] * min_p1 + w[1] * min_p2, w[2] + w[0] * max_p1 + w[1] * max_p2]

        plt.figure()
        plt.scatter(all_mean_sq_dev_planarity_w1_crop[subsample_inds], all_metrics[subsample_inds], c='b', s=50)
        plt.scatter(all_mean_sq_dev_planarity_w2_crop[subsample_inds], all_metrics[subsample_inds], c='g', s=50)
        plt.plot(x_vals, y_vals, c='r', linewidth=line_width)
        plt.xlim(min_p1-eps, max_p1+eps)
        plt.ylim(-eps,1+eps)
        plt.title('Metric %s vs Cropped Projection Window Planarity' %(metric[:10]), fontsize=font_size)
        plt.xlabel('Deviation from Planarity', fontsize=font_size)
        plt.ylabel('Quality', fontsize=font_size)
        plt.legend(('Best Fit Line (rho=%.2f)' %(rho[2,0]), 'W1', 'W2'), loc='best')

        figname = 'metric_%s_cropped_planarity_scatter.pdf' %(metric)
        plt.savefig(os.path.join(output_dir, figname), dpi=dpi)

    # save all metrics in a csv
    csvname = 'error_stats.csv'
    ErrorStats.stats_to_csv(all_error_stats, os.path.join(output_dir, csvname))
        
    IPython.embed()
    exit(0)
                       
