"""
Class to encapsulate storage of errors
Author: Jeff
"""
import csv
import IPython
import logging
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')#prevents using X server backend for matplotlib
import matplotlib.pyplot as plt

import plotting

class ContinuousErrorStats:
    TAG='tag'

    def __init__(self, truth, predictions, tag):
        if not isinstance(truth, np.ndarray):
            raise ValueError('Must provide numpy ndarrays to compute statistics')
        if not isinstance(predictions, np.ndarray):
            raise ValueError('Must provide numpy ndarrays to compute statistics')
        if truth.shape[0] != predictions.shape[0]:
            raise ValueError('Must provide numpy ndarrays to compute statistics')

        self.tag = tag # the name of the errors
        self.pred_error = truth - predictions
        self._compile_stats()

    @property
    def error_types(self):
        """ Returns a dictionary of different views of the errors in arrays"""
        # compute error types
        sq_err = self.pred_error**2
        abs_err = np.abs(self.pred_error)
        raw_err = self.pred_error

        # place in dictionary
        error_tags = ['sq_err', 'abs_err', 'raw_err']
        error_types = [sq_err, abs_err, raw_err]
        errors_dict = {}
        [errors_dict.update({k:v}) for k, v in zip(error_tags, error_types)]
        return errors_dict

    @property
    def dict(self):
        """ Error stats as a dictionary """
        output_dict = {ContinuousErrorStats.TAG: self.tag}
        output_dict.update(self.stats_dict)
        return output_dict

    def _compile_stats(self, d_pct = 10):
        """ Computes key statistics and stores in a dictionary """
        self.stats_dict = {}

        # compute errors for each type
        for tag, err in self.error_types.iteritems():            
            # save standard statistics
            self.stats_dict.update({
                    '%s_mean' %(tag): np.mean(err),
                    '%s_med' %(tag): np.median(err),
                    '%s_max' %(tag): np.max(err),
                    '%s_min' %(tag): np.min(err),
                    '%s_std' %(tag): np.std(err)
                    })

            # compute each 10th error percentile
            cur_pct = d_pct
            while cur_pct < 100:
                self.stats_dict.update({
                        '%s_%dth_pctle' %(tag, cur_pct): np.percentile(err, cur_pct)
                        })
                cur_pct += d_pct

    def plot_error_histograms(self, num_bins=100, min_range=None, max_range=None,
                              normalize=False, color='b', 
                              font_size=15, dpi=100, output_dir=None):
        """ Plots histograms of the errors. Auto-saves figures to output directory if specified """
        for err_tag, err in self.error_types.iteritems():
            # gen axis labels
            tokens = err_tag.split('_')
            x_label = ''.join(c.upper() if i == 0 else c for i, c in enumerate(self.tag)) + ' '
            for token in tokens:
                x_label += ''.join(c.upper() if i == 0 else c for i, c in enumerate(token)) + ' '
            y_label = 'Counts'
            if normalize:
                y_label = 'Normalized Counts'
            title = x_label + ' Histogram'

            # create figure of histogram
            plt.figure()
            plotting.plot_histogram(err, min_range=min_range, max_range=max_range,
                                    num_bins=num_bins, normalize=normalize, color=color)
            
            plt.xlabel(x_label, fontsize=font_size)
            plt.ylabel(y_label, fontsize=font_size)
            plt.title(title, fontsize=font_size)

            if output_dir is not None:
                figname = '%s_%s_histogram.pdf' %(self.tag, err_tag)
                plt.savefig(os.path.join(output_dir, figname), dpi=dpi)
                plt.close()
            
    @staticmethod
    def stats_to_csv(stats_list, filename):
        """ Save a list of error stats objects to a csv """
        # check valid input data
        if len(stats_list) == 0:
            return
        for i, stats in enumerate(stats_list):
            if not isinstance(stats, ContinuousErrorStats):
                raise ValueError('Item at index %d is not a ContinuousErrorStats object' %(i))

        # parse filename
        headers = stats_list[0].dict.keys()
        headers.sort()
        root, ext = os.path.splitext(filename)
        if ext.lower() != '.csv':
            raise ValueError('Must save as a .csv file')

        # open output csv file
        if os.path.exists(filename):
            f = open(filename, 'a')
            csv_writer = csv.DictWriter(f, headers)
        else:
            f = open(filename, 'w')
            csv_writer = csv.DictWriter(f, headers)
            csv_writer.writeheader()

        # write each stat
        for stats in stats_list:
            csv_writer.writerow(stats.dict)

        f.close()

def test_error_stats(num_datapoints=1000):
    mu = 2 * (np.random.rand() - 0.5)
    sigma = 2*(np.random.rand() + 0.01)
    logging.info('Generating data from gaussian with mean=%.2f and stddev=%.2f' %(mu, sigma))

    random_truth = np.random.normal(mu, sigma, size=num_datapoints)
    random_predictions = np.random.normal(mu, sigma, size=num_datapoints)

    e = ContinuousErrorStats(random_truth, random_predictions, 'random')
    e.plot_error_histograms()
    plt.show()

    e.plot_error_histograms(output_dir='results/test/error_statistics')

    e.stats_to_csv([e], 'results/test/error_statistics/out.csv')

if __name__ == '__main__':
    np.random.seed(100)
    logging.getLogger().setLevel(logging.INFO)
    test_error_stats()
