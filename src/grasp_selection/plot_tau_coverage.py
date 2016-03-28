import IPython
import json
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

masked_object_tags = ['_no_mask', '_masked_bbox', '_masked_hull']

if __name__ == '__main__':
    experiment_dir = sys.argv[1]

    # read metrics
    privacy_metric_filename = os.path.join(experiment_dir, 'privacy_metrics.json')
    f = open(privacy_metric_filename, 'r')
    privacy_metrics = json.load(f)
    f.close()
    coverage_metric_filename = os.path.join(experiment_dir, 'coverage_metrics.json')
    f = open(coverage_metric_filename, 'r')
    coverage_metrics = json.load(f)
    f.close()

    font_size = 15
    line_width = 2
    dpi = 400

    # plot
    for obj_key in coverage_metrics.keys():
        taus = [0.0]
        coverages = [np.exp(-coverage_metrics[obj_key]['raw_coll_free'])]
        for metric, val in coverage_metrics[obj_key].iteritems():
            if metric.find('hull') != -1 and metric.find('robust_tau') != -1:
                taus.append(float(metric[-2:]))
                coverages.append(np.exp(-val))

        taus_and_coverages = zip(taus, coverages)
        taus_and_coverages.sort(key= lambda x: x[0])
        taus = [t[0] for t in taus_and_coverages]
        coverages = [t[1] for t in taus_and_coverages]
                
        plt.figure()
        plt.plot(taus, coverages, linewidth=line_width, c='g')
        plt.xlabel('Robustness', fontsize=font_size)
        plt.ylabel('Coverage', fontsize=font_size)
        plt.title('Robustness vs Coverage', fontsize=font_size)
        figname = 'robustness_vs_cov_obj_%s.pdf' %(obj_key)
        plt.savefig(os.path.join(experiment_dir, figname), dpi=dpi)


