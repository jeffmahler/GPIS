import numpy as np
from scipy.stats import spearmanr
import csv
import os

class CSVStatistics:

    HIST_STATS = {
        'mean': np.mean,
        'median': np.median,
        'std': np.std
    }
    
    SCATTER_STATS = {
        'pearson': lambda x: np.corrcoef(x, rowvar=0)[1,0], 
        'spearman': lambda x: spearmanr(x)[0],
        'p': lambda x: spearmanr(x)[1]
    }

    def __init__(self, output_file, stats_map):
        self._output_file = output_file
        self._stats_map = stats_map
        
        self._all_data = {}
        self._stats = None
        
    def append_data(self, title, data):
        self._stats = None
        self._all_data[title] = data
        
    def _gen_stats(self, title, data):
        stats = {}
        stats['title'] = title
        for stat_name, stat_function in self._stats_map.items():
            stats[stat_name] = stat_function(data)
        return stats
        
    @property
    def stats(self):
        if self._stats is None:
            self._stats = []
            for title, data in self._all_data.items():
                self._stats.append(self._gen_stats(title, data))
        return self._stats
        
    def save(self, output_file=None):
        if output_file is None:
            output_file = self._output_file

        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'wb') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['title'] + self._stats_map.keys())
            writer.writeheader()
            writer.writerows(self.stats)