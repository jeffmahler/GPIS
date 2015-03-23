import numpy as np
import sdf_class as sdf
import sys

import IPython

if __name__ == '__main__':
    filename = sys.argv[1]
    s = sdf.SDF(filename)
    s.make_plot()
    IPython.embed()
    
