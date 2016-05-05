from __future__ import print_function

import unittest

class BasicTestCase(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(True, True)

class ImportTestCase(unittest.TestCase):
    def test_core(self):
        import IPython
        import matplotlib
        import numpy
        import scipy

        import PIL
        import catkin_pkg
        import cvxopt
        import h5py
        import nearpy
        import skimage
        import sklearn
        import sklearn
        import tfx
        import yaml
