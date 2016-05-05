from __future__ import print_function

import unittest

class BasicTestCase(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(True, True)

class ImportTestCase(unittest.TestCase):
    def test_core(self):
        import ipython
        import matplotlib
        import numpy
        import scipy

    def test_imaging(self):
        import PIL
        import skimage
        import sklearn

    def test_experiments(self):
        import h5py
        import yaml

    def test_misc(self):
        import nearpy
        import sklearn
