from __future__ import print_function

import unittest

class BasicTestCase(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(True, True)

class DependencyTestCase(unittest.TestCase):
    def test_numpy(self):
        import numpy
        print()
        print('[numpy]', numpy.version.version)
