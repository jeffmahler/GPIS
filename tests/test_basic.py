import unittest

class BasicTestCase(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(True, True)

class DependencyTestCase(unittest.TestCase):
    def test_numpy(self):
        import numpy
        print(numpy.version.version)
