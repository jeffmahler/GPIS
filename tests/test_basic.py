import unittest

class BasicTestCase(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(True, True)
