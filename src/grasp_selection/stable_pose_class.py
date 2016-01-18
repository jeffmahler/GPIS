"""
A basic struct-like Stable Pose class to make accessing pose probability and rotation matrix easier

Author: Nikhil Sharma
"""

class StablePose(object):
    def __init__(self, p, r, x0):
        """
        Initializes new stable pose object.

        p -- probability of given pose
        r -- rotation matrix for stable pose
        x0 -- point lying on the table
        """
        self.p = p
        self.r = r
        self.x0 = x0
