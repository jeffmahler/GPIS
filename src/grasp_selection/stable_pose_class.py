"""
A basic struct-like Stable Pose class to make accessing pose probability and rotation matrix easier

Author: Nikhil Sharma
"""
import numpy as np

class StablePose(object):
    def __init__(self, p, r, x0, stp_id=-1):
        """
        Initializes new stable pose object.

        p -- probability of given pose
        r -- rotation matrix for stable pose
        x0 -- point lying on the table
        stp_id -- string identifier for the pose in the database
        """
        self.p = p
        self.r = r
        self.x0 = x0
        self.id = stp_id

        # HACK: to fix stable pose bug
        if np.abs(np.linalg.det(self.r) + 1) < 0.01:
            self.r[1,:] = -self.r[1,:]
