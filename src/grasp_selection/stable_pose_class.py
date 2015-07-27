"""
A basic struct-like Stable Pose class to make accessing pose probability and rotation matrix easier

Author: Nikhil Sharma
"""

class StablePose:
	def __init__(self, p, r):
		"""
		Initializes new stable pose object.

		p -- probability of given pose
		r -- rotation matrix for stable pose
		"""
		self.p = p
		self.r = r

