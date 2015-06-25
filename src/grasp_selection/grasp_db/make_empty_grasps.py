# Constructs empty Grasp objects: only data they have is an image path

import pickle
from glob import glob

class Grasp:
	def __init__(self, image_path, obj_file, label=0, scale=1, mass=1):
		self.image = image_path
		self.obj_file = obj_file
		self.label = label
		self.scale = scale
		self.mass = mass

	def mark_good(self):
		self.label = 1

	def mark_bad(self):
		self.label = -1

	def mark_undecided(self):
		self.label = 0

	def mark_scale(self, scale_val):
		self.scale = scale_val

	def mark_mass(self, mass):
		self.mass = mass

# Make lists of images from folders
images1 = glob('images/3dnet_bottles/*.png')
images2 = glob('images/apc/*.png')
images3 = glob('images/ycb/*.png')
images = images1 + images2 + images3

file_name = "unlabeled_grasps.dat"
file_object = open(file_name, 'wb') # binary encoding

grasps_list = []

# Make list of Grasp objects
for i in range(len(images)):
	new_grasp = Grasp('../grasp_db/' + images[i], '../grasp_db/obj_files/Co_clean.obj')
	grasps_list += [new_grasp]

pickle.dump(grasps_list, file_object)
file_object.close()
