# Constructs empty Grasp objects: only data they have is an image path

import pickle
from glob import glob

class Grasp:
	def __init__(self, image_path, obj_file, label=0, scale=1):
		self.image = image_path
		self.obj_file = 
		self.label = 0
		self.scale = 1

	def mark_good(self):
		self.label = 1

	def mark_bad(self):
		self.label = -1

	def mark_undecided(self):
		self.label = 0

	def mark_scale(self, scale_val):
		self.scale = scale_val

# Make lists of images from folders
images1 = glob('grasp_db/3dnet_bottles/*.png')
images2 = glob('grasp_db/apc/*.png')
images3 = glob('grasp_db/ycb/*.png')
images = images1 + images2 + images3

file_name = "unlabeled_grasps.dat"
file_object = open(file_name, 'wb') # binary encoding

grasps_list = []

# Make list of Grasp objects
for i in range(len(images)):
	new_grasp = Grasp(images[i], 'monkey.dat')
	grasps_list += [new_grasp]

pickle.dump(grasps_list, file_object)
file_object.close()
