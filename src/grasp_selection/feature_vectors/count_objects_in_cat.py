import os
from glob import glob

dex_net_root = '/Users/MelRod/myProjects/dex_net/'

path_to_data_dir = dex_net_root+'data/'
path_to_image_dir = dex_net_root+'mesh_images/'
model = path_to_data_dir+'deploy.prototxt'
pretrained = dex_net_root+'caffe_trials/7/caffenet_train_iter_30000.caffemodel'


path_to_index_file = path_to_data_dir+'Cat50_ModelDatabase_index.db'
path_to_category_file = path_to_data_dir+'category_to_id.txt'

def create_name_to_category(path_to_file):
	name_to_category = {}
	with open(path_to_file) as file:
		for line in file:
			split = line.split()
			file_name = split[0]
			category = split[1]
			name_to_category[file_name] = category
	return name_to_category

name_to_category = create_name_to_category(path_to_index_file)
images_per_cat = {}

os.chdir(path_to_image_dir)
for object_name in map(lambda dir: dir[:-1], glob("*/")):
	cat = name_to_category[object_name]
	if cat in images_per_cat:
		images_per_cat[cat] = images_per_cat[cat]+1
	else:
		images_per_cat[cat] = 1

max_cat = ''
max = 0
for cat in images_per_cat.keys():
	if images_per_cat[cat] > max:
		max = images_per_cat[cat]
		max_cat = cat

print max_cat
print max