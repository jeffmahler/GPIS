import os
import yaml

object_database_name = 'Cat50'
path_to_object_database = '/Users/MelRod/myProjects/GPIS_data/Cat50_ModelDatabase/'
path_to_index = path_to_object_database+'index.db'


#generic
config = yaml.load('config/config.yaml')
path_to_mayapy = config['path_to_mayapy']

caffe_tools=caffe_root+'build/tools/'

path_to_src_dir = 'src/'
path_to_data_dir = 'data/'
path_to_image_dir = 'depth_images/'

solver = path_to_data_dir+'solver.prototxt'
weights = path_to_data_dir+'alexnet.caffemodel'
model = path_to_data_dir+'deploy.prototxt'


#script
maya_renderer_options = '--min_dist 0.25 --max_dist 0.5 --max_range 0.8 --num_radial 2 --num_lat 5 --num_long 5 -d'
os.system(path_to_mayapy+' '+path_to_src_dir+'maya_renderer.py '+path_to_image_dir+' --mesh_dir '+path_to_object_database+' '+maya_renderer_options)

os.system('python '+path_to_src_dir+'create_train_test_sets.py '+path_to_index+' '+path_to_data_dir+' '+path_to_image_dir+' --portion_training 0.75 --portion_cnn_training 0.75')

os.system(caffe_tools+'caffe train -solver '+solver+' -weights '+weights)

