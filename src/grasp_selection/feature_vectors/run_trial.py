import os
import yaml
import paths
from dataset_sorter import DatasetSorter
from object_database import Cat50ObjectDatabase
import train_cnn

object_database_name = 'Cat50'
path_to_object_database = '/Users/MelRod/myProjects/GPIS_data/Cat50_ModelDatabase/'
path_to_index = path_to_object_database+'index.db'

#script
# maya_renderer_options = '--max_range 0.8 --num_radial 2 --num_lat 5 --num_long 5 -c'
# os.system(paths.path_to_mayapy+' '+paths.path_to_src_dir+'maya_renderer.py '+paths.path_to_image_dir+' --mesh_dir '+path_to_object_database+' '+maya_renderer_options)

object_database = Cat50ObjectDatabase(path_to_index)
data_set_sorter = DatasetSorter(object_database, portion_training=0.75)
data_set_sorter.save_to_dir(paths.path_to_data_dir)

train_cnn.train(data_set_sorter.train_object_names, object_database, paths.path_to_image_dir, portion_training=0.75)


