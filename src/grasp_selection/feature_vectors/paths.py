import yaml

config = yaml.load(open('config/config.yaml', 'r'))
path_to_mayapy = config['path_to_mayapy']
caffe_root = config['caffe_root']

caffe_tools=caffe_root+'build/tools/'

path_to_src_dir = 'src/'
path_to_data_dir = 'data/'
path_to_image_dir = 'depth_images/'

solver = path_to_data_dir+'solver.prototxt'
alexnet = path_to_data_dir+'alexnet.caffemodel'
training_model = path_to_data_dir+'deploy.prototxt'
mean_file = path_to_data_dir+'mean.npy'