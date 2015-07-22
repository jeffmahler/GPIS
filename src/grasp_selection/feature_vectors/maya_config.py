import yaml
import os.path as path

MAYAPY_KEY = 'path_to_mayapy'
DEST_DIR_KEY = 'dest_dir'
MESH_DIR_KEY = 'mesh_dir'

RENDER_TYPE_KEY = 'render_type'
NUM_RADIAL_KEY = 'num_radial'
NUM_LAT_KEY = 'num_lat'
NUM_LONG_KEY = 'num_long'


class MayaConfig(object):
	def __init__(self, config=None):
		if config == None:
			config = yaml.load(open('cfg/maya_config.yaml', 'r'))
		self._parse_config(config)

	def _parse_config(self, config):
		self.mayapy_ = config[MAYAPY_KEY]
		self.dest_dir_ = config[DEST_DIR_KEY]
		self.mesh_dir_ = config[MESH_DIR_KEY]
		self.render_type_ = config[RENDER_TYPE_KEY]
		self.num_radial_ = config[NUM_RADIAL_KEY]
		self.num_lat_ = config[NUM_LAT_KEY]
		self.num_long_ = config[NUM_LONG_KEY]

	def mayapy(self):
		return self.mayapy_

	def dest_dir(self):
		return self.dest_dir_

	def mesh_dir(self):
		return self.mesh_dir_

	def render_type(self):
		return self.render_type_

	def num_radial(self):
		return self.num_radial_

	def num_lat(self):
		return self.num_lat_

	def num_long(self):
		return self.num_long_

