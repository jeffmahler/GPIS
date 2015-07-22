import os
from maya_config import MayaConfig

REL_PATH_TO_MAYA_RENDERER = 'src/grasp_selection/feature_vectors/'

if __name__ == '__main__':
	maya_config = MayaConfig()

	os_cmd = maya_config.mayapy()+\
				' '+os.path.join(REL_PATH_TO_MAYA_RENDERER+'maya_renderer.py')+\
				' '+maya_config.dest_dir()+\
				' --mesh_dir '+str(maya_config.mesh_dir())+' '+\
				' --num_radial '+str(maya_config.num_radial())+\
				' --num_lat '+str(maya_config.num_lat())+\
				' --num_long '+str(maya_config.num_long())+\
				' --'+maya_config.render_type()

	os.system(os_cmd)

