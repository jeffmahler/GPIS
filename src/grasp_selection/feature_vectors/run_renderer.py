import os
from maya_config import MayaConfig

REL_PATH_TO_MAYA_RENDERER = 'src/grasp_selection/feature_vectors/'

if __name__ == '__main__':
	maya_config = MayaConfig()

	if not os.path.exists(maya_config.dest_dir()):
		os.mkdir(maya_config.dest_dir())

	back_color = maya_config.back_color()
	os_cmd = maya_config.mayapy()+\
				' '+os.path.join(REL_PATH_TO_MAYA_RENDERER+'maya_renderer.py')+\
				' '+maya_config.dest_dir()+\
				' --mesh_dir '+str(maya_config.mesh_dir())+' '+\
				' --num_radial '+str(maya_config.num_radial())+\
				' --num_lat '+str(maya_config.num_lat())+\
				' --num_long '+str(maya_config.num_long())+\
				' --back_color '+str(back_color['r'])+' '+str(back_color['g'])+' '+str(back_color['b'])+' '+\
				' --'+maya_config.render_type()

	os.system(os_cmd)

