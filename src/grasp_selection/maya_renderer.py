import csv
import shutil
import glob
import os
import os.path as path
import math
import argparse
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
sys.path.append('/usr/local/Cellar/python/2.7.10/Frameworks/Python.framework/Versions/2.7/lib/python2.7')
import IPython
#IPython.embed()

import database as db
import experiment_config as ec

from PIL import Image
import numpy as np
import scipy.ndimage.filters as filters
import tfx

import maya.standalone
import maya.cmds as cmd
import maya.mel as mel

INCHES_TO_MM = 25.4

class MayaRenderer(object):
	def __init__(self, config):
		maya.standalone.initialize(name='python')
		mel.eval('source "renderLayerBuiltinPreset.mel"')

		self.config_ = config
		self.maya_config_ = self.config_['maya']
		self._parse_config()

		self.center_of_interest_ = [0,0,0]

		# gen layer names
		self.depth_layer_name_ = "DEPTH_LAYER"
		self.obj_name_ = "OBJECT"
		self.plane_name_ = "PLANE"

	def _parse_config(self):
		self.dest_dir_ = self.maya_config_['dest_dir']
		self.mesh_dir_ = self.maya_config_['mesh_dir']
		self.min_dist_ = self.maya_config_['min_dist']
		self.max_dist_ = self.maya_config_['max_dist']
		self.working_min_dist_ = self.min_dist_
		self.working_max_dist_ = self.max_dist_
		self.num_radial_ = self.maya_config_['num_radial']
		self.num_lat_ = self.maya_config_['num_lat']
		self.num_long_ = self.maya_config_['num_long']
		self.min_range_ = self.maya_config_['min_range']
		self.max_range_ = self.maya_config_['max_range']
		self.back_color_ = self.maya_config_['back_color']

		self.render_mode_ = self.maya_config_['render_mode']
		self.use_table_ = self.maya_config_['use_table']
		self.file_type_ = self.maya_config_['file_type']
		self.normalize_ = self.maya_config_['normalize']

	def add_depth_layer(self):
		if self.use_table_:
			cmd.select(self.obj_name_+":Mesh", self.plane_name_, r=True)
		else:
			cmd.select(self.obj_name_+":Mesh")
		cmd.createRenderLayer(name=self.depth_layer_name_)
		mel.eval("renderLayerBuiltinPreset linearDepth DEPTH_LAYER")
		cmd.disconnectAttr("samplerInfo1.cameraNearClipPlane", "setRange1.oldMinX")
		cmd.disconnectAttr("samplerInfo1.cameraFarClipPlane", "setRange1.oldMaxX")
		cmd.setAttr("setRange1.minX", 0)
		cmd.setAttr("setRange1.maxX", 1)
		cmd.setAttr("setRange1.oldMinX", self.min_range_ * self.working_min_dist_)
		cmd.setAttr("setRange1.oldMaxX", self.max_range_ * self.working_max_dist_)

	def add_object_segmentation():
		self.create_mask_for_object_with_color(self.obj_name_+":Mesh", [1, 1, 1])
		self.create_mask_for_object_with_color(self.plane_name_, [1, 0, 0])

	def create_scene_with_mesh(self, mesh_filename, rot=np.eye(3)):
		cmd.file(f=True, new=True)

		if self.use_table_:
			cmd.nurbsPlane(name=self.plane_name_, p=(0,0,0), ax=(0,0,1), w=10000, lr=1, d=3, u=1, v=1, ch=1)

		try:
			cmd.file(mesh_filename, i=True, ns=self.obj_name_)
		except RuntimeError as e:
			print 'Failed to import mesh file: '+mesh_filename
			print e.message
			return False

		axes = 'sxyz'
		r = tfx.canonical.CanonicalRotation(rot)
		euler = r.euler(axes)
		euler = [(180. / np.pi) * a for a in euler] # convert to degrees

		cmd.select(self.obj_name_+":Mesh", r=True)
		cmd.rotate(euler[0], euler[1], euler[2])

		bounding_box = cmd.polyEvaluate(b=True)
		object_height = bounding_box[2][1]
		cmd.move(object_height, z=True)
		self.center_of_interest_ = [0, 0, object_height]

		if self.min_dist_ == 0:
			major_dist = math.sqrt(math.pow(bounding_box[0][0]-bounding_box[0][1], 2) + math.pow(bounding_box[1][0]-bounding_box[1][1], 2) + math.pow(bounding_box[2][0]-bounding_box[2][1], 2))
			self.working_min_dist_ = major_dist*2
			self.working_max_dist_ = major_dist*4
			
		cmd.setAttr("defaultRenderGlobals.imageFormat", 8) # 32)
		cmd.setAttr("defaultResolution.width", 256)
		cmd.setAttr("defaultResolution.height", 256)
		cmd.setAttr("defaultResolution.deviceAspectRatio", 1.0)

		return True

	def create_mask_for_object_with_color(self, obj_name, color):
		mask_name = obj_name+"_MASK"
		group_name = obj_name+"_GROUP"

		cmd.shadingNode("surfaceShader", name=mask_name, asShader=True)
		cmd.setAttr(mask_name+".outColor", color[0], color[1], color[2], type="double3")

		cmd.sets(name=group_name, renderable=True, empty=True)
		cmd.surfaceShaderList(mask_name, add=group_name)
		cmd.sets(obj_name, e=True, forceElement=group_name)

	def normalize_image(self, im_arr):
		""" Normalize an image to between 0 and 255 """
		im_max = np.max(im_arr)
		im_min = np.min(im_arr)

		a = 255. / (im_max - im_min)
		b = - a * im_min
		im_arr = np.uint8(a * im_arr + b)
		return im_arr

	def median_filter(self, im_arr, size=3.0):
		""" Median filter """
		im_arr_filt = filters.median_filter(np.float64(im_arr), size=size)
		return im_arr_filt

	def differentiate_image(self, im_arr, normalize=False):
		""" Take x and y gradients of images """
		im_grad_y, im_grad_x = np.gradient(np.float64(im_arr[:,:,0]))
		
		im_grad_x_big = np.zeros([im_grad_x.shape[0], im_grad_x.shape[1], 3])
		im_grad_x_big[:,:,0] = im_grad_x
		im_grad_x_big[:,:,1] = im_grad_x
		im_grad_x_big[:,:,2] = im_grad_x

		im_grad_y_big = np.zeros([im_grad_y.shape[0], im_grad_y.shape[1], 3])
		im_grad_y_big[:,:,0] = im_grad_y
		im_grad_y_big[:,:,1] = im_grad_y
		im_grad_y_big[:,:,2] = im_grad_y

		return im_grad_x_big, im_grad_y_big
		
	def save_image_with_camera_pos(self, csv_writer, mesh_filename, file_ext, dest_dir, camera_pos, camera_interest_pos, obj_key):
		camera_name, camera_shape = cmd.camera(p=camera_pos, wci=camera_interest_pos)
		cmd.setAttr(camera_shape+'.backgroundColor', self.back_color_['r'], self.back_color_['g'], self.back_color_['b'], type="double3")
		cmd.setAttr(camera_shape+".renderable", 1)
		focal_length = cmd.camera(camera_shape, q=True, fl=True)

		app_horiz = cmd.camera(camera_shape, q=True, hfa=True) * INCHES_TO_MM
		app_vert = cmd.camera(camera_shape, q=True, vfa=True) * INCHES_TO_MM
		pixel_width = cmd.getAttr("defaultResolution.width")
		pixel_height = cmd.getAttr("defaultResolution.height")

		focal_length_x_pixel = pixel_width * focal_length / app_horiz
		focal_length_y_pixel = pixel_height * focal_length / app_vert

		image_src = cmd.render(camera_shape)
		image_file = mesh_filename+file_ext
		image_dest = path.join(dest_dir, obj_key, image_file)
		shutil.move(image_src, image_dest)
		im = Image.open(image_dest)
		im_arr = np.array(im)

		# postprocess the image
		im_arr = self.median_filter(im_arr)

		if self.render_mode_ == 'normal':
			im_grad_x_arr, im_grad_y_arr = self.differentiate_image(im_arr)
			
			if self.normalize_:
				im_grad_x_arr = self.normalize_image(im_grad_x_arr)
				im_grad_y_arr = self.normalize_image(im_grad_y_arr)

			image_x_dest = path.join(dest_dir, obj_key, mesh_filename + '_grad_x' + file_ext)
			image_y_dest = path.join(dest_dir, obj_key, mesh_filename + '_grad_y' + file_ext)
			im_grad_x = Image.fromarray(np.uint8(im_grad_x_arr))
			im_grad_y = Image.fromarray(np.uint8(im_grad_y_arr))
			im_grad_x.save(image_x_dest)
			im_grad_y.save(image_y_dest)

		
		# normalize the rendered image
		if self.normalize_:
			im_arr = self.normalize_image(im_arr)
		im = Image.fromarray(np.uint8(im_arr))
		im.save(image_dest)

		self.save_camera_data_to_writer(csv_writer, mesh_filename, camera_pos, camera_interest_pos, focal_length)

	def save_camera_data_to_writer(self, csv_writer, image_file, camera_pos, camera_interest_pos, focal_length):
		csv_writer.writerow(camera_pos + camera_interest_pos + [focal_length, image_file])

	def create_images_for_scene(self, csv_writer, obj_name, obj_key):
		radius = self.working_min_dist_
		radial_increment = 0 if self.num_radial_ == 1 else (self.working_max_dist_ - self.working_min_dist_) / (self.num_radial_ - 1)

		# TODO: view halo or view hemisphere when on table instead of view sphere
		mult = 1
		if self.use_table_:
			mult = 2

		for r in range(0, self.num_radial_):
			phi_increment = math.pi / (mult * (self.num_lat_ + 1))
			phi = phi_increment
			for lat in range(0, self.num_lat_):
				theta = 0
				theta_increment = 2 * math.pi / self.num_long_
				for lon in range(0, self.num_long_):
					camera_pos = [radius*math.sin(phi)*math.cos(theta), radius*math.sin(phi)*math.sin(theta), radius*math.cos(phi)]
					mesh_name = obj_name+"_"+str(r)+"_"+str(lat)+"_"+str(lon)
					self.save_image_with_camera_pos(csv_writer, mesh_name, self.file_type_, self.dest_dir_, camera_pos, self.center_of_interest_, obj_key)
					theta += theta_increment
				phi += phi_increment
			radius += radial_increment

	def render(self, graspable, rot=np.eye(3)):
		render_dir = os.path.join(self.dest_dir_, graspable.key)
		if not os.path.exists(render_dir):
			os.mkdir(render_dir)

		import_success = self.create_scene_with_mesh(graspable.model_name, rot)

		if import_success:
			with open(path.join(self.dest_dir_, graspable.key, 'camera_table.csv'), 'w') as csvfile:
				csv_writer = csv.writer(csvfile)
				csv_writer.writerow(["camera_x", "camera_y", "camera_z", "interest_x", "interest_y", "interest_z", "focal_length", "mesh_name"])

				if self.render_mode_ == 'color':
					self.create_images_for_scene(csv_writer, graspable.key+"_color", graspable.key)

				if self.render_mode_ == 'segmask':
					self.add_object_segmentation()
					self.create_images_for_scene(csv_writer, graspable.key+"_segmask", graspable.key)

				if self.render_mode_ == 'depth':
					self.add_depth_layer()
					self.create_images_for_scene(csv_writer, graspable.key+"_depth", graspable.key)

				if self.render_mode_ == 'normal':
					self.add_depth_layer()
					self.create_images_for_scene(csv_writer, graspable.key+"_normal", graspable.key)
			return True
		else:
			os.rmdir(render_dir)
			return False

if __name__ == '__main__':
	config_filename = sys.argv[1]
	config = ec.ExperimentConfig(config_filename)

	renderer = MayaRenderer(config)
	ds = db.Dataset(config['dataset'], config)
	for obj in ds:
		if obj.key == 'sissor_handtool':
			stable_poses = ds.load_stable_poses(obj)
			print stable_poses[0].r
			renderer.render(obj, rot=stable_poses[0].r)
			exit(0)

# def create_robot_pose_matrix(camera_pos, camera_interest_pos):
# 	z_axis = numpy.subtract(camera_interest_pos, camera_pos)
# 	z = numpy.linalg.norm(z_axis)
# 	x = numpy.linalg.norm(numpy.cross([0,1,0], z))
# 	y = numpy.linalg.norm(numpy.cross(z, x))
