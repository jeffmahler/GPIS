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
import rendered_image as ri

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
		# initialize maya
		maya.standalone.initialize(name='python')
		mel.eval('source "renderLayerBuiltinPreset.mel"')

		# parse config
		self.config_ = config
		self.maya_config_ = self.config_['maya']
		self._parse_config()

		self.center_of_interest_ = [0,0,0]

		# gen layer names
		self.depth_layer_name_ = "DEPTH_LAYER"
		self.obj_name_ = "OBJECT"
		self.plane_name_ = "PLANE"

	def _parse_config(self):
		""" Read in the parameters of a configuration file """
		self.dest_dir_ = self.maya_config_['dest_dir']
		self.mesh_dir_ = self.maya_config_['mesh_dir']
		self.min_dist_ = self.maya_config_['min_dist']
		self.max_dist_ = self.maya_config_['max_dist']
		self.num_radial_ = self.maya_config_['num_radial']
		self.num_lat_ = self.maya_config_['num_lat']
		self.num_long_ = self.maya_config_['num_long']
		self.min_range_ = self.maya_config_['min_range']
		self.max_range_ = self.maya_config_['max_range']
		self.back_color_ = self.maya_config_['back_color']

		self.use_table_ = self.maya_config_['use_table']
		self.hemisphere_ = self.maya_config_['hemisphere']
		self.file_type_ = self.maya_config_['file_type']
		self.normalize_ = self.maya_config_['normalize']

		self.focal_length_ = self.maya_config_['focal_length']
		self.app_horiz_ = self.maya_config_['app_horiz']
		self.app_vert_ = self.maya_config_['app_vert']
		self.image_width_ = self.maya_config_['image_width']
		self.image_height_ = self.maya_config_['image_height']
		self.image_format_ = self.maya_config_['image_format']
		self.output_image_width_ = self.maya_config_['output_image_width']
		self.output_image_height_ = self.maya_config_['output_image_height']

	def setup_depth(self):
		""" Sets up maya depth layers """
		if self.use_table_:
			cmd.select(self.obj_name_+":Mesh", self.plane_name_, r=True)
		else:
			cmd.select(self.obj_name_+":Mesh")
		cmd.createRenderLayer(name=self.depth_layer_name_)
		mel.eval("renderLayerBuiltinPreset linearDepth DEPTH_LAYER")
		cmd.disconnectAttr("samplerInfo1.cameraNearClipPlane", "setRange1.oldMinX")
		cmd.disconnectAttr("samplerInfo1.cameraFarClipPlane", "setRange1.oldMaxX")
		cmd.setAttr("setRange1.minX", 0)
		cmd.setAttr("setRange1.maxX", 1.0)
		cmd.setAttr("setRange1.minY", 0)
		cmd.setAttr("setRange1.maxY", 1.0)
		cmd.setAttr("setRange1.minZ", 0)
		cmd.setAttr("setRange1.maxZ", 1.0)
		cmd.setAttr("setRange1.oldMinX", self.min_range_)
		cmd.setAttr("setRange1.oldMaxX", self.max_range_)

	def setup_object_segmentation(self):
		""" Sets up maya object segmentation masks """
		self.create_mask_for_object_with_color(self.obj_name_+":Mesh", [1, 1, 1])

		if self.use_table_:
			cmd.nurbsPlane(name=self.plane_name_, p=(0,0,0), ax=(0,0,1), w=10000, lr=1, d=3, u=1, v=1, ch=1)
			self.create_mask_for_object_with_color(self.plane_name_, [1, 0, 0])

	def create_scene_with_mesh(self, mesh_filename, rot=np.eye(3)):
		""" Creates the maya scene with the specified mesh """
		cmd.file(f=True, new=True)

		if self.use_table_:
			cmd.nurbsPlane(name=self.plane_name_, p=(0,0,0), ax=(0,0,1), w=10000, lr=1, d=3, u=1, v=1, ch=1)

		try:
			cmd.file(mesh_filename, i=True, ns=self.obj_name_)
		except RuntimeError as e:
			print 'Failed to import mesh file: '+mesh_filename
			print e.message
			return False

		# convert the rotation to euler angles and rotate the mesh
		axes = 'sxyz'
		r = tfx.canonical.CanonicalRotation(rot)
		euler = r.euler(axes)
		euler = [(180. / np.pi) * a for a in euler] # convert to degrees
		cmd.select(self.obj_name_+":Mesh", r=True)
		cmd.rotate(euler[0], euler[1], euler[2])

		# move the object on top of the table if specified
		bounding_box = cmd.polyEvaluate(b=True)
		object_height = bounding_box[2][1]
		if self.use_table_:
			cmd.move(object_height, z=True)
			self.center_of_interest_ = [0, 0, object_height]

		# set the min and max distances
		if self.min_dist_ == 0:
			major_dist = math.sqrt(math.pow(bounding_box[0][0]-bounding_box[0][1], 2) + math.pow(bounding_box[1][0]-bounding_box[1][1], 2) + math.pow(bounding_box[2][0]-bounding_box[2][1], 2))
			self.min_dist_ = 1.0 * major_dist
			self.max_dist_ = 2.0 * major_dist
		
		# set image attributes	
		cmd.setAttr("defaultRenderGlobals.imageFormat", self.image_format_)
		cmd.setAttr("defaultResolution.width", self.image_width_)
		cmd.setAttr("defaultResolution.height", self.image_height_)
		cmd.setAttr("defaultResolution.deviceAspectRatio", 1.0)

		return True

	def create_mask_for_object_with_color(self, obj_name, color):
		""" Creates a segmentation mask for the specified object in the given color """
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
		""" Median filter an image """
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
		
	def render_image(self, camera_pos, camera_interest_pos, obj_key=None, mesh_filename=None):
		""" Creates a rendered image object for the mesh in |mesh_filename| """
		# setup camera
		camera_name, camera_shape = cmd.camera(p=camera_pos, wci=camera_interest_pos, wup=[0,0,1], fl=self.focal_length_, hfa=self.app_horiz_,
			vfa=self.app_vert_, ar=self.app_horiz_ / self.app_vert_)
		cmd.setAttr(camera_shape+'.backgroundColor', self.back_color_['r'], self.back_color_['g'], self.back_color_['b'], type="double3")
		cmd.setAttr(camera_shape+".renderable", 1)

		camera_rot = cmd.camera(camera_shape, q=True, rotation=True)
		pixel_width = cmd.getAttr("defaultResolution.width")
		pixel_height = cmd.getAttr("defaultResolution.height")

		focal_length_x_pixel = pixel_width * self.focal_length_ / self.app_horiz_
		focal_length_y_pixel = pixel_height * self.focal_length_ / self.app_vert_

		# render the image
		image_src = cmd.render(camera_shape)
		im = Image.open(image_src)
		im_arr = np.array(im)

		# optionally normalize the rendered image
		if self.normalize_:
			im_arr = self.normalize_image(im_arr)

		# crop the rendered images
		center = np.array(im_arr.shape) / 2.0
		start_i = center[0] - self.output_image_height_ / 2.0
		end_i = center[0] + self.output_image_height_ / 2.0
		start_j = center[1] - self.output_image_width_ / 2.0
		end_j = center[1] + self.output_image_width_ / 2.0
		im_arr = im_arr[start_i:end_i, start_j:end_j]

		# optional save
		if mesh_filename is not None:
			image_filename = mesh_filename+self.file_type_
			image_dest = path.join(self.dest_dir_, obj_key, image_filename)
			im = Image.fromarray(np.uint8(im_arr))
			im.save(image_dest)

		# create rendered image
		rendered_image = ri.RenderedImage(im_arr, camera_pos, camera_rot, camera_interest_pos)
		return rendered_image

	def render_images_for_object(self, obj_key, save_images=False, obj_name=None):
		""" Renders images from cameras on view spheres of increasing radii """
		# create radial increments
		radius = self.min_dist_
		radial_increment = 0 if self.num_radial_ == 1 else (self.max_dist_ - self.min_dist_) / (self.num_radial_ - 1)

		# set factor to toggle rendering on a view hemisphere
		mult = 1
		if self.use_table_ or self.hemisphere_:
			mult = 2

		# iterate through spherical coordinates, rendering an image from a virtual camera at each
		rendered_images = []
		for r in range(0, self.num_radial_):
			phi_increment = math.pi / (mult * (self.num_lat_ + 1))
			phi = phi_increment
			index = 0

			for lat in range(0, self.num_lat_):
				theta = 0
				theta_increment = 2 * math.pi / self.num_long_
				for lon in range(0, self.num_long_):
					camera_pos = [radius*math.sin(phi)*math.cos(theta), radius*math.sin(phi)*math.sin(theta), radius*math.cos(phi)]
					if save_images and obj_name is not None:
						mesh_name = obj_name+"_"+str(r)+"_"+str(lat)+"_"+str(lon)
						rendered_image = self.render_image(camera_pos, self.center_of_interest_, obj_key, mesh_filename=mesh_name)
					else:
						rendered_image = self.render_image(camera_pos, self.center_of_interest_, obj_key)
					rendered_image.id = index
					rendered_images.append(rendered_image)

					index += 1
					theta += theta_increment
				phi += phi_increment
			radius += radial_increment
		return rendered_images

	def render(self, graspable, dataset, render_mode='color', rot=np.eye(3), save_images=False, extra_key='', obj_filename=None):
		""" Renders the graspable according to the class config """
		# form key
		key = graspable.key + extra_key
		if save_images:
			render_dir = os.path.join(self.dest_dir_, key)
			if not os.path.exists(render_dir):
				os.mkdir(render_dir)

		# generate obj file and create scene
		if obj_filename is None:
			obj_filename = dataset.obj_mesh_filename(graspable.key)
		import_success = self.create_scene_with_mesh(obj_filename, rot)

		# render the images
		rendered_images = []
		if import_success:
			if render_mode == 'color':
				rendered_images = self.render_images_for_object(key, save_images=save_images, obj_name=graspable.key+"_color")
			elif render_mode == 'segmask':
				self.setup_object_segmentation()
				rendered_images = self.render_images_for_object(key, save_images=save_images, obj_name=graspable.key+"_segmask")
			elif render_mode == 'depth':
				self.setup_depth_layer()
				rendered_images = self.render_images_for_object(key, save_images=save_images, obj_name=graspable.key+"_depth")
			else:
				logging.warning('Render mode %s is not valid' %(render_mode))
		else:
			os.rmdir(render_dir)
			return False

		return rendered_images

if __name__ == '__main__':
	config_filename = sys.argv[1]
	config = ec.ExperimentConfig(config_filename)
	render_mode = config['maya']['render_mode']
	save_images = config['maya']['save_images']

	renderer = MayaRenderer(config)
	database_name = os.path.join(config['database_dir'], config['database_name'])
	database = db.Hdf5Database(database_name, config, access_level=db.READ_WRITE_ACCESS)

	for dataset_name in config['datasets'].keys():
		dataset = database.dataset(dataset_name)
		for obj in dataset:
			stable_poses = dataset.stable_poses(obj.key)
			for i, stable_pose in enumerate(stable_poses):
				if stable_pose.p > config['maya']['min_prob']:
					# HACK: to fix stable pose bug
					if np.abs(np.linalg.det(stable_pose.r) + 1.0) < 0.01: 
						stable_pose.r[1,:] = -stable_pose.r[1,:]

					rendered_images = renderer.render(obj, dataset, render_mode=render_mode, rot=stable_pose.r, extra_key='_stp_%d'%(i), save_images=save_images)
					dataset.store_rendered_images(obj.key, rendered_images, stable_pose_id=stable_pose.id, image_type=render_mode)
	
	database.close()				
					
