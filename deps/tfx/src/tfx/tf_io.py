
# Copyright 2015 Ben Kehoe
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import roslib
roslib.load_manifest('tfx')
import rospy
import tf
from tfx.tb_angles import *
import tfx.yaml as yaml
from tfx.canonical import *
import collections
import re

__version__ = '0.1'

_str_prep = lambda val: re.sub(r' +',' ',re.sub(r'[\[\]\(\),]',' ',str(val))).strip()
_str_parse = lambda l: _str_prep(l).split(' ')

def _parse_pos(data):
	pos = None
	if data.has_key('pos'):
		pos = data['pos']
	elif data.has_key('p'):
		pos = data['p']		
	elif data.has_key('translation'):
		pos = data['translation']
	return pos

def _parse_rot(data):
	rot = None
	if data.has_key('q'):
		rot = data['q']
	elif data.has_key('tb'):
		tb = data['tb']
		rot = tft.quaternion_from_matrix(tb_to_mat(tb[0],tb[1], tb[2]))
	elif data.has_key('tb_deg'):
		tb = data['tb_deg']
		rot = tft.quaternion_from_matrix(tb_to_mat(tb[0],tb[1], tb[2]))
	elif data.has_key('tb_rad'):
		tb = data['tb_rad']
		rot = tft.quaternion_from_matrix(tb_to_mat(tb[0] * 180 / pi,tb[1] * 180 / pi, tb[2] * 180 / pi))
	elif data.has_key('rotation'):
		rot = data['rotation']
	return rot

class TfData:
	def __init__(self,tf=None,from_frame=None,to_frame=None,stamp=None):
		self.from_frame = from_frame
		self.to_frame = to
		self.transform = tf
		self.stamp = stamp
	
	def get_data(self,ros=False):
		if ros: return self.get_ros_data()
		data = {'from': self.from_frame,
				'to': self.to_frame,
				'pos': list(get_tf_translation_tuple(self.transform))}
		q = get_tf_quaternion_tuple(self.transform)
		tb = get_tb_angles(q)
		data['q'] = list(q)
		data['tb'] = [tb.yaw_deg,tb.pitch_deg,tb.roll_deg]
		#data['tb_deg'] = [tb.yaw_deg,tb.pitch_deg,tb.roll_deg]
		#data['tb_rad'] = [tb.yaw,tb.pitch,tb.roll]
		
		if self.stamp:
			data['stamp'] = self.stamp.to_sec()
		return data

	def get_ros_data(self):
		p = get_tf_translation_tuple(self.transform)
		q = get_tf_quaternion_tuple(self.transform)
		data = {'translation':{'x':p[0], 'y':p[1], 'z':p[2]},
				'rotation':{'x':q[0],'y':q[1],'z':q[2],'w':q[3]}}
		if self.from_frame:
			if self.stamp:
				stamp = {'secs':self.stamp.secs,'nsecs':self.stamp.nsecs}
			else:
				stamp = {'secs':0,'nsecs':0}
			data = {'header': {'seq':0,'stamp':stamp,'frame_id':self.from_frame},'child_frame_id':self.to_frame,'transform':data}
		return data
	
	@staticmethod
	def load_from_data(data,frames_from_to=None,use_stamp=False):
		obj = TfData()
		obj.populate(data,frames_from_to,use_stamp)
		return obj
		
	
	def populate(self,data,frames_from_to=None,use_stamp=False):
		if data.has_key('header'):
			if data['header'].has_key('frame_id'):
				self.from_frame = data['header']['frame_id']
			if use_stamp and data['header'].has_key('stamp'):
				self.stamp = rospy.Time(data['header']['stamp']['secs'],data['header']['stamp']['nsecs'])
			if data.has_key('transform'):
				self.populate(data['transform'],frames_from_to,use_stamp)
				if data.has_key('child_frame_id'):
					self.to_frame = data['child_frame_id']
				else:
					raise Exception('No child_frame_id in transform')
				return
		
		for key in ['from','frame','frame_id']:
			if data.has_key(key):
				self.from_frame = data[key]
				break
		
		for key in ['to','child','child_frame','child_frame_id']:
			if data.has_key(key):
				self.to_frame = data[key]
				break
		
		pos = _parse_pos(data)
		rot = _parse_rot(data)
		
		if rot is None and data.has_key('matrix'):
			print 'matrix not implemented yet'
		
		if use_stamp and data.has_key('stamp'):
			self.stamp = rospy.Time(data['stamp'])
		
		if frames_from_to:
			self.from_frame = frames_from_to[0]
			self.to_frame = frames_from_to[1]

		self.transform = transform(pos,rot,parent=self.from_frame,child=self.to_frame,stamp=self.stamp,pose=False)

class PoseData:
	def __init__(self,pose=None,frame=None,name=None,stamp=None):
		self.frame = frame
		self.name = name
		self.pose = pose
		self.stamp = stamp
	
	def get_data(self,ros=False):
		if ros: return self.get_ros_data()
		data = {'frame': self.frame,'pos': list(get_tf_translation_tuple(self.pose))}
		if self.name:
			data['name'] = self.name
		q = get_tf_quaternion_tuple(self.pose)
		tb = get_tb_angles(q)
		data['q'] = list(q)
		data['tb'] = [tb.yaw_deg,tb.pitch_deg,tb.roll_deg]
		#data['tb_deg'] = [tb.yaw_deg,tb.pitch_deg,tb.roll_deg]
		#data['tb_rad'] = [tb.yaw,tb.pitch,tb.roll]
		
		if self.stamp:
			data['stamp'] = self.stamp.to_sec()
		return data
	
	def get_ros_data(self):
		p = get_tf_translation_tuple(self.transform)
		q = get_tf_quaternion_tuple(self.transform)
		data = {'position':{'x':p[0], 'y':p[1], 'z':p[2]},
				'orientation':{'x':q[0],'y':q[1],'z':q[2],'w':q[3]}}
		if self.frame:
			if self.stamp:
				stamp = {'secs':self.stamp.secs,'nsecs':self.stamp.nsecs}
			else:
				stamp = {'secs':0,'nsecs':0}
			data = {'header': {'seq':0,'stamp':stamp,'frame_id':self.frame},'pose':data}
		return data
	
	@staticmethod
	def load_from_data(data,frame=None,name=None,use_stamp=False):
		obj = PoseData();
		obj.populate(data,frame,name,use_stamp)
		return obj
	
	def populate(self,data,frame=None,name=None,use_stamp=False,clear=False):
		if clear:
			self.frame = None
			self.name = None
			self.pose = None
			self.stamp = None
		if data.has_key('header'):
			if data['header'].has_key('frame_id'):
				self.from_frame = data['header']['frame_id']
			if use_stamp and data['header'].has_key('stamp'):
				self.stamp = rospy.Time(data['header']['stamp']['secs'],data['header']['stamp']['nsecs'])
			if data.has_key('pose'):
				self.populate(data['pose'],frames_from_to,use_stamp)
				return
		
		for key in ['frame','frame_id']:
			if data.has_key(key):
				self.frame = data[key]
				break
		
		if data.has_key('name'):
			self.name = data['name']

		pos = _parse_pos(data)
		rot = _parse_rot(data)

		if rot is None and data.has_key('matrix'):
			print 'matrix not implemented yet'

		if use_stamp and data.has_key('stamp'):
			self.stamp = rospy.Time(data['stamp'])
		
		if frame:
			self.frame = frame
		if name:
			self.name = name;
			
		self.transform = transform(pos,rot,frame=self.frame,name=self.name,stamp=self.stamp,pose=True)
		
		

class TfFileData:
	def __init__(self):
		self.tf = None
		self.pose = None
	
	def populate_from_listener(self,from_frame,to_frame,listener=None,set_pose=None,use_stamp=False):
		if listener is None:
			listener = tf.TransformListener()
		
		listener.waitForTransform(to_frame,from_frame,rospy.Time(),rospy.Duration(10))
		(pos,q) = listener.lookupTransform(from_frame,to_frame,rospy.Time())
		if use_stamp:
			now = rospy.Time.now()
		else:
			now = None
		
		self.set_tf(from_frame,to_frame,(pos,q),stamp=now)
		
		if set_pose:
			if isinstance(set_pose,str):
				pose_frame = set_pose
			else:
				pose_frame = from_frame
			self.set_pose(frame=pose_frame,name=to_frame,pose=(pos,q),stamp=now)
		
	
	def load(self,file,tf_frames_from_to=None,pose_frame=None,use_stamp=False,pose_frame_default_to_tf_parent=False):
		if isinstance(file,basestring):
			f = open(file)
		else:
			f = file
		all_data = yaml.load(f)
		if isinstance(file,basestring):
			f.close()
		
		self.populate(all_data,tf_frames_from_to,pose_frame,use_stamp,pose_frame_default_to_tf_parent)
	
	def write(self,file,append=False,write_only=None,ros=False):
		if append:
			mode = 'a'
		else:
			mode = 'w'
		data = self.get_data(ros=ros)
		if ros and not write_only and self.tf and self.pose:
			raise Exception('ROS format cannot write tf and pose to the same file!')
		if write_only == 'tf':
			data = data['tf']
		elif write_only == 'pose':
			data = data['pose']

		if ros:
			data = data[data.keys()[0]]

		if isinstance(file,basestring):
			f = open(file,mode)
		else:
			f = file
		if append:
			f.write('---\n')
		f.write(yaml.dump(data))
		if isinstance(file,basestring):
			f.close()
	
	def set_tf(self,from_frame,to_frame,transform,stamp=None,set_pose=False,default_pose_name_to_frame=False):
		transform = transform(transform)
		self.tf = TfData()
		self.tf.from_frame = from_frame
		self.tf.to_frame = to_frame
		self.tf.transform = transform
		self.tf.stamp = stamp
		if set_pose:
			self.set_pose(from_frame,transform,stamp)
			if not self.pose.name and default_pose_name_to_frame:
				self.pose.name = to_frame
	
	def set_pose(self,frame,pose,stamp=None,name=None):
		pose = transform(pose)
		self.pose = PoseData()
		self.pose.frame = frame
		self.pose.pose = pose
		self.pose.stamp = stamp
		self.pose.name = name
	
	def set_tf_transform(self,transform,stamp=None):
		self.tf.transform = transform
		if stamp:
			self.tf.stamp = stamp
	
	def set_pose_pose(self,pose,stamp=None):
		pose = transform(pose)
		self.pose.pose = pose
		if stamp:
			self.pose.stamp = stamp
	
	def set_transform_and_pose(self,transform,stamp=None):
		transform = transform(transform)
		if self.tf:
			self.set_tf_transform(transform,stamp)
		if self.pose:
			self.set_pose_pose(transform,stamp)
	
	def set_tf_stamp(self,stamp=None):
		if stamp is None:
			stamp = rospy.Time.now()
		self.tf.stamp = stamp
	
	def set_pose_stamp(self,stamp=None):
		if stamp is None:
			stamp = rospy.Time.now()
		self.pose.stamp = stamp
	
	def set_stamp(self,stamp=None):
		self.set_tf_stamp(stamp)
		self.set_pose_stamp(stamp)

	def clear_stamp(self):
		self.set_stamp(None)
	
	def get_data(self,ros=False):
		data = {}
		if self.tf:
			data['tf'] = self.tf.get_data(ros=ros)
		if self.pose:
			data['pose'] = self.pose.get_data(ros=ros)
		return data
	
	def populate(self,data,tf_frames_from_to=None,pose_frame=None,pose_name=None,use_stamp=False,
			pose_frame_default_to_tf_parent=False,pose_name_default_to_tf_child=False):
		if not data.has_key('tf') and not data.has_key('pose'):
			if not self.tf: self.tf = TfData()
			if not self.pose: self.pose = PoseData()
			if not data.has_key('pose'):
				self.tf.populate(data,tf_frames_from_to,use_stamp)
			if not data.has_key('transform'):
				self.pose.populate(data,pose_frame,pose_name,use_stamp)
		else:
			if data.has_key('tf'):
				if not self.tf: self.tf = TfData()
				self.tf.populate(data['tf'],tf_frames_from_to,use_stamp)
			if data.has_key('pose'):
				if not self.pose: self.pose = PoseData()
				self.pose.populate(data['pose'],pose_frame,pose_name,use_stamp)
		if self.pose and self.tf:
			if not self.pose.frame and pose_frame_default_to_tf_parent:
				self.pose.frame = self.tf.from_frame
			if not self.pose.name  and pose_name_default_to_tf_child:
				self.pose.name = self.tf.to_frame

def dump_example():
	example_filename = 'example_tf.yaml'

	content = \
"""# Three examples of valid file contents.
# stamp is not a required field
# from/to and frame/name are not required
#   if provided by the loading function
---
from: /world
to: /robot
stamp: 1345164012.2246015
pos: [-0.0725, 0.1037, -0.1173]
q: [-0.0387, 0.9243, -0.3776, 0.03818] #rotation as quaternion
---
tf:
  from: /world
  to: /robot
  stamp: 1345164012.2246015
  pos: [-0.0725, 0.1037, -0.1173]
  tb: [-174.2, 2.3, -135.4] #rotation as Tait-Bryan angles (yaw pitch roll) in degrees
---
pose:
  frame: /world
  name: robot
  stamp: 1345164012.2246015
  pos: [-0.0725, 0.1037, -0.1173]
  tb_rad: [-3.040, 0.041, -2.363] #rotation as Tait-Bryan angles in radians
"""
	
	print '**** EXAMPLE TF FILE:'
	print content
	print '**** EXAMPLE SAVED TO %s' % example_filename
	
	try:
		f = open(example_filename,'w')
		f.write(content)
		f.close()
	except Exception, e:
		print 'Writing example file failed:',e
		return 1
	return 0
