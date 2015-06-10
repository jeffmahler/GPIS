
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

from __future__ import absolute_import

__version__ = '1.0.3'

import numpy as np
from numpy.linalg import inv
from math import *
import collections
from collections import namedtuple
import numbers

_ROS = False
try:
	import roslib
	roslib.load_manifest('tf')
	import tf
	import tf.transformations as tft
	import geometry_msgs.msg
	_ROS = True
except:
	from . import transformations as tft

_FLOAT_CLOSE_ENOUGH = 0.0001

_convert_to_list = lambda data: [float(num) for num in data.flat]
_convert_to_tuple = lambda data: tuple(_convert_to_list(data))

QuaternionTuple = collections.namedtuple('QuaternionTuple', ['x','y','z','w'])

def _get_deg(kwargs):
	deg = True
	if kwargs.has_key('deg'):
		deg = kwargs['deg']
	if kwargs.has_key('rad'):
		deg = not kwargs['rad']
	return deg

if _ROS:
	_ismsginstance = lambda obj,typename: isinstance(obj,typename) or (hasattr(obj,'_type') and obj._type == typename._type)

def _copy_ypr(d):
	for key in ['yaw','pitch','roll']:
		if d.has_key(key[0]) and not d.has_key(key):
			d[key] = d[key[0]]

class tb_angles(object):
	"""Tait-Bryan angles, aka aircraft angles (intrinsic yaw pitch roll)
	
	This class is intended to simplify human interaction with rotations.
	Note that all values in this class are, by default, given and displayed
	in degrees, and that the class is immutable.
	
	QUICK USAGE:
	To convert a known physical rotation to some rotation format:
	    tb_angles(yaw,pitch,roll).{quaternion, matrix, axis_angle}
	    tb_angles(yaw,pitch,roll).msg #ROS geometry_msgs/Quaternion
	To get a string representation of a rotation (in almost any format):
	    tb_angles(rotation).tostring()
	    tb_str(rotation)
	
	OVERVIEW OF TAIT-BRYAN ANGLES
	
	Quaternions and rotation matrices are computationally efficient, but cannot
	be easily visualized by inspecting their values. Even axis-angle 
	representations can be difficult to visualize immediately.
	Additionally, given a known physical rotation, it is difficult to generate
	the quaternion or rotation matrix corresponding to that rotation.
	
	However, Tait-Bryan angles, also known as aircraft angles, are very easy to
	visualize. It is an intrinsic (moving-axis) yaw-pitch-roll (zyx) rotation.
	They are very similar to Euler angles, and in fact are often referred to as 
	Euler angles. However, the term Tait-Bryan angles is used here to clearly
	distinguish them from other forms of Euler angles.
	
	Given yaw, pitch, and roll angles, the rotation is sequentially a rotation
	about the z axis (yaw), then a rotation about the rotated y axis (pitch),
	and finally a rotation about the rotated x axis (roll). The first two
	rotations correspond to the angular axes in spherical coordinates, and the
	final rotation is about the third (radial) axis.
	
	CREATING TB_ANGLES
	
	The class can be created with angles (in degrees by default) or with a
	quaternion, matrix, or ROS message.
	
	CREATING WITH TAIT-BRYAN ANGLES
	
	Examples of creating with angles:
	tb_angles(90,45,-90)
	tb_angles(pi/2,pi/4,-pi/2,rad=True)
	tb_angles({'y': pi/2, 'p': pi/4},rad=True)
	tb_angles(yaw=90,pitch=45)                   #Note roll will be 0
	tb_angles(y=90,p=45)                         #Note roll will be 0
	
	Given an existing tb_angles, the values cannot be changed as the class is
	immutable. The with_X methods create copies with changed values:
	tb = tb_angles(yaw=90,pitch=45,roll=0)
	tb2 = tb.with_roll(-90) -> [yaw: 90, pitch: 45, roll: -90]
	
	CREATING TB_ANGLES FROM OTHER ROTATION FORMATS
	
	tb_angles can be created with the following rotation formats:
	Quaternion as numpy array, list or tuple.
	Rotation matrix as 3x3 or 4x4 (translation ignored) numpy array.
	Axis-angle, with axis as numpy array, list or tuple, and angle in radians.
	    For convenience, either order can be used.
	geometry_msgs/Quaternion or any geometry_msgs message with a 
	    Quaternion field.

	CONVERSION TO ROTATION FORMATS
	
	tb_angles.quaternion
	tb_angles.matrix
	The quaternion and rotation matrix representations can be obtained with the
	attributes tb_angles.quaternion and tb_angles.matrix. These are read-only
	arrays. Writeable copies can be obtained using tb_angles.to_quaternion()
	and tb_angles.to_matrix()
	
	tb_angles.to_tf()   Returns a 4x4 transformation matrix.
	tb_angles.to_pose() is identical, but is included for convenience.
	
	tb_angles.msg Returns a new geometry_msgs/Quaternion filled in with the
	    quaternion for the rotation.
	
	tb_angles.axis_angle Returns the axis-angle rotation as a tuple of 
	    (axis,angle), with the axis as a numpy array and the angle in radians
	
	STRING OUTPUT
	
	The tostring() method has various options for formatting the string output.
	The module function tb_str(*args,**kwargs) is provided to convert almost any
	    rotation format into a tb string.
	    It is equivalent to tb_angles(*args).tostring(**kwargs)
	
	By default, the values are shown in degrees, without units.
	The keyword arguments 'deg' and 'rad' can be used to display the values in
	degrees and radians, with units:
	tb_angles.tostring():        values in degrees without units
	tb_angles.tostring(deg=True) values in degrees    with units
	tb_angles.tostring(rad=True) values in radians    with units
	tb_angles.tostring(deg=True,rad=True) will print values in both degrees
	    and radians
	
	The keyword argument 'fixed_width' will cause the output to use fixed-width
	fields for the values, which can be useful if many values are being
	printed in succession.
	
	The keyword argument 'short_names' causes the field names to be abbreviated
	to one letter, e.g., 'y' instead of 'yaw'
	
	The initial letter of the keyword arguments can be used with string
	formatting.
	Example: '{0:fs}'.format(tb) to get fixed_width and short_names
	
	CONVENIENCE SHORTCUT FUNCTIONS
	
	For convenience, the following functions are included in the module:
	tb_str(*args,**kwargs)
	    Equivalent to tb_angles(*args).tostring(**kwargs)
	
	tb_to_quat(*args,**kwargs)
	    Equivalent to tb_angles(*args,**kwargs).to_quaternion()
	tb_to_mat(*args,**kwargs)
	    Equivalent to tb_angles(*args,**kwargs).to_matrix()
	tb_to_tf(*args,**kwargs)
	    Equivalent to tb_angles(*args,**kwargs).to_tf()
	tb_to_pose(*args,**kwargs)
	    Equivalent to tb_angles(*args,**kwargs).to_pose()
	
	get_tb_angles(*args,**kwargs)
	    Null-safe creation of tb_angles
	"""
	@property
	def yaw_deg(self):
		""" The yaw angle in degrees """
		return self._yaw
	
	@property
	def yaw_rad(self):
		""" The yaw angle in radians """
		return self._yaw * pi / 180.
	
	@property
	def pitch_deg(self):
		""" The pitch angle in degrees """
		return self._pitch
	
	@property
	def pitch_rad(self):
		""" The pitch angle in radians """
		return self._pitch * pi / 180.
	
	@property
	def roll_deg(self):
		""" The roll angle in degrees """
		return self._roll
	
	@property
	def roll_rad(self):
		""" The roll angle in radians """
		return self._roll * pi / 180.
	
	def with_yaw(self,yaw,**kwargs):
		"""Return a new tb_angles with the specified yaw.
		
		Args:
		    yaw (float): The yaw (in degrees by default)
		    deg (bool): Optional, whether to interpret input as degrees
		    rad (bool): Optional, whether to interpret input as radians
		
		Returns:
		    A new tb_angles with the input yaw and the existing pitch and roll
		"""
		deg = _get_deg(kwargs)
		if deg:
			return tb_angles(yaw,self.pitch_deg,self.roll_deg,deg=True)
		else:
			return tb_angles(yaw,self.pitch_rad,self.roll_rad,rad=True)
	
	def with_pitch(self,pitch,**kwargs):
		"""Return a new tb_angles with the specified pitch.
		
		Args:
		    pitch (float): The pitch (in degrees by default)
		    deg (bool): Optional, whether to interpret input as degrees
		    rad (bool): Optional, whether to interpret input as radians
		
		Returns:
		    A new tb_angles with the input pitch and the existing yaw and roll
		"""
		deg = _get_deg(kwargs)
		if deg:
			return tb_angles(self.yaw_deg,pitch,self.roll_deg,deg=True)
		else:
			return tb_angles(self.yaw_rad,pitch,self.roll_rad,rad=True)
	
	def with_roll(self,roll,**kwargs):
		"""Return a new tb_angles with the specified roll.
		
		Args:
		    roll (float): The roll (in degrees by default)
		    deg (bool): Optional, whether to interpret input as degrees
		    rad (bool): Optional, whether to interpret input as radians
		
		Returns:
		    A new tb_angles with the input roll and the existing yaw and pitch
		"""
		deg = _get_deg(kwargs)
		if deg:
			return tb_angles(self.yaw_deg,self.pitch_deg,roll,deg=True)
		else:
			return tb_angles(self.yaw_rad,self.pitch_rad,roll,rad=True)
	
	@property
	def quaternion(self):
		"""The quaternion corresponding to this rotation"""
		if self._quaternion is None:
			self._quaternion = tft.quaternion_from_matrix(self.to_tf())
			if self._quaternion[3] < 0:
				self._quaternion = -self._quaternion
			self._quaternion.flags.writeable = False
		return self._quaternion
	
	def to_quaternion(self):
		"""Get the quaternion corresponding to this rotation.
		
		Returns:
		    A quaternion as a numpy 1-D array
		"""
		return self.quaternion.copy()
	
	def to_quaternion_list(self):
		"""Get the quaternion corresponding to this rotation as a list.
		
		Returns:
		    A quaternion as a list
		"""
		return _convert_to_list(self.quaternion)
	
	def to_quaternion_tuple(self):
		return _convert_to_tuple(self.quaternion)
		"""Get the quaternion corresponding to this rotation as a tuple.
		
		Returns:
		    A quaternion as a namedtuple with x, y, z, and w fields
		"""
		return QuaternionTuple._make(_convert_to_tuple(self.quaternion))
	
	if _ROS:
		def to_quaternion_msg(self):
			"""Get the quaternion corresponding to this rotation as a ROS message.
			
			Returns:
			    A quaternion as a geometry_msgs/Quaternion
			"""
			q = self.quaternion
			return geometry_msgs.msg.Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])
		
		@property
		def msg(self):
			"""The quaternion as a ROS message"""
			return self.to_quaternion_msg()
	
	@property
	def rotation_matrix(self):
		"""The rotation corresponding to this rotation"""
		if self._matrix is None:
			yaw = self.yaw_rad
			pitch = self.pitch_rad
			roll = self.roll_rad
			Ryaw = np.matrix(
					[[cos(yaw), -sin(yaw), 0],
					[sin(yaw),  cos(yaw), 0],
					[0,		 0,		1]])
			Rpitch = np.matrix(
					 [[cos(pitch), 0, sin(pitch)],
					 [0,		  1, 0],
					[-sin(pitch), 0, cos(pitch)]])
			Rroll = np.matrix(
					[[1,  0,		  0],
					[0,  cos(roll), -sin(roll)],
					[0,  sin(roll),  cos(roll)]])
			self._matrix = Ryaw * Rpitch * Rroll
			self._matrix.flags.writeable = False
		return self._matrix
	
	
	matrix = rotation_matrix
		
	def to_matrix(self):
		"""Get the rotation matrix corresponding to this rotation.
		
		Returns:
		    A 3x3 numpy matrix
		"""
		return self.matrix.copy()
	
	def to_rotation_matrix(self):
		"""Get the rotation matrix corresponding to this rotation.
		
		Returns:
		    A 3x3 numpy matrix
		"""
		return self.rotation_matrix.copy()
	
	@property
	def axis_angle(self):
		"""The axis and angle for this rotation"""
		if all(np.absolute([self.yaw_deg,self.pitch_deg,self.roll_deg]) < _FLOAT_CLOSE_ENOUGH):
			return np.array([1.,0.,0.]), 0.
		
		q = self.quaternion
		xyz_norm = np.linalg.norm(q[0:3])
		axis = np.empty(3)
		axis[:] = q[0:3] / xyz_norm
		angle = 2. * atan2(xyz_norm, q[3])
		return axis,angle
	
	def to_axis_angle(self):
		"""Get the axis-angle representation of this rotation
		
		Returns:
		    A tuple of axis (as numpy 1-D array) and angle in radians
		"""
		return self.axis_angle
	
	def to_tf(self):
		"""Get the transformation matrix corresponding to this rotation.
		
		Returns:
		    A 4x4 numpy matrix
		"""
		tf = np.mat(np.identity(4))
		tf[0:3,0:3] = self.matrix
		return tf
	def to_pose(self):
		"""Get the pose matrix corresponding to this rotation.
		
		Returns:
		    A 4x4 numpy matrix
		"""
		tf = np.mat(np.identity(4))
		tf[0:3,0:3] = self.matrix
		return tf
	
	def rotation_to(self,*args,**kwargs):
		"""Get the rotation (as tb_angles) which, when applied to this rotation
		by right-multiplication, produces the input rotation"""
		other = tb_angles(*args,**kwargs)
		out = tb_angles(tft.quaternion_multiply(tft.quaternion_inverse(self.quaternion), other.quaternion))
		return out
	
	def rotation_from(self,*args,**kwargs):
		"""Get the rotation (as tb_angles) which, when applied to the input rotation
		by right-multiplication, produces this rotation"""
		other = tb_angles(*args,**kwargs)
		out = tb_angles(tft.quaternion_multiply(tft.quaternion_inverse(other.quaternion), self.quaternion))
		return out
	
	def __init__(self,*args,**kwargs):
		"""Construct tb_angles from angles or from another rotation format.
		
		The class can be created with angles (in degrees by default) or with a
		quaternion, matrix, or ROS message.
		
		Examples of creating with angles:
		tb_angles(90,45,-90)
		tb_angles(pi/2,pi/4,-pi/2,rad=True)
		tb_angles({'y': pi/2, 'p': pi/4},rad=True)
		tb_angles(yaw=90,pitch=45)                   #Note roll will be 0
		tb_angles(y=90,p=45)                         #Note roll will be 0
		
		tb_angles can be created with the following rotation formats:
		Quaternion as numpy array, list or tuple.
		Rotation matrix as 3x3 or 4x4 (translation ignored) numpy array.
		Axis-angle, with axis as numpy array, list or tuple, and angle in radians.
		    The order does not matter.
		geometry_msgs/Quaternion or any geometry_msgs message with a 
		    Quaternion field.
		"""
		self._matrix = None
		self._quaternion = None
		
		
		deg = _get_deg(kwargs)
		if deg:
			conv = 1.
		else:
			conv = 180. / pi
		
		def set_direct(yaw,pitch,roll):
			self._yaw = yaw * conv
			self._pitch = pitch * conv
			self._roll = roll * conv
		
		if not args:
			_copy_ypr(kwargs)
			set_direct(kwargs.get('yaw',0.),kwargs.get('pitch',0.),kwargs.get('roll',0.))
			return
		elif len(args) == 3:
			set_direct(args[0],args[1],args[2])
			return
		elif len(args) == 0 and isinstance(args[0],tb_angles):
			self._yaw = args[0]._yaw
			self._pitch = args[0]._pitch
			self._roll = args[0]._roll
			return
		
		R = None
		if len(args) == 1 and isinstance(args[0],tb_angles):
			R = args[0].matrix
		if len(args) == 2 or (len(args) == 1 and isinstance(args[0],collections.Sequence) and len(args[0]) == 2):
			if len(args) == 2:
				val1 = args[0]
				val2 = args[1]
			else:
				val1 = args[0][0]
				val2 = args[0][1]
			if isinstance(val1,numbers.Number) and isinstance(val2,collections.Sequence) and len(val2) == 3:
				axis,angle = (np.array(val2),val1)
			elif isinstance(val2,numbers.Number) and isinstance(val1,collections.Sequence) and len(val1) == 3:
				axis,angle = (np.array(val1),val2)
			elif isinstance(val1,numbers.Number) and isinstance(val2,np.ndarray) and val2.size == 3:
				axis,angle = (val2.reshape((3,)),val1)
			elif isinstance(val2,numbers.Number) and isinstance(val1,np.ndarray) and val1.size == 3:
				axis,angle = (val1.reshape((3,)),val2)
			else:
				raise ValueError("Unknown data for tb_angles: (%s, %s)" % args)
			R = tft.quaternion_matrix(tft.quaternion_about_axis(angle, axis))[0:3,0:3]
		elif len(args) > 1:
			raise ValueError("Unknown data for tb_angles: %s" % str(args))
		else:
			data = args[0]
			
			if isinstance(data,dict):
				_copy_ypr(data)
				set_direct(data.get('yaw',0.),data.get('pitch',0.),data.get('roll',0.))
				return
			
			if _ROS and _ismsginstance(data,geometry_msgs.msg.QuaternionStamped):
				data = data.quaternion
			elif _ROS and _ismsginstance(data,geometry_msgs.msg.Pose):
				data = data.orientation
			elif _ROS and _ismsginstance(data,geometry_msgs.msg.PoseStamped):
				data = data.pose.orientation
			elif _ROS and _ismsginstance(data,geometry_msgs.msg.Transform):
				data = data.rotation
			elif _ROS and _ismsginstance(data,geometry_msgs.msg.TransformStamped):
				data = data.transform.rotation
			
			if _ROS and _ismsginstance(data,geometry_msgs.msg.Quaternion):
				R = tft.quaternion_matrix([data.x,data.y,data.z,data.w])[0:3,0:3]
			elif isinstance(data,np.ndarray) and data.size == 4:
				R = tft.quaternion_matrix(data)[0:3,0:3]
			elif isinstance(data,np.ndarray) and (data.shape == (3,3) or data.shape == (4,4)):
				R = np.mat(np.empty((3,3)))
				R[:,:] = data[0:3,0:3]
			elif isinstance(data,collections.Sequence):
				if (len(data) == 3 and all([isinstance(d, collections.Sequence) and len(d) == 3 for d in data])) \
						or (len(data) == 4 and all([isinstance(d, collections.Sequence) and len(d) == 4 for d in data])):
					R = np.matrix(data)[0:3,0:3]
				elif len(data) == 4 and all([isinstance(d, numbers.Number) for d in data]):
					R = tft.quaternion_matrix(data)[0:3,0:3]
				
			if R is None:
				raise ValueError("Unknown data for tb_angles: %s" % data)
			
		yaw = 0;
		pitch = 0;
		roll = 0;
		
		skip = False
		if fabs(R[0,1]-R[1,0]) < _FLOAT_CLOSE_ENOUGH and fabs(R[0,2]-R[2,0]) < _FLOAT_CLOSE_ENOUGH and fabs(R[1,2]-R[2,1]) < _FLOAT_CLOSE_ENOUGH:
			#matrix is symmetric
			if fabs(R[0,1]+R[1,0]) < _FLOAT_CLOSE_ENOUGH and fabs(R[0,2]+R[2,0]) < _FLOAT_CLOSE_ENOUGH and fabs(R[1,2]+R[2,1]) < _FLOAT_CLOSE_ENOUGH:
				#diagonal
				if R[0,0] > -_FLOAT_CLOSE_ENOUGH:
					if R[1,1] > -_FLOAT_CLOSE_ENOUGH:
						skip = True
					else:
						roll = pi;
				elif R[1,1] > -_FLOAT_CLOSE_ENOUGH:
					pitch = pi;
				else:
					yaw = pi;
				skip=True
		
		if not skip:
			vx = R[0:3,0:3] * np.matrix([1,0,0]).transpose();
			vy = R[0:3,0:3] * np.matrix([0,1,0]).transpose();
	
			yaw = atan2(vx[1,0],vx[0,0]);
			pitch = atan2(-vx[2,0], sqrt(vx[0,0]*vx[0,0] + vx[1,0]*vx[1,0]));
	
			Ryaw = np.matrix(
						 [[cos(yaw), -sin(yaw), 0],
						 [sin(yaw),  cos(yaw), 0],
						 [0,		 0,		1]]);
			Rpitch = np.matrix(
					 [[cos(pitch), 0, sin(pitch)],
					 [0,		  1, 0],
					[-sin(pitch), 0, cos(pitch)]]);
			vyp = Ryaw * Rpitch * np.matrix([0,1,0]).transpose();
			vzp = Ryaw * Rpitch * np.matrix([0,0,1]).transpose();
	
			if vzp.transpose() * vy >= 0:
				coeff = 1
			else:
				coeff = -1
	
			val = vyp.transpose() * vy
			if val > 1:
				val = 1
			elif val < -1:
				val = -1
			roll = coeff * acos(val);
		
		self._yaw = yaw * 180. / pi;
		self._pitch = pitch * 180. / pi;
		self._roll = roll * 180. / pi;
	
	def tostring(self,**kwargs):
		"""Get the string representation of this rotation.
		
		The initial letter of the keyword arguments can be used with string
		formatting.
		Example: '{0:fs}'.format(tb) to get fixed_width and short_names
		
		Args:
		    fixed_width (bool): Optional, whether to print the values in a  
		        fixed width format (useful when printing many values to the 
		        screen)
		    short_names (bool): Optional, whether to abbreviate the field
		        names with single letters, e.g. 'y' for 'yaw'
		    deg (bool): Optional, whether to print the fields in degrees.
		        Degrees are printed by default, but if the field is not
		        specified, no units will be printed. Setting deg = True
		        will cause the units to be included.
		    rad (bool): Optional, whether to print the fields in radians
		Returns:
		    The string representation of the object.
		"""
		deg = None
		rad = None
		default_deg = False
		
		if kwargs.has_key('deg'):
			deg = kwargs['deg']
		
		if kwargs.has_key('rad'):
			rad = kwargs['rad']
		
		if deg is None and rad is None:
			deg = True
			default_deg = True
		
		if deg is False and rad is False:
			raise ValueError('Cannot set both deg and rad to False!')
		elif deg is None and rad is False:
			deg = True
		elif deg is False and rad is None:
			rad = True
		
		fixed_width = kwargs.get('fixed_width',False)
		
		if fixed_width:
			deg_fmt = '% 6.1f'
			rad_fmt = '% 6.3f rad'
		else:
			deg_fmt = '%.1f'
			rad_fmt = '%.3f rad'
		if not default_deg:
			deg_fmt += ' deg'
		
		short_names = kwargs.get('short_names',False)
		
		brackets = kwargs.get('brackets','[]') 
		
		colon = ':'
		
		s = ''
		if brackets:
			s += brackets[0]

		if short_names:
			s += 'y'
		else:
			s += 'yaw'
		s += colon
		if deg:
			s += deg_fmt % self.yaw_deg
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % self.yaw_rad
			if deg:
				s += ')'
		
		s += ', '
		if short_names:
			s += 'p'
		else:
			s += 'pitch'
		s += colon
		if deg:
			s += deg_fmt % self.pitch_deg
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % self.pitch_rad
			if deg:
				s += ')'
	
		s += ', '
		if short_names:
			s += 'r'
		else:
			s += 'roll'
		s += colon
		if deg:
			s += deg_fmt % self.roll_deg
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % self.roll_rad
			if deg:
				s += ')'
		if brackets:
			s += brackets[1]
		
		return s
	
	def __str__(self):
		return self.tostring()
	
	def __format__(self,spec):
		kwargs = {}
		for key in ['deg', 'rad', 'fixed_width', 'short_names']:
			if key[0] in spec:
				kwargs[key] = True
		if 'p' in spec:
			kwargs['brackets'] = '()'
		return self.tostring(**kwargs)
	
	def __repr__(self):
		return 'tb_angles(%f, %f, %f)' % (self.yaw_deg, self.pitch_deg, self.roll_deg)
	
	def __copy__(self):
		return tb_angles(self._yaw,self._pitch,self._roll,deg=True)

def tb_str(*args,**kwargs):
	"""Construct a string showing the Tait-Bryan angles (yaw-pitch-roll) of 
	    the given rotation"""
	if len(args) == 1 and isinstance(args[0],tb_angles):
		tb = args[0]
	else:
		tb = get_tb_angles(*args)
	
	return tb.tostring(**kwargs)

def tb_to_quat(*args,**kwargs):
	"""Convert Tait-Bryan angles (intrinsic yaw-pitch-roll) in degrees to quaternion"""
	if len(args) == 1 and isinstance(args[0],tb_angles):
		tb = args[0]
	else:
		tb = get_tb_angles(*args,**kwargs)
	
	return tb.to_quaternion()

def tb_to_mat(*args,**kwargs):
	"""Convert Tait-Bryan angles (intrinsic yaw-pitch-roll) in degrees to rotation matrix"""
	if len(args) == 1 and isinstance(args[0],tb_angles):
		tb = args[0]
	else:
		tb = get_tb_angles(*args,**kwargs)
	
	return tb.to_matrix()

def tb_to_tf(*args,**kwargs):
	"""Convert Tait-Bryan angles (intrinsic yaw-pitch-roll) in degrees to transform matrix"""
	if len(args) == 1 and isinstance(args[0],tb_angles):
		tb = args[0]
	else:
		tb = get_tb_angles(*args,**kwargs)
	
	return tb.to_tf()

def tb_to_pose(*args,**kwargs):
	"""Convert Tait-Bryan angles (intrinsic yaw-pitch-roll) in degrees to pose matrix"""
	if len(args) == 1 and isinstance(args[0],tb_angles):
		tb = args[0]
	else:
		tb = get_tb_angles(*args,**kwargs)
	
	return tb.to_pose()

def get_tb_angles(*args,**kwargs):
	"""Null-safe creation of tb_angles"""
	if len(args) == 1 and (isinstance(args[0],tb_angles) or args[0] is None):
		return args[0]
	return tb_angles(*args,**kwargs)
