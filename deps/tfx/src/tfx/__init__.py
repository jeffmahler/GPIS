
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

"""The tfx module provides convenience classes for working with transformations.
The main classes are tb_angles, which represents rotations as easily-understood
Tait-Bryan angles (aircraft angles), and the classes from the canonical module.

Additionally, importing the tfx module installs a singleton factory on
tf.TransformListener, which frees up code from having to pass around a listener,
since only one instance can exist in a node. This singleton can be accessed
through the function tfx.TransformListener(), providing a drop-in replacement
in existing code. The module also provides convenience functions for 
commonly-used operations on TransformListener.

tb_angles exists to make it easier to view and create rotations. Most rotation
representations, such as quaternions, rotation matrices, and even axis-angle
are hard to visualize directly. However, Tait-Bryan angles, also known as 
aircraft angles, are very easy to visualize. It is an intrinsic (moving-axis)
yaw-pitch-roll (zyx) rotation. They are very similar to Euler angles, and in 
fact are often referred to as Euler angles. However, the term Tait-Bryan angles
is used here to clearly distinguish them from other forms of Euler angles.
Further documentation is available in the class doc.

The canonical module contains several classes that have the purpose of making
it easier to convert between formats. There are five main classes:
CanonicalStamp (timestamps)
CanonicalDuration (time duration)
CanonicalPoint (3D point or vector with optional frame and timestamp)
CanonicalRotation (3D rotation with optional frame and timestamp)
CanonicalTransform (3D transform or pose)

Each class is capable of being created from a variety of formats, and each has
methods for converting back to these formats.

The classes should not be created directly. The following functions, which are
imported into the tfx namespace, provide None-safe creation, error checking,
and, in the case of transform() and pose(), the ability to convert ROS message
types that contain arrays (PoseArray and tfMessage) into lists of 
CanonicalTransforms:
stamp()
duration()
time(): allows conversion to stamp or duration based on input
point() or vector()
rotation() or quaternion(): equivalent, both provided for convenience
transform() and pose(): nearly equivalent, see below.

Two methods 'rotation_tb' and 'rotation_euler' are provided for creating
CanonicalRotations from Tait-Bryan and Euler angles, respectively.

Methods for creating identity transforms, poses, and rotations, and random
points, rotations, transforms and poses are also provided.

CanonicalTransform represents both transforms and poses. The transform from
frame A to frame B is equal to the pose of frame A's axes in frame B. This
equivalence is used in CanonicalTransform. Whether the object is a pose or a
transform is stored in the object's 'pose' attribute.
There are two frame fields on CanonicalTransform, each of which has multiple
aliases:
'parent' and 'frame':
    The destination frame of the transform or the frame of the pose.
'child' and 'name':
    The source frame of the transform or a name for the pose.

Mathematical operations on canonical objects use operator overloading.
Multiplying two transforms T1 and T2 is T1 * T2. To this end, CanonicalPoint,
CanonicalRotation, and CanonicalTransform are subclasses of numpy.matrix, which
has the same semantics. If a numpy array is needed, the 'array' property will
return a view of the data as an array.
Mathematical operations are auto-converting; for example, given a
CanonicalTransform, right-multiplying a (non-canonical) point will succeed if
the point is in any format that can be recognized by CanonicalPoint.
Left-multiplication of a non-canonical object will autoconvert if the object
does not support multiplication with canonical objects; because the left object
has priority in applying the multiplication, conversion cannot be guaranteed in
all cases.
The only exception is multiplication of a transform with a sequence of length 4
with elements 0,0,0,1: this could be an identity rotation as a quaternion, or
the point (0,0,0) as a 4-vector. In this case, an exception will be raised.

Operations on canonical objects are frame-aware. If the objects
have frames defined, and the frames conflict, an exception will be raised.
If a transform from frame A to frame B is applied by left-multiplication to a
point or rotation in frame B, the transform will automatically apply its
inverse and return the result in frame A.

The copy() methods on the objects allow keyword arguments for replacing fields
on the copy; for example, for a CanonicalPoint p, p.copy(x=0,stamp='now') will
return a copy with the same frame (if any), y and z values, but with x set to 0
and the timestamp set to the current time.

Conversion to ROS for CanonicalStamp and CanonicalDuration is provided by the
'ros' attribute, which returns a rospy.Time or rospy.Duration, respectively.
On CanonicalPoint, CanonicalRotation, and CanonicalTransform, conversion to ROS
messages is provided by a generator object available at the 'msg' attribute.
This generator object has methods named for each of the messages (despite being
methods, the initial letter is capitalized). The generators have two methods in
common, named 'Header' and 'stamp'. Each of these has two keyword arguments,
'default_stamp' and 'stamp_now'. Setting default_stamp to True indicates that
if no stamp is set, the current time should be used. Setting stamp_now to True 
overrides default_stamp if it is set, and always sets the stamp to be the 
current time (without changing the stamp on the object). The methods for 
creating stamped messages also take these keyword arguments. The 'tfMessage' 
method on the message generator for CanonicalTransform is slightly different; 
see its documentation for details.
"""

from __future__ import absolute_import

from . import canonical
from .canonical import stamp, duration, time, \
		transform, pose, \
		point, vector, \
		quaternion, rotation, rotation_tb, rotation_euler, \
		inverse_tf, \
		identity_tf, identity_rotation, \
		random_tf, random_pose, \
		random_point, random_vector, random_translation_tf, random_translation_pose, \
		random_rotation, random_quaternion, random_rotation_tf, random_rotation_pose

from .tb_angles import tb_angles, tb_str, tb_to_quat, tb_to_mat, tb_to_tf, tb_to_pose, get_tb_angles

try:
	import roslib as _roslib
	_roslib.load_manifest('tfx')
	import tf as rostf
	from tf import Transformer, TransformerROS, TransformBroadcaster
	def _instance():
		if rostf.TransformListener._INSTANCE is None:
			rostf.TransformListener._INSTANCE = rostf.TransformListener()
		return rostf.TransformListener._INSTANCE
		
	setattr(rostf.TransformListener,'_INSTANCE',None)
	setattr(rostf.TransformListener,'instance',staticmethod(_instance))
	
	def TransformListener():
		"""Drop-in replacement for tf.TransformListener creation that uses
		the singleton factory method that tfx installs on it. Equivalent to 
		calling tf.TransformListener.instance().
		"""
		return rostf.TransformListener.instance()
	
	def lookupTransform(to_frame, from_frame, wait=True, time=None):
		"""Shortcut for tf.TransformListener.instance().lookupTransform().
		
		This function will not work if any TransformListeners have been
		created directly; use tfx.TransformListener() to get the singleton.
		
		If the keyword argument 'wait' is True (the default), waitForTransform()
		will be called with a timeout of 5 seconds.
		If wait is a Duration or a number, waitForTransform() will be called with
		the value as the timeout.
		If wait is False or None, the lookup will not wait, and will throw an
		exception if the transform is not available.
		
		If the keyword argument 'time' is None (the default), the latest common
		time will be used. Otherwise, the time given will be used when looking
		up the transform.
		""" 
		import rospy
		listener = rostf.TransformListener.instance()
		if wait is not None:
			if wait is True:
				wait = rospy.Duration(5)
			else:
				wait = duration(wait).ros
			listener.waitForTransform(from_frame,to_frame,rospy.Time(),wait)
		if time is not None:
			t = stamp(time).ros
		else:
			t = listener.getLatestCommonTime(from_frame,to_frame)
		return transform(listener.lookupTransform(to_frame,from_frame,t),from_frame=from_frame,to_frame=to_frame,stamp=t)
	
	def convertToFrame(obj, to_frame, from_frame=None, ignore_stamp=False, wait=True):
		"""Convert the given object to the given frame.
		
		This function will not work if any TransformListeners have been
		created directly; use tfx.TransformListener() to get the singleton.
		If the object is already in the requested frame, no tf calls
		are made.
		
		If the keyword argument 'from_frame' is given, it will be used only if
		the given object does not have a frame already.
		
		If the keyword argument 'ignore_stamp' is False (the default), the 
		transform to the requested frame will use the timestamp of the given
		object, if it has one.
		If ignore_stamp is True, the most recent available transform will be used.
		
		If the keyword argument 'wait' is True (the default), waitForTransform()
		will be called with a timeout of 5 seconds.
		If wait is a Duration or a number, waitForTransform() will be called with
		the value as the timeout.
		If wait is False or None, the lookup will not wait, and will throw an
		exception if the transform is not available.
		"""
		obj_type = canonical._get_type(obj, default_4_to_quat=True).split('/')[0]
		if obj_type == 'p':
			obj = point(obj)
		elif obj_type in ['r','q']:
			obj = rotation(obj)
		else:
			obj = transform(obj)
		if not obj.frame and from_frame is None:
			raise ValueError("Converting a %s with no frame not allowed!" % obj._type_str)
		elif obj.frame:
			from_frame = obj.frame
		if obj.frame == to_frame:
			return obj
		
		if obj.stamp and not ignore_stamp:
			t = obj.stamp
		else:
			t = None
		return lookupTransform(to_frame,from_frame,wait=wait,time=t) * obj

except:
	pass
