
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

__version__ = '0.3.4'

from .tb_angles import tb_angles, get_tb_angles

import readline
import numpy
from math import *
import collections, numbers
import re
import time as pytime
import datetime as pydatetime
import sys
import copy
import itertools

_ROS = False
try:
	roslib = __import__('roslib')
	try:
		roslib.load_manifest('tfx')
	except:
		roslib.load_manifest('tf')
		roslib.load_manifest('geometry_msgs')
		roslib.load_manifest('std_msgs')
	rospy = __import__('rospy')
	try:
		genpy = __import__('genpy')
	except:
		genpy = rospy.rostime
	geometry_msgs = __import__('geometry_msgs.msg')
	gm = geometry_msgs.msg
	std_msgs = __import__('std_msgs.msg')
	sm = std_msgs.msg
	
	tf = __import__('tf')
	tft = tf.transformations
	#import_module('tf.msg')
	_ROS = True
except:
	from . import transformations as tft 

_PANDAS = False
try:
	pd = __import__('pandas')
	_PANDAS = True
except:
	pass

_FRAME_KEYS = ['frame','frame_id']
_PARENT_KEYS = ['parent','to_frame','frame_to'] + _FRAME_KEYS
_CHILD_KEYS = ['child','from_frame','frame_from','name','child_frame_id']

_POS_KEYS = ['position','pos','translation','trans']
_EXTENDED_POS_KEYS = _POS_KEYS + ['p','t']
_ROT_KEYS = ['orientation','ori','rotation','rot']
_EXTENDED_ROT_KEYS = _ROT_KEYS + ['o','r']
_TB_KEYS = ['tb_angles','tb']
_ROT_AND_TB_KEYS = _ROT_KEYS + _TB_KEYS

_NANO_KEYS = ['nanoseconds','nanosecond','nanos','nano','nsecs','nsec','ns']
_MICRO_KEYS = ['microseconds','microsecond','micros','micro','usecs','usec','us']
_MILLI_KEYS = ['milliseconds','millisecond','millis','milli','msecs','msec','ms']
_SEC_KEYS = ['seconds','second','secs','sec','s']

def _get_kwarg(kwargs,keys,pop=True):
	val = None
	for key in keys:
		if key in kwargs:
			if pop:
				val = kwargs.pop(key)
			else:
				val = kwargs.get(key)
	return val

_prep_str = lambda val: re.sub(r' +',' ',re.sub(r'[\[\]\(\),]',' ',str(val))).strip()
_parse_str = lambda val: _prep_str(val).split(' ')

_convert_to_list = lambda data: [float(num) for num in data.flat]
_convert_to_tuple = lambda data: tuple(_convert_to_list(data))

if _ROS:
	def _ismsginstance(obj,typename):
		if not isinstance(typename,tuple):
			typename = (typename,)
		for t in typename:
			if isinstance(obj,t) or (hasattr(obj,'_type') and obj._type == t._type):
				return True
		return False

def _extract_stamp(data,data2=None):
	if data2 is not None:
		stamp1 = _extract_stamp(data)
		stamp2 = _extract_stamp(data2)
		
		if stamp1 is None and stamp2 is None:
			if isinstance(data,collections.Sequence) and len(data) == 2:
				stamp1 = _extract_stamp(data[0])
				stamp2 = _extract_stamp(data[1])
		
		if stamp1 is not None and stamp2 is not None:
			if stamp1 > stamp2:
				return stamp1
			else:
				return stamp2
		elif stamp1 is not None:
			return stamp1
		elif stamp2 is not None:
			return stamp2
		else:
			return None
	
	if data is None or (isinstance(data,collections.Sequence) and len(data) == 0):
		return None
	
	stamp1 = None
	
	data_type = _get_type(data)
	sub_types = data_type.split('/')[1:]
	if data_type == 'u':
		pass #TODO: complain
	if sub_types[0] == 'c' and data.stamp is not None:
		stamp1 = data.stamp.copy()
	elif data_type.find('msg') != -1 and sub_types[-1] == 'h':
		stamp1 = stamp(data.header.stamp)
	elif isinstance(data,dict):
		tried_t = None
		for key, value in data.iteritems():
			if key == 'stamp':
				stamp1 = stamp(value,copy=True)
				break
			if key == 't':
				tried_t = value
				try:
					stamp1 = stamp(value,copy=True)
					break
				except:
					pass
			if isinstance(value,dict):
				stamp1 = _extract_stamp(value)
				if stamp1 is not None:
					break
		if tried_t and stamp1 is None:
			raise UnknownInputError(data1=tried_t,input_type='stamp')
	
	return stamp1

def _extract_frame(data,data2=None):
	if data2 is not None:
		frame1 = _extract_frame(data)
		frame2 = _extract_frame(data2)
		
		if frame1 is None and frame2 is None:
			if isinstance(data,collections.Sequence) and len(data) == 2:
				frame1 = _extract_frame(data[0])
				frame2 = _extract_frame(data[1])
		
		if frame1 is not None and frame2 is not None:
			if frame1 != frame2:
				#TODO: flip out
				pass
		elif frame1 is not None:
			return frame1
		elif frame2 is not None:
			return frame2
		else:
			return None
	
	if data is None or (isinstance(data,collections.Sequence) and len(data) == 0):
		return None
	
	frame1 = None
	
	data_type = _get_type(data)
	sub_types = data_type.split('/')[1:]
	if data_type == 'u':
		pass #TODO: complain
	if sub_types[0] == 'c':
		frame1 = data._frame
	elif data_type.find('msg') != -1 and sub_types[-1] == 'h':
		frame1 = data.header.frame_id
	elif isinstance(data,dict):
		for key, value in data.iteritems():
			if key in _PARENT_KEYS and isinstance(value,basestring):
				frame1 = value
				break
			if isinstance(value,dict):
				frame1 = _extract_frame(value)
				if frame1 is not None:
					break
	
	return frame1

def _extract_name(data):
	if data is None or (isinstance(data,collections.Sequence) and len(data) == 0):
		return None
	
	name = None
	
	data_type = _get_type(data)
	base_type = data_type.split('/')[0]
	sub_types = data_type.split('/')[1:]
	if data_type == 'u':
		pass #TODO: complain
	if base_type in ['tf','ps'] and sub_types[0] == 'c':
		name = data._name
	elif hasattr(data,'child_frame_id'):
		name = data.child_frame_id
	elif isinstance(data,dict):
		for key, value in data.iteritems():
			if key in _CHILD_KEYS and isinstance(value,basestring):
				name = value
				break
			if isinstance(value,dict):
				name = _extract_name(value)
				if name is not None:
					break
	
	return name

def _extract_header_frame_and_stamp(kwargs):
	has_header = False
	header_frame = None
	header_stamp = None
	if kwargs.has_key('header'):
		has_header = True
		header = kwargs['header']
		if isinstance(header,dict):
			for key in _FRAME_KEYS:
				if header.has_key(key):
					header_frame = header[key]
			header_stamp = header.pop('stamp',None)
		else:
			if hasattr(header,'frame_id'):
				header_frame = header.frame_id
			if hasattr(header,'stamp'):
				header_stamp = header.stamp
	return has_header, header_frame, header_stamp

def _get_axis_angle(val1,val2):
	if isinstance(val1,numbers.Number) and isinstance(val2,collections.Sequence) and len(val2) == 3:
		return (numpy.array(val2),val1)
	elif isinstance(val2,numbers.Number) and isinstance(val1,collections.Sequence) and len(val1) == 3:
		return (numpy.array(val1),val2)
	elif isinstance(val1,numbers.Number) and isinstance(val2,numpy.ndarray) and val2.size == 3:
		return (val2.reshape((3,)),val1)
	elif isinstance(val2,numbers.Number) and isinstance(val1,numpy.ndarray) and val1.size == 3:
		return (val1.reshape((3,)),val2)
	else:
		return None

def _get_pt_string_fmt(kwargs):
	for fmt_key in ['fmt','format']:
		if not kwargs.has_key(fmt_key): continue
		fmt = kwargs[fmt_key]
		for key in ['a','array']:
			if fmt == key:
				return 'a'
		for key in ['m','mat','matrix']:
			if fmt == key:
				return 'm'
	return 'a'

def _get_rot_string_fmt(kwargs):
	for fmt_key in ['fmt','format']:
		if not kwargs.has_key(fmt_key): continue
		fmt = kwargs[fmt_key]
		for key in ['q','quat','quaternion']:
			if fmt == key:
				return 'q'
		for key in ['m','mat','matrix']:
			if fmt == key:
				return 'm'
	return 'tb'

def _get_print_key(kwargs,keys):
	print_key = {}
	all = None
	if kwargs.has_key('all'):
		all = kwargs['all']
	elif kwargs.has_key('simple'):
		all = not kwargs['simple']
	else:
		all = not any([kwargs.get(key,False) for key in keys])
	print_key = collections.defaultdict(lambda: all,**kwargs)
	return print_key

def _prep_subkwargs(_kwargs,_prefixes,_remove,**set_kwargs):
	subkwargs = _kwargs.copy()
	subkwargs.update(set_kwargs)
	if _remove:
		for key in _remove:
			subkwargs.pop(key,None)
	add = {}
	for key,value in subkwargs.iteritems():
		for prefix in _prefixes:
			if key.startswith(prefix + '_'):
				add[key[len(prefix)+1:]] = value
	subkwargs.update(add)
	return subkwargs

def _prep_pt_subkwargs(_kwargs,_remove,**set_kwargs):
	return _prep_subkwargs(_kwargs, _EXTENDED_POS_KEYS,_remove,**set_kwargs)

def _prep_rot_subkwargs(_kwargs,_remove,**set_kwargs):
	return _prep_subkwargs(_kwargs, _EXTENDED_ROT_KEYS,_remove,**set_kwargs)

def _prep_stamp_subkwargs(_kwargs,_remove = [],**set_kwargs):
	prefix = 'stamp'
	subkwargs = {}
	subkwargs.update(set_kwargs)
	for key,value in _kwargs.iteritems():
		if key.startswith(prefix + '_'):
			new_key = key[len(prefix)+1:]
			if new_key in _remove:
				continue
			subkwargs[new_key] = value
	if _kwargs.has_key('stamp') and isinstance(_kwargs['stamp'],basestring):
		subkwargs[_kwargs['stamp']] = True
	return subkwargs

class UnknownInputError(TypeError):
	def __init__(self,type1=None,data1=None,type2=None,data2=None,input_type = None,msg=None):
		if msg is None:
			msg = 'Unknown '
			if input_type is not None:
				msg += input_type + ' '
			msg += 'input'
			if type2 is not None:
				msg += 's'
			
			if type1 is not None:
				msg += ' '
				msg += type1
			if type2 is not None:
				msg += ', ' + type2
			
			ds1 = ''
			if data1 is not None:
				msg += ': '
				ds1 = repr(data1)
				if ds1.find('\n') != -1:
					msg += '\n'
					
				msg += ds1
			
			ds2 = ''
			if data2 is not None:
				msg += ', '
				
				ds2 = repr(data2)
				if ds1.find('\n') != -1 or ds2.find('\n') != -1:
					msg += '\n'
					
				msg += ds2
		self.msg = msg
	
	def __str__(self):
		return self.msg

class FrameError(ValueError):
	def __init__(self,*args):
		if len(args) == 1:
			self.msg = args[0]
		else:
			obj1 = args[0]
			frame1 = args[1]
			obj2 = args[2]
			frame2 = args[3]
			self.msg = "Can't apply {obj1} with frame {frame1} to {obj2} with frame {frame2}".format(obj1=obj1,frame1=frame1,obj2=obj2,frame2=frame2)
	
	def __str__(self):
		return self.msg

class DerivedObjectError(RuntimeError):
	def __init__(self,msg):
		self.msg = msg
	
	def __str__(self):
		return self.msg
_CanonicalStampData = collections.namedtuple('_CanonicalStampData', ['nanoseconds','datetime'])

class CanonicalStamp(object):
	"""A class representing a timestamp."""
	@staticmethod
	def now():
		"""Return a stamp of the current time"""
		return CanonicalStamp('now')
	
	@staticmethod
	def zero():
		"""Return a stamp of time 0 (usually represents null, unknown or invalid)"""
		return CanonicalStamp(0)

	def __init__(self,*args,**kwargs):
		"""Constructor. This simply calls set() with the arguments given. With no arguments,
		sets to now."""
		self._data = None
		self._master = kwargs.get('_master',None)
		if self._master is not None:
			return
		if not args and not kwargs:
			args = ('now',)
		self.set(*args,**kwargs)
	
	def set(self,*args,**kwargs):
		"""Set the stamp to the input value.
		
		Valid input types are float, time, datetime, various string formats,
		ros.Time, pandas.Timestamp (both only if available).
		"""
		if self._master is not None:
			raise DerivedObjectError("This stamp is derived from another object and cannot be changed")
		
		if len(args) == 1:
			stamp = args[0]
		else:
			stamp = args
		
		nano_kwarg = None
		for key in _NANO_KEYS:
			if kwargs.has_key(key):
				nano_kwarg = kwargs[key]
				break
		micro_kwarg = None
		for key in _MICRO_KEYS:
			if kwargs.has_key(key):
				micro_kwarg = kwargs[key]
				break
		milli_kwarg = None
		for key in _MILLI_KEYS:
			if kwargs.has_key(key):
				milli_kwarg = kwargs[key]
				break
		s_kwarg = None
		for key in _SEC_KEYS:
			if kwargs.has_key(key):
				s_kwarg = kwargs[key]
				break
		
		default_nanos = nano_kwarg is True
		default_micros = micro_kwarg is True
		default_millis = milli_kwarg is True
		
		seconds = None
		nanoseconds = None
		dt = None
		if any(not isinstance(v,bool) and isinstance(v,numbers.Number) for v in [nano_kwarg,micro_kwarg,milli_kwarg,s_kwarg]):
			nanoseconds = 0
			if nano_kwarg is not None:
				nanoseconds += long(nano_kwarg)
			if micro_kwarg is not None:
				nanoseconds += long(micro_kwarg) * 1e3
			if milli_kwarg is not None:
				nanoseconds += long(milli_kwarg) * 1e6
			if s_kwarg is not None:
				nanoseconds += float(s_kwarg) * 1e9
		elif isinstance(stamp,basestring):
			if stamp == 'now':
				if _ROS and rospy.rostime._rostime_initialized:
					nanoseconds = rospy.Time.now().to_nsec()
				else:
					dt = pydatetime.datetime.now()
			else:
				try:
					if default_nanos:
						nanoseconds = long(stamp)
					elif default_micros:
						nanoseconds = long(stamp) * 1e3
					elif default_millis:
						nanoseconds = long(stamp) * 1e6
					else:
						seconds = float(stamp)
				except:
					raise UnknownInputError(data1=stamp, input_type='stamp')
		elif isinstance(stamp,CanonicalStamp):
			dt = stamp.datetime
			nanoseconds = stamp.nanoseconds
		elif isinstance(stamp,pydatetime.datetime):
			dt = stamp
		elif _PANDAS and hasattr(pd,'Timestamp') and isinstance(stamp,pd.Timestamp):
			dt = stamp.to_datetime()
		elif isinstance(stamp,dict):
			for nskey in _NANO_KEYS:
				if stamp.has_key(nskey):
					nanoseconds = long(stamp[nskey])
					for skey in _SEC_KEYS:
						if stamp.has_key(skey):
							nanoseconds += long(stamp[skey]) * long(1e9)
							break
				break
			else:
				for skey in _SEC_KEYS:
					if stamp.has_key(skey):
						seconds = float(stamp[skey])
						break
				else:
					raise UnknownInputError(data1=stamp,input_type='stamp')
		else:
			if isinstance(stamp,tuple) and len(stamp) == 2:
				nanoseconds = long(stamp[0]) * long(1e9) + long(stamp[1])
			elif isinstance(stamp,pytime.struct_time) or \
				(isinstance(stamp,tuple) and len(stamp) == 9):
				seconds = pytime.mktime(stamp)
			elif _ROS and isinstance(stamp,genpy.Time):
				nanoseconds = stamp.to_nsec()
			elif _ROS and _ismsginstance(stamp,std_msgs.msg.Time):
				nanoseconds = stamp.data.to_nsec()
			elif _ROS and _ismsginstance(stamp,std_msgs.msg.Header):
				nanoseconds = stamp.stamp.to_nsec()
			elif hasattr(stamp,'header') and hasattr(stamp.header,'stamp'):
				nanoseconds = stamp.header.stamp.to_nsec()
			else:
				try:
					if default_nanos:
						nanoseconds = long(stamp)
					elif default_micros:
						nanoseconds = long(stamp) * 1e3
					elif default_millis:
						nanoseconds = long(stamp) * 1e6
					else:
						seconds = float(stamp)
				except:
					raise UnknownInputError(data1=stamp,input_type='stamp')
		
		if nanoseconds is not None:
			seconds = nanoseconds / 1e9
		
		if seconds is None:
			#must have dt
			seconds = pytime.mktime(dt.timetuple())+1e-6*dt.microsecond
		if dt is None:
			#must have seconds
			dt = pydatetime.datetime.fromtimestamp(seconds)
		if nanoseconds is None:
			#must have seconds
			int_secs = int(seconds)
			int_nsecs = int((seconds - int_secs) * 1000000000)
			nanoseconds = int_secs * long(1e9) + int_nsecs
		
		#I thought Java's datetime API was bad, but Python is miserable
		#TODO: Timezone handling (not in the Python std library)
		self._data = _CanonicalStampData(nanoseconds=nanoseconds,datetime=dt)
	
	@property
	def datetime(self):
		"""The timestamp as a datetime object."""
		if self._master is not None:
			if self._master.stamp is None:
				raise DerivedObjectError("This stamp is derived from another object and is no longer valid")
			return self._master.stamp.datetime
		return self._data.datetime
	@datetime.setter
	def datetime(self,value):
		self.set(value)
	
	@property
	def time(self):
		"""The timestamp as a time tuple."""
		if self._master is not None:
			if self._master.stamp is None:
				raise DerivedObjectError("This stamp is derived from another object and is no longer valid")
			return self._master.stamp.time
		return self.datetime.timetuple()
	@time.setter
	def time(self,value):
		self.set(value)
	
	@property
	def seconds(self):
		"""The timestamp as floating-point seconds since the epoch."""
		if self._master is not None:
			if self._master.stamp is None:
				raise DerivedObjectError("This stamp is derived from another object and is no longer valid")
			return self._master.stamp.seconds
		return self._data.nanoseconds / 1e9
	@seconds.setter
	def seconds(self,value):
		self.set(value)
	
	@property
	def nanoseconds(self):
		"""The timestamp as floating-point nanoseconds since the epoch."""
		if self._master is not None:
			if self._master.stamp is None:
				raise DerivedObjectError("This stamp is derived from another object and is no longer valid")
			return self._master.stamp.nanoseconds
		return self._data.nanoseconds
	@nanoseconds.setter
	def nanoseconds(self,value):
		self.set(value,nanos=True)
	
	if _ROS:
		@property
		def ros(self):
			"""The timestamp as ROS Time object."""
			if self._master is not None:
				if self._master.stamp is None:
					raise DerivedObjectError("This stamp is derived from another object and is no longer valid")
				return self._master.stamp.ros
			int_secs = int(self.nanoseconds / 1e9)
			int_nsecs = int(self.nanoseconds % long(1e9))
			return rospy.Time(int_secs,int_nsecs)
		@ros.setter
		def ros(self,value):
			self.set(value)
	
	def is_zero(self):
		"""Test if this timestamp is set to zero."""
		return self.seconds == 0
	
	def tostring(self,**kwargs):
		"""Return a string representation of the timestamp.
		
		With no arguments, returns the ISO 8601 representation, unless
		the time is 0, in which case it returns the string '0'.
		
		A format may be specified with the keyword argument 'fmt' or 'format'.
		
		If the keyword argument 'seconds' is given and is True, the format will be
		interpreted as a format string for the value of the timestamp in seconds.
		If no format is given, the default is '%f'.
		As a shortcut, the value of the 'seconds' keyword argument can be the format.
		
		If the format is 'iso', the string will be in ISO 8601 format even if the
		timestamp is zero (i.e., it will return 1970-01-01T00:00:00.000Z).
		
		Otherwise, the format will be used as input to datetime.datetime.strftime()."""
		fmt = None
		for key in ['fmt','format']:
			if kwargs.has_key(key):
				fmt = kwargs[key]
		if fmt:
			if fmt == 'iso':
				return self.datetime.isoformat()
			else:
				return self.datetime.strftime(fmt)
		elif kwargs.has_key('seconds'):
			sec_fmt = kwargs['seconds']
			if sec_fmt is True:
				if fmt:
					sec_fmt = fmt
				elif self.is_zero():
					return '0'
				else:
					sec_fmt = 'f'
			if not sec_fmt.startswith('%'):
				sec_fmt = '%' + sec_fmt
			return sec_fmt % self.seconds
		else:
			if self.is_zero():
				return '0'
			else:
				return self.datetime.isoformat()
	
	@property
	def _type_str(self):
		return 'stamp'
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		if self.seconds == 0:
			return 'stamp(0)'
		else:
			return 'stamp(%f)'% self.seconds

	def copy(self):
		"""Return a copy of this timestamp."""
		return stamp(self,copy=True)
	
	def __copy__(self):
		return self.copy()
	
	def __cmp__(self,other):
		if isinstance(other,(pytime.struct_time,tuple)):
			s1 = floor(self.seconds)
		else:
			s1 = self.seconds
		s2 = time(other).seconds
		if s1 < s2:
			return -1
		elif s1 == s2:
			return 0
		else:
			return 1
	
	def __lt__(self,other):
		return self.__cmp__(other) < 0
	
	def __le__(self,other):
		return self.__cmp__(other) <= 0
	
	def __eq__(self,other):
		return self.__cmp__(other) == 0
	
	def __ne__(self,other):
		return self.__cmp__(other) != 0
	
	def __gt__(self,other):
		return self.__cmp__(other) > 0
	
	def __ge__(self,other):
		return self.__cmp__(other) >= 0
	
	def __nonzero__(self):
		return self.seconds != 0
	
	def __add__(self,other):
		other = duration(other)
		s = self.nanoseconds + other.nanoseconds
		return stamp(s,nanos=True)
	
	def __radd__(self,other):
		other = duration(other)
		s = other.nanoseconds + self.nanoseconds
		return stamp(s,nanos=True)
	
	def __iadd__(self,other):
		other = duration(other)
		s = self.nanoseconds + other.nanoseconds
		self.nanoseconds = s
		return self
	
	def __sub__(self,other):
		other = time(other,default_duration=True)
		s = self.nanoseconds - other.nanoseconds
		if isinstance(other,CanonicalDuration):
			return stamp(s,nanos=True)
		else:
			return duration(s,nanos=True)
	
	def __rsub__(self,other):
		other = stamp(other)
		s = other.nanoseconds - self.nanoseconds
		return duration(s,nanos=True)
	
	def __isub__(self,other):
		other = duration(other)
		s = self.nanoseconds - other.nanoseconds
		self.nanoseconds = s
		return self

class CanonicalDuration(object):
	"""A class representing a duration."""
	def __init__(self,*args,**kwargs):
		self._nanoseconds = None
		self.set(*args,**kwargs)
	
	def set(self,*args,**kwargs):
		if not args and not kwargs:
			self._nanoseconds = 0
			return
		
		if not args:
			dur = pydatetime.timedelta(**kwargs)
		elif len(args) == 1:
			dur = args[0]
		else:
			dur = args
		
		nano_kwarg = None
		for key in _NANO_KEYS:
			if kwargs.has_key(key):
				nano_kwarg = kwargs[key]
				break
		micro_kwarg = None
		for key in _MICRO_KEYS:
			if kwargs.has_key(key):
				micro_kwarg = kwargs[key]
				break
		milli_kwarg = None
		for key in _MILLI_KEYS:
			if kwargs.has_key(key):
				milli_kwarg = kwargs[key]
				break
		s_kwarg = None
		for key in _SEC_KEYS:
			if kwargs.has_key(key):
				s_kwarg = kwargs[key]
				break
		
		default_nanos = nano_kwarg is True
		default_micros = micro_kwarg is True
		default_millis = milli_kwarg is True
		
		nanoseconds = None
		seconds = None
		if any(not isinstance(v,bool) and isinstance(v,numbers.Number) for v in [nano_kwarg,micro_kwarg,milli_kwarg,s_kwarg]):
			nanoseconds = 0
			if nano_kwarg is not None:
				nanoseconds += long(nano_kwarg)
			if micro_kwarg is not None:
				nanoseconds += long(micro_kwarg) * 1e3
			if milli_kwarg is not None:
				nanoseconds += long(milli_kwarg) * 1e6
			if s_kwarg is not None:
				nanoseconds += long(s_kwarg) * 1e9
		elif isinstance(dur,basestring):
			try:
				if default_nanos:
					nanoseconds = long(dur)
				elif default_micros:
					nanoseconds = long(dur) * 1e3
				elif default_millis:
					nanoseconds = long(dur) * 1e6
				else:
					seconds = float(dur)
			except:
				raise UnknownInputError(data1=dur, input_type='duration')
		elif isinstance(dur,tuple) and len(dur) == 2:
			nanoseconds = long(dur[0]) * long(1e9) + long(dur[1])
		elif isinstance(dur,CanonicalDuration):
			nanoseconds = dur.nanoseconds
		elif isinstance(dur,pydatetime.timedelta):
			seconds = dur.total_seconds()
		elif _ROS and isinstance(dur, genpy.Duration):
			nanoseconds = dur.to_nsec()
		elif _ROS and _ismsginstance(dur,std_msgs.msg.Duration):
			nanoseconds = dur.data.to_nsec()
		else:
			try:
				if default_nanos:
					nanoseconds = long(dur)
				elif default_micros:
					nanoseconds = long(dur) * 1e3
				elif default_millis:
					nanoseconds = long(dur) * 1e6
				else:
					seconds = float(dur)
			except:
				raise UnknownInputError(data1=dur, input_type='duration')
		
		if nanoseconds is None:
			int_secs = int(seconds)
			int_nsecs = int((seconds - int_secs) * 1000000000)
			nanoseconds = int_secs * long(1e9) + int_nsecs
		
		self._nanoseconds = nanoseconds
	
	@property
	def seconds(self):
		"""The duration in seconds"""
		return self._nanoseconds / 1e9
	@seconds.setter
	def seconds(self,value):
		self.set(value)
	
	@property
	def nanoseconds(self):
		"""The duration in nanoseconds"""
		return self._nanoseconds
	@nanoseconds.setter
	def nanoseconds(self,value):
		self.set(value,nanos=True)
	
	@property
	def timedelta(self):
		"""The duration as a datetime timedelta"""
		return pydatetime.timedelta(seconds=self.seconds)
	@timedelta.setter
	def timedelta(self,value):
		self.set(value)
	
	if _ROS:
		@property
		def ros(self):
			"""The duration as a ROS Duration"""
			int_secs = int(self.nanoseconds / 1e9)
			int_nsecs = int(self.nanoseconds % long(1e9))
			return rospy.Duration(int_secs,int_nsecs)
		@ros.setter
		def ros(self,value):
			self.set(value)
	
	def tostring(self,**kwargs):
		"""Convert the duration to a string, by default as the number of seconds.
		
		A format may be specified using the keyword argument 'fmt' or 'format'.
		This must be a format string. By default, the format string is '%f'."""
		fmt = None
		for key in ['fmt','format']:
			if kwargs.has_key(key):
				fmt = kwargs[key]
		if fmt:
			if not fmt.startswith('%'):
				fmt = '%' + fmt
			return fmt % self.seconds
		else:
			if self.seconds == 0:
				return '0'
			else:
				return '%f' % self.seconds
	
	@property
	def _type_str(self):
		return 'duration'
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		if self.seconds == 0:
			return 'duration(0)'
		else:
			return 'duration(%f)'% self.seconds

	def copy(self):
		return duration(self,copy=True)
	
	def __copy__(self):
		return self.copy()
	
	def __lt__(self,other):
		other = time(other,default_duration=True)
		return self.seconds < other.seconds
	
	def __le__(self,other):
		other = time(other,default_duration=True)
		return self.seconds <= other.seconds
	
	def __eq__(self,other):
		other = time(other,default_duration=True)
		return self.seconds == other.seconds
	
	def __ne__(self,other):
		other = time(other,default_duration=True)
		return self.seconds != other.seconds
	
	def __gt__(self,other):
		other = time(other,default_duration=True)
		return self.seconds > other.seconds
	
	def __ge__(self,other):
		other = time(other,default_duration=True)
		return self.seconds >= other.seconds
	
	def __nonzero__(self):
		return self.seconds != 0
	
	def __add__(self,other):
		other = time(other,default_duration=True)
		s = self.nanoseconds + other.nanoseconds
		if isinstance(other,CanonicalDuration):
			return duration(s,nanos=True)
		else:
			return stamp(s,nanos=True)
	
	def __radd__(self,other):
		other = time(other,default_duration=True)
		s = other.nanoseconds + self.nanoseconds
		if isinstance(other,CanonicalDuration):
			return duration(s,nanos=True)
		else:
			return stamp(s,nanos=True)
	
	def __iadd__(self,other):
		other = duration(other)
		s = self.nanoseconds + other.nanoseconds
		self.nanoseconds = s
		return self
	
	def __sub__(self,other):
		other = duration(other)
		s = self.nanoseconds - other.nanoseconds
		return duration(s,nanos=True)
	
	def __rsub__(self,other):
		other = time(other,default_duration=True)
		s = other.nanoseconds - self.nanoseconds
		if isinstance(other,CanonicalDuration):
			return duration(s,nanos=True)
		else:
			return stamp(s,nanos=True)
	
	def __isub__(self,other):
		other = duration(other)
		s = self.nanoseconds - other.nanoseconds
		self.nanoseconds = s
		return self
	
	def __mul__(self,other):
		s = self.seconds * other
		return duration(s)
	
	def __rmul__(self,other):
		s = other * self.seconds
		return duration(s)
	
	def __imul__(self,other):
		s = self.seconds * other
		self.seconds = s
		return self
	
	def __div__(self,other):
		s = self.seconds / other
		return duration(s)
	
	def __idiv__(self,other):
		s = self.seconds / other
		self.seconds = s
		return self

PointTuple = collections.namedtuple('PointTuple', ['x','y','z'])
QuaternionTuple = collections.namedtuple('QuaternionTuple', ['x','y','z','w'])

if _ROS:
	class _CanonicalPointMsgs(object):
		def __init__(self,point):
			self._obj = point
		
		def Point(self):
			"""Get a geometry_msgs/Point message."""
			return gm.Point(*self._obj.tuple)
		
		def PointStamped(self,**kwargs):
			"""Get a geometry_msgs/PointStamped message.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			msg = gm.PointStamped()
			msg.header = self.Header(**kwargs)
			msg.point = self.Point()
			return msg
		
		def Vector3(self):
			"""Get a geometry_msgs/Vector3 message."""
			return gm.Vector3(*self._obj.tuple)
		
		def Vector3Stamped(self,**kwargs):
			"""Get a geometry_msgs/Vector3Stamped message.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			msg = gm.Vector3Stamped()
			msg.header = self.Header(**kwargs)
			msg.vector = self.Vector3()
			return msg
		
		def Header(self,**kwargs):
			"""Get a std_msgs/Header for the frame and stamp.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			kwargs['default_stamp'] = kwargs.get('default_stamp',True)
			header = std_msgs.msg.Header()
			if self._obj.frame:
				header.frame_id = self._obj.frame
			header.stamp = self.stamp(**kwargs)
			return header
		
		def stamp(self,**kwargs):
			"""Get the stamp as a rospy.Time object.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to True to use the current time if the stamp is None
			"""
			default_stamp = kwargs.get('default_stamp')
			stamp_now = kwargs.get('stamp_now')
			the_stamp = kwargs.get('stamp')
			if stamp_now or the_stamp is True:
				return stamp('now').ros
			elif the_stamp is not None:
				return stamp(the_stamp).ros
			elif self._obj.stamp is not None:
				return self._obj.stamp.ros
			elif default_stamp:
				return stamp('now').ros
			else:
				return rospy.Time(0)
		

class CanonicalPoint(numpy.matrix):
	@staticmethod
	def random(scale=1,mean=0,dist='uniform',**kwargs):
		"""Return a random point.
		
		Options for distribution:
		'uniform', 'normal' or 'gaussian'
		
		Keyword args are passed into the constructor."""
		if isinstance(scale,numbers.Number):
			scale = float(scale) * numpy.ones(3, dtype=float)
		else:
			scale = numpy.array(scale)
		if isinstance(mean,numbers.Number):
			mean = float(mean) * numpy.ones(3, dtype=float)
		else:
			mean = numpy.array(mean)
		dist = kwargs.pop('distribution',dist)
		if dist == 'uniform':
			data = scale * numpy.random.rand(3) + mean
		elif dist in ['normal','gaussian']:
			data = scale * numpy.random.randn(3) + mean
		return CanonicalPoint(data,**kwargs)
	
	@property
	def frame(self):
		"""The frame of this point."""
		if self._master is not None:
			return self._master._frame
		else:
			return self._frame
	@frame.setter
	def frame(self,value):
		if self._master is not None:
			raise DerivedObjectError('This point is derived from another object, the frame cannot be changed')
		self._frame = value
	parent = frame
	
	@property
	def stamp(self):
		"""The timestamp as a CanonicalStamp."""
		if self._master is not None:
			return self._master.stamp
		else:
			return self._stamp
	@stamp.setter
	def stamp(self,value):
		if self._master is not None:
			raise DerivedObjectError('This point is derived from another object, the stamp cannot be changed')
		elif value is None:
			self._stamp = None
		elif not self._stamp:
			self._stamp = stamp(value,copy=True)
		else:
			self._stamp.set(value)
	def set_stamp_now(self):
		self.stamp = 'now'
	
	@property
	def x(self):
		return self[0,0]
	@x.setter
	def x(self,value):
		self[0,0] = value
		
	@property
	def y(self):
		return self[1,0]
	@y.setter
	def y(self,value):
		self[1,0] = value
	
	@property
	def z(self):
		return self[2,0]
	@z.setter
	def z(self,value):
		self[2,0] = value
	
	@property
	def vector3(self):
		"""A copy of this point as a 3x1 numpy matrix"""
		v = numpy.matrix(numpy.ones((3,1)))
		v[0:3,0] = self
		return v
	
	@property
	def vector4(self):
		"""A copy of this point as a 4x1 numpy matrix (with 1 as the fourth element)"""
		v = numpy.matrix(numpy.ones((4,1)))
		v[0:3,0] = self
		return v
	
	@property
	def matrix(self):
		"""A view of this point as a 3x1 numpy matrix (not array)."""
		return self.view(numpy.matrix)
	
	@property
	def array(self):
		"""A view of this point as a 3-element 1D numpy array."""
		return self.view(numpy.ndarray).reshape(3)
	
	@property
	def array2D(self):
		"""A view of this point as a 3x1 numpy array."""
		return self.view(numpy.ndarray).reshape((3,1))
	
	@property
	def list(self):
		"""The point as a 3-element list"""
		return _convert_to_list(self)
	
	@property
	def tuple(self):
		"""The point as a 3-element namedtuple with fields x, y, and z."""
		return PointTuple._make(_convert_to_tuple(self))
	
	@property
	def dict(self):
		return self.todict()
	
	def as_transform(self,**kwargs):
		"""Return a transform with this translation.
		Keyword args are passed into the constructor."""
		kwargs['frame'] = kwargs.get('frame',self.frame)
		kwargs['stamp'] = kwargs.get('stamp',self.stamp)
		return transform(self.view(numpy.ndarray),**kwargs)
	as_tf = as_transform
	def as_pose(self,**kwargs):
		"""Return a pose with this position.
		Keyword args are passed into the constructor."""
		kwargs['frame'] = kwargs.get('frame',self.frame)
		kwargs['stamp'] = kwargs.get('stamp',self.stamp)
		return pose(self.view(numpy.ndarray),**kwargs)
	
	if _ROS:
		@property
		def msg(self):
			"""A ROS message creator object.
			
			The message creator has methods for the various ROS message types:
			Point
			PointStamped
			Vector3
			Vector3Stamped
			Header
			
			and an additional method, stamp, which returns a rospy.Time object for 
			the stamp.
			
			The PointStamped, VectorStamped, Header, and stamp methods all take
			two keyword arguments:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			"""
			if self._msg is None:
				self._msg = _CanonicalPointMsgs(self)
			return self._msg
	
	@property
	def norm(self):
		"""The 2-norm."""
		return numpy.linalg.norm(self)
	@property
	def norm_1(self):
		return numpy.linalg.norm(self,1)
	@property
	def norm_2(self):
		return numpy.linalg.norm(self,2)
	@property
	def norm_inf(self):
		return numpy.linalg.norm(self,numpy.inf)
	
	@property
	def unit_vector(self):
		"""A copy of this point as a unit vector."""
		return self / self.norm
	
	def interpolate(self,other,fraction):
		"""Interpolate between this and another point."""
		other = point(other)
		return point(self + (other-self) * float(fraction))
	
	def capped(self,distance):
		"""Return a new point whose norm is at most the input value."""
		l = self.norm_2
		if l < distance:
			return point(self,copy=True)
		return self * distance / l

	def distance(self,other):
		"""The distance to another point."""
		other = point(other)
		diff = other - self
		return diff.norm

	def rotation_to(self,other):
		other = point(other)
		if self.frame and other.frame and self.frame != other.frame:
			raise FrameError("Can't get rotation between vectors in frames %s and %s" % (self.frame,other.frame))
		return rotation(self,other)
	def rotation_from(self,other):
		other = point(other)
		if self.frame and other.frame and self.frame != other.frame:
			raise FrameError("Can't get rotation between vectors in frames %s and %s" % (self.frame,other.frame))
		return rotation(other,self)
	
	def tostring(self,**kwargs):
		"""Return a string representation of this point.
		
		This is different from the tostring method of numpy arrays. To get that
		behavior, set the keyword argument 'numpy' to True.
		"""
		if kwargs.get('numpy',kwargs.get('np',False)):
			return super(CanonicalPoint,self).tostring()
		
		output_type = _get_pt_string_fmt(kwargs)
		
		brackets = kwargs.get('brackets')
		if not kwargs.has_key('brackets') or brackets is True:
			if output_type == 'm':
				brackets = '[]'
			else:
				brackets = '()'
		
		if kwargs.get('fixed_width',False):
			fmt = '% 7.3f'
		else:
			fmt = '% .3f'
			
		main_str = (fmt + ', ' + fmt + ', '  + fmt) % (self.x, self.y, self.z)
		if kwargs.get('simple',False):
			s = ''
			if brackets:
				s += brackets[0]
			s += main_str
			if brackets:
				s += brackets[1]
			return s
		
		keys = ['frame','name','stamp']
		print_key = _get_print_key(kwargs,keys)
		
		frame_on = print_key['frame']
		stamp_on = print_key['stamp']
		
		stamp_show_zero = kwargs.get('stamp_show_zero',not kwargs.get('stamp_hide_zero',False))
		if kwargs.has_key('stamp'):
			if kwargs['stamp'] == 'show_zero':
				stamp_show_zero = True
				del kwargs['stamp']
			if kwargs['stamp'] == 'hide_zero':
				stamp_show_zero = False
				del kwargs['stamp']
		stamp_kwargs = _prep_stamp_subkwargs(kwargs, ['show_zero','hide_zero'])
		
		stamp_show = self.stamp is not None and (not self.stamp.is_zero() or stamp_show_zero)
		
		if output_type == 'm':
			s = ''
			if frame_on and self.frame:
				s += 'frame:' + self.frame
			if stamp_on and stamp_show:
				if s:
					s += ', '
				s += 'stamp='
				s += self.stamp.tostring(**stamp_kwargs)
			
			if s:
				s+= '\n'
			if brackets:
				s += brackets[0]
			s += main_str
			if brackets:
				s += brackets[1]
			return s
		
		s = ''
		if brackets:
			s += brackets[0]
		
		if frame_on and self.frame:
			s += self.frame + ':'
		
		s += main_str
		
		if stamp_on and stamp_show:
			s += ',stamp='
			s += self.stamp.tostring(**stamp_kwargs)
			
		if brackets:
			s += brackets[1]
		
		return s
	
	def todict(self,**kwargs):
		d = {}
		
		outer = kwargs.get('outer',False)
		short_names = kwargs.get('short_names',False)
		name = kwargs.get('name',None)
		combine_fields = kwargs.get('combine_fields',False)
		
		ros_on = kwargs.get('ros',False)
		frame_on = kwargs.get('frame',True)
		stamp_on = kwargs.get('stamp',True)
		
		as_ros_vector = kwargs.get('vector',False)
		
		if outer:
			if name:
				key = name
			elif ros_on:
				if as_ros_vector:
					key = 'vector'
				else:
					key = 'point'
			elif short_names:
				key = 'p'
			else:
				key = 'position'
			if combine_fields:
				d[key] = self.list
			else:
				inner_kwargs = kwargs.copy()
				inner_kwargs['outer'] = False
				d[key] = self.todict(**inner_kwargs)
			return d
		
		if ros_on:
			header_on = kwargs.get('header',None)
			if header_on is None:
				header_on = (self.frame or self.stamp)
			
			d = {'x': self.x, 'y': self.y, 'z': self.z}
			
			if header_on:
				header_d = {'seq': 0}
				s = self.stamp
				if s is None:
					s = stamp(0)
				header_d['stamp'] = {'secs': s.ros.secs, 'nsecs': s.ros.nsecs}
				if self.frame:
					header_d['frame_id'] = self.frame
				else:
					header_d['frame_id'] = ''
				if as_ros_vector:
					field_name = 'vector'
				else:
					field_name = 'point'
				d = {'header': header_d, field_name: d}
			return d
		
		if combine_fields:
			d['xyz'] = self.list
		else:
			d['x'] = self.x
			d['y'] = self.y
			d['z'] = self.z
		
		if frame_on and self.frame:
			d['frame'] = self.frame
		
		if stamp_on and self.stamp is not None:
			if short_names:
				d['t'] = self.stamp.seconds
			else:
				d['stamp'] = self.stamp.seconds
		
		return d
	
	def __new__(cls,*args,**kwargs):
		super_kwargs = kwargs.copy()
		
		for key in _PARENT_KEYS + _CHILD_KEYS + ['stamp','_master','header']:
			if super_kwargs.has_key(key):
				del super_kwargs[key]
		
		super_kwargs['dtype'] = float
		
		if len(args) == 1:
			arg_data = args[0]
		else:
			arg_data = args
		
		data_type, data = _get_data(arg_data,default_4_to_pt=True)
		if data_type == 'tf':
			data = data[0:3,3].reshape((3,1))
		elif data_type != 'p':
			raise UnknownInputError(type1=data_type,data1=arg_data,input_type='point')
		
		obj = super(CanonicalPoint,cls).__new__(cls,data,**super_kwargs)
		
		return obj
	
	def __init__(self,*args,**kwargs):
		if len(args) == 1:
			arg_data = args[0]
		else:
			arg_data = args
			
		self._stamp = None
		self._frame = None
		
		self._setup_header(kwargs, arg_data)
		
		self._msg = None
		
		self._master = kwargs.get('_master',None)
	
	def _setup_header(self,kwargs,arg_data=None):
		has_header, header_frame, header_stamp = _extract_header_frame_and_stamp(kwargs)
		
		if kwargs.has_key('stamp'):
			if not isinstance(kwargs['stamp'],bool):
				self._stamp = stamp(kwargs['stamp'],copy=True)
			elif kwargs['stamp']:
				self._stamp = stamp('now')
		elif has_header:
			self._stamp = stamp(header_stamp,copy=True)
		elif arg_data is not None:
			self._stamp = _extract_stamp(arg_data)
		
		for key in _PARENT_KEYS:
			if kwargs.has_key(key):
				self._frame = kwargs[key]
				break
		else:
			if has_header:
				self._frame = header_frame
			elif arg_data is not None:
				self._frame = _extract_frame(arg_data)
	
	def __reduce__(self):
		val = list(super(CanonicalPoint,self).__reduce__())
		state = val[2]
		state = (state, (self._frame, self._stamp))
		val[2] = tuple(state)
		return tuple(val)
	
	def __setstate__(self,state):
		ndarray_state = state[0]
		super(CanonicalPoint,self).__setstate__(ndarray_state)
		state = state[1]
		
		setattr(self,'_msg',None)
		setattr(self,'_master',None)
		
		setattr(self,'_frame',state[0])
		setattr(self,'_stamp',state[1])
	
	@property
	def _type_str(self):
		return 'point'
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		kwargs = ''
		if self.frame is not None:
			kwargs += ",frame='%s'" % self.frame
		if self.stamp is not None:
			kwargs += ",stamp=%s" % repr(self.stamp)
			
		return 'point(%f,%f,%f%s)' % (self.x, self.y, self.z,kwargs)
	
	@property
	def _mat_view(self):
		return self.view(type=numpy.matrix)
	
	def __getitem__(self,key):
		if not isinstance(key,tuple):
			return numpy.matrix.__getitem__(self.vector3,(key,0))
		else:
			return numpy.matrix.__getitem__(self.vector3,key)
	
	def __getslice__(self,i,j):
		return numpy.matrix.__getslice__(self.vector3,i,j)
	
	def __setitem__(self,key,value):
		if not isinstance(key,tuple):
			return numpy.matrix.__setitem__(self,(key,0),value)
		else:
			return numpy.matrix.__setitem__(self,key,value)
	
	def _init_after_copy(self,new_obj,**kwargs):
		setattr(new_obj,'_msg',None)
		setattr(new_obj,'_master',None)
		for field in ['_frame', '_stamp']:
			setattr(new_obj,field,None)
			if hasattr(self,field):
				setattr(new_obj,field,copy.copy(getattr(self, field)))
		new_obj._setup_header(kwargs)
	
	def copy(self,**kwargs):
		"""Return a copy of this point.
		
		The x, y, and z values of the copy can be set using keyword args.
		All other keyword args are passed into the CanonicalPoint constructor."""
		new_x = kwargs.pop('x',None)
		new_y = kwargs.pop('y',None)
		new_z = kwargs.pop('z',None)
		new_obj = numpy.matrix.copy(self)
		self._init_after_copy(new_obj,**kwargs)
		if new_x:
			new_obj.x = new_x
		if new_y:
			new_obj.y = new_y
		if new_z:
			new_obj.z = new_z
		return new_obj
	
	def __copy__(self):
		new_obj = numpy.matrix.__copy__(self)
		self._init_after_copy(new_obj)
		return new_obj
	
	def _init_after_add(self,other,pt):
		stamp_to_set = None
		other_stamp = None
		if isinstance(other,CanonicalPoint):
			if self.frame and other.frame and self.frame != other.frame:
				raise FrameError("Trying to add points from frames %s and %s" % (self.frame,other.frame))
			if other.frame:
				pt.frame = other.frame
			other_stamp = other.stamp
		if self.stamp:
			if not other_stamp:
				stamp_to_set = self.stamp
			elif self.stamp > other_stamp:
				stamp_to_set = self.stamp
			else:
				stamp_to_set = other_stamp
		elif other_stamp:
			stamp_to_set = other_stamp
		else:
			stamp_to_set = None
		pt.stamp = copy.copy(stamp_to_set)
		
		if self.frame:
			pt.frame = self.frame
		else:
			pt.frame = None
		return pt
	
	def __add__(self,other):
		if not isinstance(other,numbers.Number):
			other = point(other)
		pt = CanonicalPoint(numpy.matrix.__add__(self._mat_view,other))
			
		return self._init_after_add(other,pt)
	
	def __radd__(self,other):
		if not isinstance(other,numbers.Number):
			other = point(other)
			pt = CanonicalPoint(numpy.matrix.__radd__(self._mat_view,other._mat_view))
		else:
			pt = CanonicalPoint(numpy.matrix.__radd__(self._mat_view,other))
			
		return self._init_after_add(other,pt)
	
	def __iadd__(self,other):
		if not isinstance(other,numbers.Number):
			other = point(other)
		numpy.matrix.__iadd__(self,other)
		self._init_after_add(other, self)
		return self
	
	def __sub__(self,other):
		if not isinstance(other,numbers.Number):
			other = point(other)
		pt = CanonicalPoint(numpy.matrix.__sub__(self._mat_view,other))
			
		return self._init_after_add(other,pt)
	
	def __rsub__(self,other):
		if not isinstance(other,numbers.Number):
			other = point(other)
			pt = CanonicalPoint(numpy.matrix.__rsub__(self._mat_view,other._mat_view))
		else:
			pt = CanonicalPoint(numpy.matrix.__rsub__(self._mat_view,other))
			
		return self._init_after_add(other,pt)
	
	def __isub__(self,other):
		if isinstance(other,collections.Sequence) or isinstance(other,numpy.ndarray):
			other = point(other)
		numpy.matrix.__isub__(self,other)
		self._init_after_add(other, self)
		return self
	
	def __div__(self,other):
		if isinstance(other,CanonicalPoint):
			raise TypeError('Cannot divide a point by another point!')
		elif not isinstance(other,numbers.Number):
			other = numpy.mat(other).reshape(self.shape)
		return CanonicalPoint(numpy.matrix.__div__(self._mat_view,other),frame=self.frame,stamp=self.stamp)
	
	def __rdiv__(self,other):
		if isinstance(other,CanonicalPoint):
			raise TypeError('Cannot divide a point by another point!')
		elif not isinstance(other,numbers.Number):
			other = numpy.mat(other).reshape(self.shape)
		return CanonicalPoint(numpy.matrix.__rdiv__(self._mat_view,other),frame=self.frame,stamp=self.stamp)
	
	def __idiv__(self,other):
		if isinstance(other,CanonicalPoint):
			raise TypeError('Cannot divide a point by another point!')
		elif not isinstance(other,numbers.Number):
			other = numpy.mat(other).reshape(self.shape)
		return numpy.matrix.__idiv__(self,other)
	
	def __mul__(self,other):
		return _canonical_mul(self, other)
	
	def __rmul__(self,other):
		return _canonical_mul(other, self)
	
	def __imul__(self,other):
		pt = _canonical_mul(self, other)
		self[:] = pt[:]
		self.frame = pt.frame
		self.stamp = pt.stamp
		return self

if _ROS:
	class _CanonicalRotationMsgs(object):
		def __init__(self,q):
			self._obj = q
		
		def Quaternion(self):
			"""Get a geometry_msgs/Qaaternion message."""
			return gm.Quaternion(*self._obj.tuple)
		
		def QuaternionStamped(self,**kwargs):
			"""Get a geometry_msgs/QuaternionStamped message.
			
			Args:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			msg = gm.QuaternionStamped()
			msg.header = self.Header(**kwargs)
			msg.quaternion = self.Quaternion()
			return msg
		
		def Header(self,**kwargs):
			"""Get a std_msgs/Header for the frame and stamp.
			
			Args:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			kwargs['default_stamp'] = kwargs.get('default_stamp',True)
			header = std_msgs.msg.Header()
			if self._obj.frame:
				header.frame_id = self._obj.frame
			header.stamp = self.stamp(**kwargs)
			return header
		
		def stamp(self,**kwargs):
			"""Get the stamp as a rospy.Time object.
			
			Args:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			    default_stamp: set to True to use the current time if the stamp is None
			"""
			default_stamp = kwargs.get('default_stamp')
			stamp_now = kwargs.get('stamp_now')
			the_stamp = kwargs.get('stamp')
			if stamp_now or the_stamp is True:
				return stamp('now').ros
			elif the_stamp is not None:
				return stamp(the_stamp).ros
			elif self._obj.stamp is not None:
				return self._obj.stamp.ros
			elif default_stamp:
				return stamp('now').ros
			else:
				return rospy.Time(0)

class _CanonicalRotationEuler(object):
	def __init__(self,master):
		object.__setattr__(self,'_master',master)
	
	def _get(self,axes):
		return tft.euler_from_quaternion(self._master.quaternion, axes)
	
	def _set(self,*args,**kwargs):
		raise TypeError('CanonicalRotation is immutable!')
	
	__call__ = _get
	__getitem__ = _get
	__setitem__ = _set
	__setattr__ = _set
	
	def __getattribute__(self,name):
		if re.search(r'^[rs][xyz]{3}$',name):
			return object.__getattribute__(self,'_get')(name)
		return object.__getattribute__(self,name)
	
	def tostring(self,axes=None,**kwargs):
		if axes is None:
			axes = 'sxyz'
		ai,aj,ak = self._get(axes)
		
		brackets = kwargs.get('brackets')
		if not kwargs.has_key('brackets') or brackets is True:
			brackets = '[]'
		
		deg = None
		rad = None
		default_rad = False
		
		if kwargs.has_key('deg'):
			deg = kwargs['deg']
		if kwargs.has_key('rad'):
			rad = kwargs['rad']
		
		if deg is None and rad is None:
			rad = True
			default_rad = True
		
		if deg is False and rad is False:
			raise ValueError('Cannot set both deg and rad to False!')
		elif deg is None and rad is False:
			deg = True
		elif deg is False and rad is None:
			rad = True
		
		fixed_width = kwargs.get('fixed_width',False)
		
		if fixed_width:
			deg_fmt = '% 6.1f deg'
			rad_fmt = '% 6.3f'
		else:
			deg_fmt = '%.1f deg'
			rad_fmt = '%.3f'
		if not default_rad:
			rad_fmt += ' rad'
		
		
		s = ''
		if brackets:
			s += brackets[0]
		
		s += axes + ':'

		if deg:
			s += deg_fmt % (ai * 180. / pi)
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % ai
			if deg:
				s += ')'
		
		s += ', '
		if deg:
			s += deg_fmt % (aj * 180. / pi)
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % aj
			if deg:
				s += ')'
	
		s += ', '
		if deg:
			s += deg_fmt % (ak * 180. / pi)
		if rad:
			if deg:
				s += ' ('
			s += rad_fmt % ak
			if deg:
				s += ')'
		if brackets:
			s += brackets[1]
		
		return s
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		axes = 'sxyz'
		ai,aj,ak = self._get(axes)
		return "rotation_euler(%f,%f,%f,'%s')" % (ai,aj,ak,axes)

class CanonicalRotation(numpy.matrix):
	@staticmethod
	def random(**kwargs):
		"""Return a random rotation. Keyword args are passed into the constructor."""
		return CanonicalRotation(tft.random_quaternion(),**kwargs)

	@staticmethod
	def identity(**kwargs):
		"""Return an identity rotation. Keyword args are passed to the constructor."""
		return CanonicalRotation([0,0,0,1],**kwargs)
	
	@property
	def frame(self):
		"""The frame of this rotation."""
		if self._master is not None:
			return self._master._frame
		else:
			return self._frame
	@frame.setter
	def frame(self,value):
		if self._master is not None:
			raise DerivedObjectError('This rotation is derived from another object, the frame cannot be changed')
		self._frame = value
	parent = frame
	
	@property
	def stamp(self):
		"""The timestamp as a CanonicalStamp."""
		if self._master is not None:
			return self._master.stamp
		else:
			return self._stamp
	@stamp.setter
	def stamp(self,value):
		if self._master is not None:
			raise DerivedObjectError('This rotation is derived from another object, the stamp cannot be changed')
		elif value is None:
			self._stamp = None
		elif not self._stamp:
			self._stamp = stamp(value,copy=True)
		else:
			self._stamp.set(value)
	def set_stamp_now(self):
		self.stamp = 'now'
	
	@property
	def x(self):
		return self[0]
	
	@property
	def y(self):
		return self[1]
	
	@property
	def z(self):
		return self[2]
	
	@property
	def w(self):
		return self[3]
	
	@property
	def quaternion(self):
		"""The quaternion as a numpy array."""
		if self._master is not None:
			q = tft.quaternion_from_matrix(self._master)
			if q[3] < 0:
				q = -q
			q.flags.writeable = False
		elif self._quaternion is None:
			M = numpy.identity(4)
			M[0:3,0:3] = self
			q = tft.quaternion_from_matrix(M)
			if q[3] < 0:
				q = -q
			q.flags.writeable = False
			self._quaternion = q
		else:
			q = self._quaternion
		return q
	
	@property
	def matrix(self):
		"""A view of this rotation as a 3x3 numpy matrix (not array)."""
		M = self.view(dtype=numpy.matrix)
		M.flags.writeable = False
		return M
	
	@property
	def array(self):
		"""A view of this rotation as a 3x3 numpy array."""
		M = self.view(dtype=numpy.ndarray)
		M.flags.writeable = False
		return M
	
	@property
	def list(self):
		"""A quaternion representation as a list."""
		return _convert_to_list(self.quaternion)
	
	@property
	def tuple(self):
		"""A quaternion representation as a tuple, with x, y, z, and w fields."""
		return QuaternionTuple._make(_convert_to_tuple(self.quaternion))
	
	@property
	def dict(self):
		return self.todict()
	
	@property
	def tb_angles(self):
		"""A tb_angles object representing this rotation."""
		return get_tb_angles(self)
	
	@property
	def angle(self):
		"""The magnitude of this rotation."""
		xyz_norm = numpy.linalg.norm([self.x,self.y,self.z])
		angle = 2. * atan2(xyz_norm, self.w)
		return abs(angle)
	@property
	def axis_angle(self):
		"""The axis and angle representing this rotation."""
		xyz = numpy.array([self.x,self.y,self.z])
		xyz_norm = numpy.linalg.norm(xyz)
		axis = xyz.copy()
		axis = axis / xyz_norm
		angle = 2. * atan2(xyz_norm, self.w)
		if angle < 0:
			axis = -axis
			angle = -angle
		return axis,angle
	
	def inverse(self):
		"""Return the inverse of this rotation."""
		return rotation(tft.quaternion_inverse(self.quaternion))
	
	@property
	def euler(self):
		"""Access to Euler angles for this rotation.
		
		Euler angles can be retrieved from the returned object by calling,
		indexing, or field access using the desired axes. The axes must be 
		given as (r|s)<axes>, where 'r' indicates rotating axes and 's' indicates
		stationary axes. <axes> must be the axes to rotate about. For example, 
		stationary xyz would be given as sxyz."""
		if self._euler is None:
			self._euler = _CanonicalRotationEuler(self)
		return self._euler
	
	def apply(self,value,**kwargs):
		"""Method version of * operator, with options.
		
		Constructor keyword arguments can be used to set the fields of the
		returned object."""
		return _canonical_mul(self, value, **kwargs)
	
	def apply_to_many(self,value,axis=None,canonical=False,**kwargs):
		"""Apply this rotation to multiple objects.
		
		If the input is a sequence of objects, a list of canonical-ized values
		will be returned.
		
		If the input is a numpy array, the input will be split up along either
		the first axis (the default) or the last axis (which can be specified
		either absolutely or as -1). Splitting along the first axis means that
		to apply this rotation to N points, the input should be Nx3.
		Assuming splitting along the first axis, the input can be:
		Nx4x4: transforms or poses
		Nx3x3: rotation matrices
		Nx3 or Nx3x1: points
		
		The returned value will have the same shape as the input, unless the
		keyword argument 'canonical' is True, in which case the output will
		be a list of the canonical-ized resulting values.
		
		Keyword arguments:
		    axis: The axis to split along, currently only the first or last.
		    canonical: Force the output to be a list of canonical objects.
		    Constructor keyword arguments can be used to set the fields of the
		        returned objects if they are canonical."""
		if isinstance(value,collections.Sequence):
			ret_value = []
			for val in value:
				ret_value.append(self * val)
			return ret_value
		
		value = numpy.array(value)
		if canonical:
			ret_value = []
		else:
			ret_value = numpy.empty_like(value)
		if axis is None:
			axis = 0
		size_on_axis = value.shape[axis]
		remaining_size = value.shape[:axis] + value.shape[axis+1:]
		_sl = [slice(None,None,None)] * (value.ndim-1)
		def get_slice(i):
			if axis == 0:
				return tuple([i] + _sl)
			elif axis == -1 or axis == value.ndim-1:
				return tuple(_sl + [i])
			else:
				ValueError("Axis only allowed to be 0, -1, or ndim-1")
		for i in xrange(value.shape[axis]):
			sl = get_slice(i)
			val = value[sl]
			if canonical:
				ret_value.append(self * val)
			else:
				if val.shape == (4,4):
					ret_value[sl] = self.as_tf().matrix * val
				if val.shape == (3,3):
					ret_value[sl] = self.matrix * val
				elif val.size == 3:
					ret_value[sl] = (self * val).array
				elif val.size == 4:
					raise NotImplementedError("I don't understand 4-vectors yet!")
				else:
					raise TypeError("Unknown input!")
			
		return ret_value
	
	def as_transform(self,**kwargs):
		"""Return a transform with this rotation.
		Keyword args are passed into the constructor."""
		kwargs['frame'] = kwargs.get('frame',self.frame)
		kwargs['stamp'] = kwargs.get('stamp',self.stamp)
		return transform(self.matrix,**kwargs)
	as_tf = as_transform
	def as_pose(self,**kwargs):
		"""Return a pose with this orientation.
		Keyword args are passed into the constructor."""
		kwargs['frame'] = kwargs.get('frame',self.frame)
		kwargs['stamp'] = kwargs.get('stamp',self.stamp)
		return pose(self.matrix,**kwargs)
	
	def interpolate(self,other,fraction, spin=0, shortestpath=True):
		"""Interpolate between this and another rotation using quaternion slerp."""
		other = rotation(other)
		return rotation(tft.quaternion_slerp(self.quaternion,other.quaternion,float(fraction),spin=spin,shortestpath=shortestpath))
	slerp = interpolate
	
	def capped(self,angle):
		"""Return a copy whose angle is at most the input value"""
		ax, a = self.axis_angle
		a = min(a,angle)
		return rotation(ax,a,frame=self.frame,stamp=self.stamp)
	
	def rotation_to(self,other):
		"""Get the rotation which, when applied to this rotation
		by right-multiplication, produces the input rotation"""
		other = rotation(other)
		if self.frame and other.frame and self.frame != other.frame:
			raise FrameError("Can't get rotation between rotations in frames %s and %s" % (self.frame,other.frame))
		out = rotation(tft.quaternion_multiply(tft.quaternion_inverse(self.quaternion), other.quaternion))
		_init_after_quat_mul(other,self,out)
		return out
	
	def rotation_from(self,other):
		"""Get the rotation which, when applied to the input rotation
		by right-multiplication, produces this rotation"""
		other = rotation(other)
		if self.frame and other.frame and self.frame != other.frame:
			raise FrameError("Can't get rotation between rotations in frames %s and %s" % (self.frame,other.frame))
		out = rotation(tft.quaternion_multiply(tft.quaternion_inverse(other.quaternion), self.quaternion))
		_init_after_quat_mul(self,other,out)
		return out
	
	if _ROS:
		@property
		def msg(self):
			"""A ROS message creator object.
			
			The message creator has methods for the various ROS message types:
			Quaternion
			QuaternionStamped
			Header
			
			and an additional method, stamp, which returns a rospy.Time object for 
			the stamp.
			
			The QuaternionStamped, Header, and stamp methods all take
			two keyword arguments:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			"""
			if self._msg is None:
				self._msg = _CanonicalRotationMsgs(self)
			return self._msg
	
	def tostring(self,**kwargs):
		if kwargs.get('numpy',kwargs.get('np',False)):
			return super(CanonicalRotation,self).tostring()
		
		brackets = kwargs.get('brackets')
		if not kwargs.has_key('brackets') or brackets is True:
			brackets = '[]'
		
		fmt = _get_rot_string_fmt(kwargs)
		
		if fmt == 'q':
			main_str = '% .3f,% .3f,% .3f,% .3f' % (self.x, self.y, self.z, self.w)
		elif fmt == 'tb':
			tb_kwargs = {'brackets': None}
			for key in ['deg','rad','short_names','fixed_width']:
				if kwargs.has_key(key):
					tb_kwargs[key] = kwargs[key]
			main_str = self.tb_angles.tostring(**tb_kwargs)
		elif fmt == 'm':
			brackets = ''
			main_str = numpy.matrix.__str__(self.matrix)
		
		if kwargs.get('simple',False):
			s = ''
			if brackets:
				s += brackets[0]
			s += main_str
			if brackets:
				s += brackets[1]
			return s
		
		keys = ['frame','stamp']
		print_key = _get_print_key(kwargs,keys)
		
		frame_on = print_key['frame']
		stamp_on = print_key['stamp']
		
		stamp_show_zero = kwargs.get('stamp_show_zero',not kwargs.get('stamp_hide_zero',False))
		if kwargs.has_key('stamp'):
			if kwargs['stamp'] == 'show_zero':
				stamp_show_zero = True
				del kwargs['stamp']
			if kwargs['stamp'] == 'hide_zero':
				stamp_show_zero = False
				del kwargs['stamp']			
		stamp_kwargs = _prep_stamp_subkwargs(kwargs, ['show_zero','hide_zero'])
		
		stamp_show = self.stamp is not None and (not self.stamp.is_zero() or stamp_show_zero)
		
		if fmt == 'm':
			s = ''
			if frame_on and self.frame:
				s += 'frame:' + self.frame
			if stamp_on and stamp_show:
				if s:
					s += ', '
				s += 'stamp='
				s += self.stamp.tostring(**stamp_kwargs)
			
			if s:
				s+= '\n'
			s += main_str
			return s
		
		s = ''
		if brackets:
			s += brackets[0]
		
		if frame_on and self.frame:
			s += self.frame + ':'
		
		s += main_str
		
		if stamp_on and stamp_show:
			s += ',stamp='
			s += self.stamp.tostring(**stamp_kwargs)
			
		if brackets:
			s += brackets[1]
		
		return s
	
	def todict(self,**kwargs):
		d = {}
		
		outer = kwargs.get('outer',False)
		short_names = kwargs.get('short_names',False)
		name = kwargs.get('name',None)
		combine_fields = kwargs.get('combine_fields',False)
		
		ros_on = kwargs.get('ros',False)
		
		quat_on = None
		for key in ['q','quat','quaternion']:
			if kwargs.has_key(key):
				quat_on = kwargs[key]
				break
		mat_on = None
		for key in ['m','mat','matrix']:
			if kwargs.has_key(key):
				mat_on = kwargs[key]
				break
		tb_on = False
		for key in ['tb','tb_angles']:
			if kwargs.has_key(key):
				tb_on = kwargs[key]
				break
		
		default_quat = False
		if quat_on is None and mat_on is None and not tb_on:
			quat_on = True
			default_quat = True
			
		if not tb_on:
			if quat_on is False and mat_on is False:
				raise ValueError('Cannot set both quat and mat to False!')
			elif quat_on is None and mat_on is False:
				quat_on = True
			elif quat_on is False and mat_on is None:
				mat_on = True
		
		
		frame_on = kwargs.get('frame',True)
		stamp_on = kwargs.get('stamp',True)
		
		if outer:
			if name:
				key = name
			elif ros_on:
				key = 'quaternion'
			elif short_names:
				key = 'r'
			else:
				key = 'rotation'
			
			if combine_fields and default_quat:
				d[key] = self.list
			else:
				inner_kwargs = kwargs.copy()
				inner_kwargs['outer'] = False
				d[key] = self.todict(**inner_kwargs)
			return d
		
		if ros_on:
			header_on = kwargs.get('header',None)
			if header_on is None:
				header_on = (self.frame or self.stamp)
			
			d = {'x': self.x, 'y': self.y, 'z': self.z, 'w': self.w}
			
			if header_on:
				header_d = {'seq': 0}
				s = self.stamp
				if s is None:
					s = stamp(0)
				header_d['stamp'] = {'secs': s.ros.secs, 'nsecs': s.ros.nsecs}
				if self.frame:
					header_d['frame_id'] = self.frame
				else:
					header_d['frame_id'] = ''
				d = {'header': header_d, 'quaternion': d}
			return d
		
		if combine_fields:
			if default_quat:
				d['xyzw'] = self.list
			else:
				if quat_on:
					if short_names:
						key = 'q'
					else:
						key = 'quaternion'
					d[key] = self.list
				if mat_on:
					if short_names:
						key = 'm'
					else:
						key = 'matrix'
					d[key] = self.matrix.tolist()
				if tb_on:
					if short_names:
						key = 'tb'
						yaw_key = 'y'
						pitch_key = 'p'
						roll_key = 'r'
					else:
						key = 'tb_angles'
						yaw_key = 'yaw'
						pitch_key = 'pitch'
						roll_key = 'roll'
					d[key] = {yaw_key: self.tb_angles.yaw_deg, pitch_key: self.tb_angles.pitch_deg, roll_key: self.tb_angles.roll_deg}
		else:
			
			if default_quat:
				d['qx'] = self.x
				d['qy'] = self.y
				d['qz'] = self.z
				d['qw'] = self.w
			else:
				if quat_on:
					if short_names:
						key = 'q'
					else:
						key = 'quaternion'
					d[key] = {'qx': self.x, 'qy': self.y, 'qz': self.z, 'qw': self.w}
				if mat_on:
					if short_names:
						key = 'm'
					else:
						key = 'matrix'
					d[key] = self.matrix.tolist()
				if tb_on:
					if short_names:
						key = 'tb'
						yaw_key = 'y'
						pitch_key = 'p'
						roll_key = 'r'
					else:
						key = 'tb_angles'
						yaw_key = 'yaw'
						pitch_key = 'pitch'
						roll_key = 'roll'
					d[key] = {yaw_key: self.tb_angles.yaw_deg, pitch_key: self.tb_angles.pitch_deg, roll_key: self.tb_angles.roll_deg}
		
		if frame_on and self.frame:
			d['frame'] = self.frame
		
		if stamp_on and self.stamp is not None:
			if short_names:
				d['t'] = self.stamp.seconds
			else:
				d['stamp'] = self.stamp.seconds
		
		return d	
	
	def __new__(cls,*args,**kwargs):
		super_kwargs = kwargs.copy()
		
		for key in _PARENT_KEYS + _CHILD_KEYS + ['stamp','_master','header']:
			if super_kwargs.has_key(key):
				del super_kwargs[key]
		
		super_kwargs['dtype'] = float
		
		if not args:
			data = numpy.identity(3)
		else:
			if len(args) == 1:
				arg_data = args[0]
			else:
				arg_data = args
			
			data_type, data = _get_data(arg_data,default_4_to_quat=True)
			if data_type in ['tf','ps']:
				data_type = 'r'
				data = data[0:3,0:3]
			
			if data_type == 'q':
				data_type = 'r'
				data = tft.quaternion_matrix(data)[0:3,0:3]
			
			if data_type != 'r':
				raise UnknownInputError(type1=data_type,data1=arg_data)
		
		if numpy.isnan(data).any():
			raise UnknownInputError(msg = "Input results in invalid rotation! %s" % str(args))
		
		obj = super(CanonicalRotation,cls).__new__(cls,data,**super_kwargs)
		
		return obj
	
	def __init__(self,*args,**kwargs):
		self.flags.writeable = False
		
		if len(args) == 1:
			arg_data = args[0]
		else:
			arg_data = args
		
		self._stamp = None
		self._frame = None
		
		self._setup_header(kwargs, arg_data)

		self._quaternion = None
		
		self._msg = None
		
		self._euler = None
		
		self._master = kwargs.get('_master',None)
	
	def _setup_header(self,kwargs,arg_data=None):
		has_header, header_frame, header_stamp = _extract_header_frame_and_stamp(kwargs)
		
		if kwargs.has_key('stamp'):
			if not isinstance(kwargs['stamp'],bool):
				self._stamp = stamp(kwargs['stamp'],copy=True)
			elif kwargs['stamp']:
				self._stamp = stamp('now')
		elif has_header:
			self._stamp = stamp(header_stamp,copy=True)
		elif arg_data is not None:
			self._stamp = _extract_stamp(arg_data)
		
		for key in _PARENT_KEYS:
			if kwargs.has_key(key):
				self._frame = kwargs[key]
				break
		else:
			if has_header:
				self._frame = header_frame
			elif arg_data is not None:
				self._frame = _extract_frame(arg_data)
	
	def __reduce__(self):
		val = list(super(CanonicalRotation,self).__reduce__())
		state = val[2]
		state = (state, (self._frame, self._stamp))
		val[2] = tuple(state)
		return tuple(val)
	
	def __setstate__(self,state):
		ndarray_state = state[0]
		super(CanonicalRotation,self).__setstate__(ndarray_state)
		state = state[1]
		
		setattr(self,'_msg',None)
		setattr(self,'_master',None)
		setattr(self,'_euler',None)
		setattr(self,'_quaternion',None)
		
		setattr(self,'_frame',state[0])
		setattr(self,'_stamp',state[1])
	
	@property
	def _type_str(self):
		return 'rotation'
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		kwargs = ''
		if self.frame is not None:
			kwargs += ",frame='%s'" % self.frame
		if self.stamp is not None:
			kwargs += ",stamp=%s" % repr(self.stamp)
		
		return 'rotation(%f,%f,%f,%f%s)' % (self.x, self.y, self.z, self.w, kwargs)
	
	def __getitem__(self,key):
		if isinstance(key,tuple):
			return numpy.matrix.__getitem__(self,key)
		else:
			return numpy.ndarray.__getitem__(self.quaternion,key)
	
	def __setitem__(self,key,value):
		raise TypeError("CanonicalRotation is immutable!")
	
	def __getattribute__(self,name):
		regex = r'^euler_[sr][xyz]{3}$'
		if re.match(regex,name):
			axes = name[-4:]
			return self.euler(axes)
		else:
			return numpy.matrix.__getattribute__(self,name)
	
	def _init_after_copy(self,new_obj,**kwargs):
		setattr(new_obj,'_msg',None)
		setattr(new_obj,'_master',None)
		setattr(new_obj,'_euler',None)
		for field in ['_frame', '_stamp', '_quaternion']:
			setattr(new_obj,field,None)
			if hasattr(self,field):
				setattr(new_obj,field,copy.copy(getattr(self, field)))
		new_obj._setup_header(kwargs)
	
	def copy(self,**kwargs):
		"""Return a copy of this rotation.
		Keyword args are passed to the constructor."""
		new_obj = numpy.matrix.copy(self)
		self._init_after_copy(new_obj,**kwargs)
		return new_obj
	
	def __copy__(self):
		new_obj = numpy.matrix.__copy__(self)
		self._init_after_copy(new_obj)
		return new_obj
	
	def __mul__(self,other):
		return _canonical_mul(self, other)
	
	def __rmul__(self,other):
		return _canonical_mul(other, self)
	
	def _bad_op(self,other):
		return NotImplemented

	__imul__ = _bad_op
	__add__ = _bad_op
	__radd__ = _bad_op
	__iadd__ = _bad_op
	__sub__ = _bad_op
	__rsub__ = _bad_op
	__isub__ = _bad_op
	__div__ = _bad_op
	__rdiv__ = _bad_op
	__idiv__ = _bad_op

if _ROS:
	class _CanonicalTransformMsgs(object):
		def __init__(self,transform):
			self._obj = transform
		
		def Pose(self):
			"""Get a geometry_msgs/Pose message."""
			pose_p = self._obj.translation.tuple
			pose_q = self._obj.rotation.tuple
			msg = gm.Pose()
			msg.position = gm.Point(*pose_p)
			msg.orientation = gm.Quaternion(*pose_q)
			return msg
		
		def PoseStamped(self,**kwargs):
			"""Get a geometry_msgs/PoseStamped message.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			msg = gm.PoseStamped()
			msg.header = self.Header(**kwargs)
			msg.pose = self.Pose()
			return msg
		
		def Transform(self):
			"""Get a geometry_msgs/Transform message."""
			pose_p = self._obj.translation.tuple
			pose_q = self._obj.rotation.tuple
			msg = gm.Transform()
			msg.translation = gm.Vector3(*pose_p)
			msg.rotation = gm.Quaternion(*pose_q)
			return msg
	
		def TransformStamped(self,**kwargs):
			"""Get a geometry_msgs/TransformStamped message.
			
			Args:
			    check_frame: set to False to allow transforms with missing frames
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			if kwargs.get('check_frame',kwargs.get('check_frames',True)) and (not self._obj._parent or not self._obj._child): 
				raise FrameError('Parent or child frame missing!')
			msg = gm.TransformStamped()
			msg.header = self.Header(**kwargs)
			if self._obj._child:
				msg.child_frame_id = self._obj._child
			msg.transform = self.Transform()
			return msg
	
		def tfMessage(self, stamp_now=False, stamp=None, append_to=None):
			"""Create or append to a tfMessage.
			
			If no stamp is set, the TransformStamped message will have
			the current stamp.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: use the current time instead of the stamp.
			    append_to: set this to a tfMessage object to append the
			        TransformStamped object to it."""
			msg = self.TransformStamped(stamp_now=stamp_now, stamp=stamp)
			if append_to:
				append_to.transforms.append(msg)
				return
			else:
				msgs = tf.msg.tfMessage()
				msgs.transforms.append(msg)
				return msgs
		
		def Header(self,**kwargs):
			"""Get a std_msgs/Header for the frame and stamp.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to False to use the stamp even if it's None
			"""
			kwargs['default_stamp'] = kwargs.get('default_stamp',True)
			header = std_msgs.msg.Header()
			if self._obj._frame:
				header.frame_id = self._obj._frame
			header.stamp = self.stamp(**kwargs)
			return header
		
		def stamp(self,**kwargs):
			"""Get the stamp as a rospy.Time object.
			
			Args:
			    stamp: a stamp to use, or True to use the current time
			    stamp_now: always use the current time
			    default_stamp: set to True to use the current time if the stamp is None
			"""
			default_stamp = kwargs.get('default_stamp')
			stamp_now = kwargs.get('stamp_now')
			the_stamp = kwargs.get('stamp')
			if stamp_now or the_stamp is True:
				return stamp('now').ros
			elif the_stamp is not None:
				return stamp(the_stamp).ros
			elif self._obj.stamp is not None:
				return self._obj.stamp.ros
			elif default_stamp:
				return stamp('now').ros
			else:
				return rospy.Time(0)

class CanonicalTransform(numpy.matrix):
	"""A class representing a transform or pose."""
	
	@staticmethod
	def identity(**kwargs):
		"""Return an identity transform or pose.
		Keyword args are passed to the constructor."""
		return CanonicalTransform(numpy.identity(4),**kwargs)
	
	@staticmethod
	def random(scale=1,mean=0,dist='uniform',**kwargs):
		"""Return a transform or pose with random rotation and translation.
		Keyword args are passed to the constructor."""
		t = CanonicalTransform.random_translation(scale=scale,mean=0,dist=dist) * CanonicalTransform.random_rotation()
		return CanonicalTransform(t,**kwargs)
	
	@staticmethod
	def random_translation(scale=1,mean=0,dist='uniform',**kwargs):
		"""Return a transform or pose with identity rotation and random translation.
		
		Options for distribution:
		'uniform', 'normal' or 'gaussian'
		
		Keyword args are passed into the constructor."""
		if isinstance(scale,numbers.Number):
			scale = float(scale) * numpy.ones(3, dtype=float)
		else:
			scale = numpy.array(scale)
		if isinstance(mean,numbers.Number):
			mean = float(mean) * numpy.ones(3, dtype=float)
		else:
			mean = numpy.array(mean)
		dist = kwargs.pop('distribution',dist)
		if dist == 'uniform':
			data = scale * numpy.random.rand(3) + mean
		elif dist in ['normal','gaussian']:
			data = scale * numpy.random.randn(3) + mean
		return CanonicalTransform(data,**kwargs)
	
	@staticmethod
	def random_rotation(**kwargs):
		"""Return a transform or pose with random rotation and no translation.
		Keyword args are passed to the constructor."""
		return CanonicalTransform(tft.random_quaternion(),**kwargs)
	
	@property
	def pose(self):
		return self._pose
	@pose.setter
	def pose(self,value):
		prev_pose = self._pose
		new_pose = bool(value)
		self._pose = new_pose
		if prev_pose != new_pose or prev_pose is None:
			self._update_frame_fields()
	
	def _update_frame_fields(self):
		if self._pose:
			[object.__delattr__(self,attr) for attr in ['parent','to_frame'] if attr in self.__dict__]
			[object.__setattr__(self,attr,'<dummy>') for attr in ['frame']]
			
			[object.__delattr__(self,attr) for attr in ['child','from_frame'] if attr in self.__dict__]
			[object.__setattr__(self,attr,'<dummy>') for attr in ['name']]
		else:
			[object.__setattr__(self,attr,'<dummy>') for attr in ['frame']]
			#[object.__delattr__(self,attr) for attr in ['frame'] if hasattr(self,attr)]
			
			[object.__setattr__(self,attr,'<dummy>') for attr in ['parent','to_frame']]
			
			[object.__delattr__(self,attr) for attr in ['name'] if attr in self.__dict__]
			[object.__setattr__(self,attr,'<dummy>') for attr in ['child','from_frame']]
	
	@staticmethod
	def frame_README():
		"""A transform has properties from_frame and to_frame, which can also be accessed as
		child and parent, respectively. A pose has attributes frame and name.
		When converting a transform to a pose and vice versa, frame is mapped to parent
		and name is mapped to child. This corresponds to the transform being the pose of
		the child axes in the parent frame."""
		
		return """A transform has properties from_frame and to_frame, which can also be accessed as
child and parent, respectively. A pose has attributes frame and name.
When converting a transform to a pose and vice versa, frame is mapped to parent
and name is mapped to child. This corresponds to the transform being the pose of
the child axes in the parent frame."""
	@property
	def _frame_property(self):
		"""The parent (destination) frame of the transform, or the frame of the pose."""
		return self._frame
	@_frame_property.setter
	def _frame_property(self,value):
		self._frame = value
	_parent = _frame_property
	
	@property
	def _name_property(self):
		"""The child (source) frame of the transform, or a name for the pose."""
		return self._name
	@_name_property.setter
	def _name_property(self,value):
		self._name = value
	_child = _name_property
	
	@property
	def frame_str(self):
		"""A string representation of the frame of this transform or pose.
		
		For poses, this is simply the frame.
		For transforms, it is a string of the form 'parent<-child', that is,
		this transform takes objects in the child frame and transforms them into
		the parent frame. This format allows for more intuitive understanding of
		concatenating transforms."""
		if self.pose:
			if self._frame:
				return self._frame
			else:
				return ''
		frame_str = ''
		if self._parent:
			frame_str += self._parent
		else:
			frame_str += '?'
		frame_str += '<-'
		if self._child:
			frame_str += self._child
		else:
			frame_str += '?'
		return frame_str
	@frame_str.setter
	def frame_str(self,frames):
		try:
			if frames.find('<-') != -1:
				self._frame, self._name = frames.split('<-')
				return
			elif frames.find('->') != -1:
				self._name, self._frame = frames.split('->')
				return
		except:
			pass
		raise TypeError('Frames %s cannot be parsed' % frames)
	
	@property
	def stamp(self):
		"""The timestamp as a CanonicalStamp."""
		return self._stamp
	@stamp.setter
	def stamp(self,value):
		if value is None:
			self._stamp = None
		elif not self._stamp:
			self._stamp = stamp(value,copy=True)
		else:
			self._stamp.set(value)
	def set_stamp_now(self):
		self.stamp = 'now'

	@property
	def matrix(self):
		"""A view of this transform or pose as a 4x4 numpy matrix (not array)."""
		return self.view(type=numpy.matrix)
	@matrix.setter
	def matrix(self,value):
		if value.shape == (3,3):
			self[0:3,3] = 0
			self[0:3,0:3] = value
		else:
			self[0:4,0:4] = value

	@property
	def array(self):
		"""A view of this transform or pose as a 4x4 numpy array."""
		return self.view(type=numpy.ndarray)
	@array.setter
	def array(self,value):
		if value.shape == (3,3):
			self[0:3,3] = 0
			self[0:3,0:3] = value
		else:
			self[0:4,0:4] = value

	@property
	def translation(self):
		"""The translational component as a CanonicalPoint.
		Modifying the point will modify this object."""
		pt = self[0:3,3].view(type=CanonicalPoint)
		pt.__init__(frame=self._frame,stamp=self.stamp,_master=self)
		return pt
	@translation.setter
	def translation(self,value):
		if value is None:
			self[0:3,3] = 0
		elif _get_type(value) == 's':
			self[0:3,3] = value
		else:
			pt = point(value)
			self[0:3,3] = pt.vector3
		
	position = translation

	@property
	def rotation(self):
		"""The rotational component as a CanonicalRotation.
		Modifying the rotation will modify this object."""
		r = self[0:3,0:3].view(type=CanonicalRotation)
		r.__init__(frame=self._frame,stamp=self.stamp,_master=self)
		return r
	@rotation.setter
	def rotation(self,value):
		if value is None:
			self[0:3,0:3] = numpy.eye(3)
		else:
			r = rotation(value)
			self[0:3,0:3] = r.matrix
	
	orientation = rotation
	quaternion = rotation
	
	@property
	def tb_angles(self):
		"""The rotational component as Tait-Bryan angles."""
		return get_tb_angles(self)
	@tb_angles.setter
	def tb_angles(self,value):
		tb = get_tb_angles(value)
		self[0:3,0:3] = tb.matrix
	
	@property
	def dict(self):
		"""A dictionary representing this object"""
		return self.todict()
	
	if _ROS:
		@property
		def msg(self):
			"""A ROS message creator object.
			
			The message creator has methods for the various ROS message types:
			Pose
			PoseStamped
			Transform
			TransformStamped
			tfMessage
			Header
			
			and an additional method, stamp, which returns a rospy.Time object for 
			the stamp.
			
			The PoseStamped, TransformStamped, Header, and stamp methods all take
			two keyword arguments:
			    default_stamp: if the stamp is None, use the current time
			    stamp_now: always use the current time
			
			tfMessage has two keyword arguments:
			    stamp_now: like above. tfMessage will always default the stamp to now.
			    append_to: if set to a tfMessage object, the TransformStamped message
			    will be appended to it.
			"""
			if self._msg is None:
				self._msg = _CanonicalTransformMsgs(self)
			return self._msg
	
	def inverse(self,newname=None):
		"""Return the inverse of this transform or pose.
		
		If this is a transform and both parent and child frames are specified,
		they will be flipped in the result.
		Otherwise, the specified frame will remain."""
		inv = tft.inverse_matrix(self.matrix)
		if self.pose:
			newframe = self._frame
		elif self._frame and self._name:
			newframe = self._name
			newname = self._frame
		else:
			newframe = self._frame
			newname = self._name
			
		return CanonicalTransform(inv,frame=newframe,name=newname,stamp=self.stamp,pose=self.pose)

	def as_transform(self,**kwargs):
		"""Return a copy as a transform."""
		if not kwargs.has_key('pose'):
			kwargs['pose'] = False
		kwargs['copy'] = True
		return transform(self,**kwargs)
	as_tf = as_transform
	def as_pose(self,**kwargs):
		"""Return a copy as a pose."""
		kwargs['copy'] = True
		return pose(self,**kwargs)
	
	@property
	def distance_and_angle(self):
		"""Return the magnitudes of the translation and rotation components"""
		return self.position.norm, self.rotation.angle
	
	def apply(self,value,**kwargs):
		"""Method version of * operator, with options.
		
		Constructor keyword arguments can be used to set the fields of the
		returned object.
		
		Additionally, the 'keep_frames' keyword argument can be used when
		applying to another transform. By default, the resulting transform will
		have the parent of this transform and the child of the input. However,
		if keep_frames is True and one of the transforms lacks frames, the frames
		of the other will be used."""
		return _canonical_mul(self, value, **kwargs)
	
	def apply_to_many(self,value,axis=None,canonical=False,**kwargs):
		"""Apply this transform to multiple objects.
		
		If the input is a sequence of objects, a list of canonical-ized values
		will be returned.
		
		If the input is a numpy array, the input will be split up along either
		the first axis (the default) or the last axis (which can be specified
		either absolutely or as -1). Splitting along the first axis means that
		to apply this rotation to N points, the input should be Nx3.
		Assuming splitting along the first axis, the input can be:
		Nx4x4: transforms or poses
		Nx3x3: rotation matrices
		Nx3 or Nx3x1: points
		
		The returned value will have the same shape as the input, unless the
		keyword argument 'canonical' is True, in which case the output will
		be a list of the canonical-ized resulting values.
		
		Keyword arguments:
		    axis: The axis to split along, currently only the first or last.
		    canonical: Force the output to be a list of canonical objects.
		    Constructor keyword arguments can be used to set the fields of the
		        returned object."""
		if self.pose:
			raise TypeError("Can't left-multiply a pose!")
		if isinstance(value,collections.Sequence):
			ret_value = []
			for val in value:
				ret_value.append(self * val)
			return ret_value
		
		value = numpy.array(value)
		if canonical:
			ret_value = []
		else:
			ret_value = numpy.empty_like(value)
		if axis is None:
			axis = 0
		size_on_axis = value.shape[axis]
		remaining_size = value.shape[:axis] + value.shape[axis+1:]
		_sl = [slice(None,None,None)] * (value.ndim-1)
		def get_slice(i):
			if axis == 0:
				return tuple([i] + _sl)
			elif axis == -1 or axis == value.ndim-1:
				return tuple(_sl + [i])
			else:
				ValueError("Axis only allowed to be 0, -1, or ndim-1")
		for i in xrange(value.shape[axis]):
			sl = get_slice(i)
			val = value[sl]
			if canonical:
				ret_value.append(self * val)
			else:
				if val.shape == (4,4):
					ret_value[sl] = self.matrix * val
				if val.shape == (3,3):
					ret_value[sl] = self.rotation.matrix * val
				elif val.size == 3:
					ret_value[sl] = (self * val).array
				elif val.size == 4:
					raise NotImplementedError("I don't understand 4-vectors yet!")
				else:
					raise TypeError("Unknown input!")
			
		return ret_value
	
	def interpolate(self,other,*args,**kwargs):
		"""Interpolate both translation and rotation. A single value can be given
		to interpolate both simultaneously, or separate values can be given as
		keyword arguments using the field names."""
		other = transform(other)
		fraction = None
		if args:
			if len(args) > 1:
				raise TypeError('Only a single fraction can be specified with positional args')
			fraction = args[0]
		
		pt_fraction = _get_kwarg(kwargs, _EXTENDED_POS_KEYS)
		rot_fraction = _get_kwarg(kwargs, _EXTENDED_ROT_KEYS)
		if fraction is None and all(frac is None or isinstance(frac,bool) for frac in [pt_fraction, rot_fraction]):
			raise ValueError('No interpolation value specified!')
		elif pt_fraction is False and rot_fraction is False:
			raise ValueError("Position and rotation can't both be disabled!")
		
		if pt_fraction is False and rot_fraction is None:
			pt_fraction = 0
			rot_fraction = fraction
		elif rot_fraction is False and pt_fraction is None:
			pt_fraction = fraction
			rot_fraction = 0
		elif pt_fraction is None and rot_fraction is None:
			pt_fraction = fraction
			rot_fraction = fraction
		else:
			if pt_fraction is True:
				pt_fraction = fraction
			elif pt_fraction is None or pt_fraction is False:
				pt_fraction = 0
			
			if rot_fraction is True:
				rot_fraction = fraction
			elif rot_fraction is None or rot_fraction is False:
				rot_fraction = 0
		
		pt = self.position.interpolate(other.position, pt_fraction)
		rot_kwargs = {}
		for key in ['spin','shortestpath']:
			if kwargs.has_key(key):
				rot_kwargs[key] = kwargs[key]
		rot = self.rotation.interpolate(other.rotation, rot_fraction, **rot_kwargs)
		return transform(pt,rot)
	
	def capped(self,distance=None,angle=None):
		"""Return a copy with translation and rotation capped (each optional)"""
		if distance is None:
			p = self.position
		else:
			p = self.position.capped(distance)
		if angle is None:
			r = self.rotation
		else:
			r = self.rotation.capped(angle)
		return transform(p,r,parent=self._parent,child=self._child,stamp=self.stamp,pose=self.pose)
	
	def transform_to(self,other):
		if isinstance(other,CanonicalTransform) and self.pose != other.pose:
			raise TypeError("Can't get transform between a %s and a %s" % (self._type_str,other._type_str))
		other = transform(other,pose=self.pose)
		if self.pose:
			if self._frame and other._frame and self._frame != other._frame:
				raise FrameError("Can't get transform between poses in frames %s and %s" % (self._frame,other._frame))
		else:
			raise NotImplementedError()
		self_inv = self.inverse()
		out = transform(other.matrix * self_inv.matrix)
		_init_after_tf_mul(other,self_inv,out,skip_pose_check=True,dont_set_pose=True)
		return out
	
	def transform_from(self,other):
		raise NotImplementedError()
	
	def tostring(self,**kwargs):
		if kwargs.get('numpy',kwargs.get('np',False)):
			return super(CanonicalTransform,self).tostring()
		
		multiline = kwargs.get('multiline',False)
		if multiline:
			default_brackets = None
		else:
			default_brackets = '[]'
		brackets = kwargs.pop('brackets', default_brackets)
		if brackets is True:
			brackets = default_brackets
		simple = kwargs.pop('simple', False)
		fmt = _get_rot_string_fmt(kwargs)
		
		keys = ['frame','name','stamp']
		print_key = _get_print_key(kwargs,keys)
		
		frame_on = print_key['frame']
		name_on = print_key['name']
		stamp_on = print_key['stamp']
		
		frame_str = ''
		if frame_on and (self._parent or self._child):
			frame_str = self.frame_str
		
		stamp_show_zero = kwargs.get('stamp_show_zero',not kwargs.get('stamp_hide_zero',False))
		if kwargs.has_key('stamp'):
			if kwargs['stamp'] == 'show_zero':
				stamp_show_zero = True
				del kwargs['stamp']
			if kwargs['stamp'] == 'hide_zero':
				stamp_show_zero = False
				del kwargs['stamp']			
		stamp_kwargs = _prep_stamp_subkwargs(kwargs, ['show_zero','hide_zero'])
		
		stamp_show = self.stamp is not None and (not self.stamp.is_zero() or stamp_show_zero)
		
		stamp_str = ''
		if stamp_on and stamp_show:
			stamp_str = 'stamp='
			stamp_str += self.stamp.tostring(**stamp_kwargs)
		
		pt_kwargs = _prep_pt_subkwargs(kwargs, None, simple=True, brackets = '()')
		rot_kwargs = _prep_rot_subkwargs(kwargs, None, simple=True, brackets = '()')
		
		if simple:
			s = ''
			if brackets:
				s += brackets[0]
			s += self.translation.tostring(**pt_kwargs) + ',' + self.rotation.tostring(**rot_kwargs)
			if brackets:
				s += brackets[1]
			return s
		elif fmt == 'm':
			s = ''
			if self.pose:
				if frame_on and self._frame:
					s += 'frame=' + self._frame
				if name_on and self._name:
					if s:
						s += ','
					s += 'name=' + self._name
			else:
				if frame_on and (self._parent or self._child):
					s += frame_str
			if stamp_on and stamp_show:
				if s:
					s += ','
				s += stamp_str
			if s:
				s += '\n'
			s += numpy.matrix.__str__(self)
			return s
		
		if multiline:
			main_str = self.translation.tostring(**pt_kwargs) + '\n' + self.rotation.tostring(**rot_kwargs)
		else:
			main_str = self.translation.tostring(**pt_kwargs) + ',' + self.rotation.tostring(**rot_kwargs)
		
		s = ''
		if brackets:
			s += brackets[0]
		if self.pose:
			if multiline and name_on and self._name:
				s += self._name
				s += '\n'
			if frame_on and self._frame:
				if multiline:
					s += 'frame: '
				s += self._frame
				if multiline:
					s += '\n'
				else:
					s += ':'
			if multiline and stamp_on and stamp_show:
				if self.stamp.is_zero() and not stamp_kwargs:
					s += 'stamp:0'
				else:
					s += self.stamp.tostring(**stamp_kwargs)
				s += '\n'
			s += main_str
			if not multiline and name_on and self._name:
				s += ','
				s += 'name=' + self._name
		else:
			if frame_on and (self._parent or self._child):
				s += frame_str
				if multiline:
					if stamp_on and stamp_show:
						s += '\n' #', '
						if self.stamp.is_zero() and not stamp_kwargs:
							s += 'stamp:0'
						else:
							s += self.stamp.tostring(**stamp_kwargs)
					s += '\n'
				else:
					s += ':'
			elif multiline and stamp_on and stamp_show:
				if self.stamp.is_zero() and not stamp_kwargs:
					s += 'stamp:0'
				else:
					s += self.stamp.tostring(**stamp_kwargs)
				s += '\n'
			s += main_str
		if not multiline and stamp_on and stamp_show:
			s += ','
			s += stamp_str
		if brackets:
			s += brackets[1]
		return s	
	
	def todict(self,**kwargs):
		d = {}
		
		outer = kwargs.get('outer',False)
		short_names = kwargs.get('short_names',False)
		name = kwargs.get('name',None)
		combine_fields = kwargs.get('combine_fields',False)
		
		ros_on = kwargs.get('ros',False)
		type_on = kwargs.get('type',True)
		frame_on = kwargs.get('frame',True)
		stamp_on = kwargs.get('stamp',True)
		
		if outer:
			if name:
				key = name
			elif self.pose:
				key = 'pose'
			else:
				if short_names and not ros_on:
					key = 'tf'
				else:
					key = 'transform'
			inner_kwargs = kwargs.copy()
			inner_kwargs['outer'] = False
			d[key] = self.todict(**inner_kwargs)
			return d
		
		if ros_on:
			header_on = kwargs.get('header',None)
			if header_on is None:
				header_on = (self._frame or self.stamp)
			
			pd = {'x': self.translation.x, 'y': self.translation.y, 'z': self.translation.z}
			if self.pose:
				d['position'] = pd
			else:
				d['translation'] = pd
			
			rd = {'x': self.rotation.x, 'y': self.rotation.y, 'z': self.rotation.z, 'w': self.rotation.w}
			if self.pose:
				d['orientation'] = rd
			else:
				d['rotation'] = rd
			
			if header_on:
				header_d = {'seq': 0}
				s = self.stamp
				if s is None:
					s = stamp(0)
				header_d['stamp'] = {'secs': s.ros.secs, 'nsecs': s.ros.nsecs}
				if self._frame:
					header_d['frame_id'] = self._frame
				else:
					header_d['frame_id'] = ''
				if self.pose:
					field_name = 'pose'
				else:
					field_name = 'transform'
				d = {'header': header_d, field_name: d}
			return d
				
		
		if type_on:
			if self.pose:
				d['is_pose'] = True
			elif short_names:
				d['is_tf'] = True
			else:
				d['is_transform'] = True
		
		if self.pose:
			if short_names:
				p_key = 'pos'
				r_key = 'ori'
			else:
				p_key = 'position'
				r_key = 'orientation'
		else:
			if short_names:
				p_key = 'trans'
				r_key = 'rot'
			else:
				p_key = 'translation'
				r_key = 'rotation'
		
		p_kwargs = _prep_pt_subkwargs(kwargs,None,frame=False,stamp=False,outer=True,name=p_key)
		r_kwargs = _prep_rot_subkwargs(kwargs,None,frame=False,stamp=False,outer=True,name=r_key)
		
		pd = self.translation.todict(**p_kwargs)
		rd = self.rotation.todict(**r_kwargs)
		
		d.update(pd)
		d.update(rd)
		
		if frame_on and self._frame:
			d['frame'] = self._frame
		
		if stamp_on and self.stamp is not None:
			if short_names:
				d['t'] = self.stamp.seconds
			else:
				d['stamp'] = self.stamp.seconds
		
		return d
		
	def __new__(cls,data,data2=None,**kwargs):
		super_kwargs = kwargs.copy()
		
		for key in ['header','frames',
				'frame','frame_id','parent','to_frame','frame_to',
				'name','child','child_frame_id','from_frame','frame_from',
				'stamp',
				'pose']:
			if super_kwargs.has_key(key):
				del super_kwargs[key]
		
		super_kwargs['dtype'] = float
		
		if data2 is not None:
			data = (data,data2)
		
		d = _get_canonical_tf_new(data)
		
		return super(CanonicalTransform,cls).__new__(cls,d,**super_kwargs)
	
	def __init__(self,data,data2=None,**kwargs):
		""" frame/parent=None,name/child=None,stamp=None,pose=False """
		
		self._stamp = None
		self._frame = None
		self._name = None
		
		self._setup_header(kwargs, (data,data2))
		
		self._pose = None
		if kwargs.has_key('pose'):
			self.pose = kwargs['pose']
		else:
			if data2 is not None:
				type_data = (data,data2)
			else:
				type_data = data
			data_type = _get_type(type_data)
			self.pose = data_type.startswith('ps')
		
		self._msg = None
	
	def _setup_header(self,kwargs,data_and_data2=None):
		has_header, header_frame, header_stamp = _extract_header_frame_and_stamp(kwargs)
		
		if kwargs.has_key('stamp'):
			if not isinstance(kwargs['stamp'],bool):
				self._stamp = stamp(kwargs['stamp'],copy=True)
			elif kwargs['stamp']:
				self._stamp = stamp('now')
		elif has_header:
			self._stamp = stamp(header_stamp,copy=True)
		elif data_and_data2 is not None:
			self._stamp = _extract_stamp(*data_and_data2)
		
		frames = None
		if kwargs.has_key('frames'):
			frames = kwargs['frames']
		elif re.search(r'<-|->',kwargs.get('frame') or ''):
			frames = kwargs['frame']

		if frames is not None:
				err = True
				try:
					if frames.find('<-') != -1:
						self._frame, self._name = frames.split('<-')
						err = False
					elif frames.find('->') != -1:
						self._name, self._frame = frames.split('->')
						err = False
				except:
					pass
				if err:
					raise ValueError('Frames %s cannot be parsed' % frames)
		else:
			for key in _PARENT_KEYS:
				if kwargs.has_key(key):
					self._frame = kwargs[key]
					break
			else:
				if has_header:
					self._frame = header_frame
				elif data_and_data2 is not None:
					self._frame = _extract_frame(*data_and_data2)
			
			for key in _CHILD_KEYS:
				if kwargs.has_key(key):
					self._name = kwargs[key]
					break
			else:
				if data_and_data2 is not None:
					self._name = _extract_name(data_and_data2[0])
	
	def __reduce__(self):
		val = list(super(CanonicalTransform,self).__reduce__())
		state = val[2]
		state = (state, (self._pose, self._frame, self._name, self._stamp))
		val[2] = state
		return tuple(val)
	
	def __setstate__(self,state):
		ndarray_state = state[0]
		super(CanonicalTransform,self).__setstate__(ndarray_state)
		state = state[1]
		
		setattr(self,'_msg',None)
		
		setattr(self,'_pose',state[0])
		setattr(self,'_frame',state[1])
		setattr(self,'_name',state[2])
		setattr(self,'_stamp',state[3])
		
		self._update_frame_fields()
		
	
	@property
	def _type_str(self):
		if self.pose:
			return 'pose'
		else:
			return 'transform'
	
	def __str__(self):
		return self.tostring()
	
	def __repr__(self):
		kwargs = ''
		if self.pose:
			if self._frame is not None:
				kwargs += ",frame='%s'" % self._frame
			if self._name is not None:
				kwargs += ",name='%s'" % self._name
		else:
			if self._name is not None:
				kwargs += ",from_frame='%s'" % self._name
			if self._frame is not None:
				kwargs += ",to_frame='%s'" % self._frame
		if self.stamp is not None:
			kwargs += ",stamp=%s" % repr(self.stamp)
		return self._type_str + '(' + str(tuple(self.translation.list)) + ',' + str(tuple(self.rotation.list)) + kwargs + ')'

	def __getitem__(self, *args, **kwargs):
		return numpy.matrix.__getitem__(self.matrix, *args, **kwargs)
	
	def __getattribute__(self,key):
		if key in _PARENT_KEYS:
			return self._frame
		elif key in _CHILD_KEYS:
			return self._name
		else:
			return super(CanonicalTransform,self).__getattribute__(key)
	
	def __setattr__(self,key,value):
		if key in _PARENT_KEYS:
			self._frame = value
		elif key in _CHILD_KEYS:
			self._name = value
		else:
			return object.__setattr__(self,key,value)

	def _init_after_copy(self,new_obj,**kwargs):
		pose = kwargs.pop('pose',None)
		setattr(new_obj,'_msg',None)
		setattr(new_obj,'_pose',None)
		for field in ['_frame', '_name', '_stamp']:
			setattr(new_obj,field,None)
			if hasattr(self,field):
				setattr(new_obj,field,copy.copy(getattr(self, field)))
		new_obj._setup_header(kwargs)
		if pose is not None:
			new_obj._pose = pose
		else:
			new_obj._pose = getattr(self,'_pose',False)
		new_obj._update_frame_fields()

	def copy(self,**kwargs):
		"""Copy this transform or pose.
		Keyword args are passed to the constructor."""
		
		new_trans = _get_kwarg(kwargs, _POS_KEYS)
		if new_trans is not None:
			p_type, p_data = _get_data(new_trans,default_4_to_quat=False,default_4_to_pt=True)
			if p_type.startswith('u'):
				raise UnknownInputError(type1=p_type,data1=p_data,input_type='point')
			new_trans = p_data
		
		new_rot = None
		if new_rot is not None:
			r_type, r_data = _get_data(new_rot,default_4_to_quat=True,default_4_to_pt=False)
			if r_type.startswith('u'):
				raise UnknownInputError(type1=r_type,data1=r_data)
			new_rot = r_data

		new_obj = numpy.matrix.copy(self)
		self._init_after_copy(new_obj,**kwargs)
		if new_trans is not None:
			new_obj.translation = new_trans
		if new_rot is not None:
			new_obj.rotation = new_rot
		return new_obj
	
	def __copy__(self):
		new_obj = numpy.matrix.__copy__(self)
		self._init_after_copy(new_obj)
		return new_obj
	
	def __mul__(self,other):
		return _canonical_mul(self, other)
	
	def __rmul__(self,other):
		return _canonical_mul(other, self)
	
	def _init_for_add(self,other):
		data_type = _get_type(other)
		if data_type.split('/')[0] != 'p':
			raise TypeError('Cannot add %s and %s' % (self._type_str,other))
		p = point(other)
		if p.frame  and p.frame != self._frame:
			raise FrameError('Cannot add translation in frame %s to %s in frame %s' % (p.frame,self._type_str,self.frame_str))
		return p

	def __add__(self,other):
		p = self._init_for_add(other)
		tf = self.copy()
		numpy.matrix.__iadd__(tf[0:3,3],p.vector3)
		return tf
	
	def __iadd__(self,other):
		p = self._init_for_add(other)
		numpy.matrix.__iadd__(self[0:3,3],p.vector3)
		return self
	
	def __sub__(self,other):
		p = self._init_for_add(other)
		tf = self.copy()
		numpy.matrix.__isub__(tf[0:3,3],p.vector3)
		return tf
	
	def __isub__(self,other):
		p = self._init_for_add(other)
		numpy.matrix.__isub__(self[0:3,3],p.vector3)
		return self
	
	def _bad_op(self,other):
		return NotImplemented

	__radd__ = _bad_op
	__rsub__ = _bad_op
	__imul__ = _bad_op
	__div__ = _bad_op
	__rdiv__ = _bad_op
	__idiv__ = _bad_op

def _now():
	return CanonicalStamp.now()
_now.__name__ = 'now'
if _ROS:
	_now.__doc__ = """Returns the current time.
	Uses rospy.Time.now() if it is initialized, otherwise datetime.datetime.now()"""
else:
	_now.__doc__ = """Returns the current time.
	Uses datetime.datetime.now()"""

def _zerostamp():
	"""Returns a zero time, which may indicate null or unknown."""
	return CanonicalStamp.zero()
_zerostamp.__name__ = 'zero'

def stamp(*args,**kwargs):
	"""Create a CanonicalStamp out of almost anything."""
	if not args:
		return CanonicalStamp(**kwargs)
	if args[0] is None:
		return None
	copy = kwargs.pop('copy',False)
	if isinstance(args[0],CanonicalStamp) and not copy:
		return args[0]
	return CanonicalStamp(*args,**kwargs)
setattr(stamp,'now',_now)
setattr(stamp,'zero',_zerostamp)

_CanonicalStampAttrs = ['set','datetime','time','seconds','nanoseconds','is_zero','tostring','copy']
if _ROS: _CanonicalStampAttrs.append('ros')

for attr in _CanonicalStampAttrs:
	setattr(stamp,attr,getattr(CanonicalStamp,attr))

def duration(*args,**kwargs):
	"""Create a CanonicalDuration out of almost anything."""
	if args and args[0] is None:
		return None
	copy = kwargs.pop('copy',False)
	if isinstance(args[0],CanonicalDuration) and not copy:
		return args[0]
	return CanonicalDuration(*args,**kwargs)

_CanonicalDurationAttrs = ['set','seconds','nanoseconds','timedelta','tostring','copy']
if _ROS: _CanonicalDurationAttrs.append('ros')

for attr in _CanonicalDurationAttrs:
	setattr(duration,attr,getattr(CanonicalDuration,attr))

def time(*args,**kwargs):
	"""Create a CanonicalStamp or CanonicalDuration based on the input data.
	
	If the data is already a CanonicalStamp or CanonicalDuration object, it will
	be returned unless the keyword argument 'copy' is set to True.
	
	If the data may be ambiguous, the keyword arguments default_duration or
	default_stamp may be set to True."""
	if not args:
		return CanonicalStamp()
	if args[0] is None:
		return None
	copy = kwargs.pop('copy',False)
	default_duration = kwargs.pop('default_duration',not kwargs.pop('default_stamp',True))
	ambiguous = isinstance(args[0],(float,long,int)) or len(args) > 1 or \
			(isinstance(args[0],collections.Sequence) and not isinstance(args[0],basestring)) or \
			(isinstance(args[0],basestring) and args[0] != 'now')
	if isinstance(args[0],(CanonicalStamp,CanonicalDuration)):
		if copy:
			return args[0].copy()
		else:
			return args[0]
	if (ambiguous and default_duration) or isinstance(args[0],pydatetime.timedelta) or \
			(_ROS and (isinstance(args[0],rospy.Duration) or _ismsginstance(args[0],std_msgs.msg.Duration))):
		return duration(*args,**kwargs)
	else:
		return stamp(*args,**kwargs)
setattr(time,'now',_now)

def transform(data,data2=None,**kwargs):
	"""Create a CanonicalTransform object out of almost anything.
	
	Unlike the pose() function, transform() will by default return a transform or
	pose depending on the input data. To force it to return a transform or pose,
	use the 'pose' keyword argument.
	
	If the data is already a CanonicalTransform object, it will be returned 
	unless the keyword argument 'copy' is set to True.
	
	If the data is a tfMessage or PoseArray, the output will be a list of
	transforms or poses, respectively.
	
	Keyword args:
	    The (parent) frame can be set with any of the following:
	    'frame','frame_id','parent','to_frame','frame_to'
	    
	    The child frame (for a transform) or name (for a pose) can be set with:
	    'name','child','child_frame_id','from_frame','frame_from'
	    
	    The stamp can be set with the keyword arg 'stamp'.
	    
	    The frame and the stamp can be set together with the keyword arg 'header',
	    by giving it a ROS std_msgs/Header, or a dict with the fields 'stamp'
	    and 'frame' or 'frame_id'."""
	if data is None and data2 is None:
		return None
	copy = kwargs.pop('copy',False)
	if data2 is None:
		if isinstance(data,CanonicalTransform) and not copy and data.pose == kwargs.get('pose',False):
			return data
		if isinstance(data,CanonicalPoint) or isinstance(data,CanonicalRotation):
			return data.as_tf()
	if _ROS and _ismsginstance(data,tf.msg.tfMessage):
		return [transform(tf_data,**kwargs) for tf_data in data.transforms]
	elif _ROS and _ismsginstance(data,gm.PoseArray):
		if not kwargs.has_key('pose'):
			kwargs['pose'] = True
		return [transform(gm.PoseStamped(header=data.header,pose=pose_data),**kwargs) for pose_data in data.poses]
	if not kwargs.has_key('pose') and _ROS and _ismsginstance(data,(gm.Pose,gm.PoseStamped,gm.PoseWithCovariance,gm.PoseWithCovarianceStamped)):
			kwargs['pose'] = True
	return CanonicalTransform(data,data2,**kwargs)

def pose(data,data2=None,**kwargs):
	"""Create a CanonicalTransform object out of almost anything.
	
	Unlike the transform() function, pose() will always return a pose unless the
	'pose' keyword argument is set to False.
	
	If the data is already a CanonicalTransform object, it will	be returned 
	unless the keyword argument 'copy' is set to True.
	
	If the data is a PoseArray or tfMessage, the output will be a list of
	poses or transforms, respectively.
	
	Keyword args:
	    The (parent) frame can be set with any of the following:
	    'frame','frame_id','parent','to_frame','frame_to'
	    
	    The name or child frame (for a transform) can be set with:
	    'name','child','child_frame_id','from_frame','frame_from'
	    
	    The stamp can be set with the keyword arg 'stamp'.
	    
	    The frame and the stamp can be set together with the keyword arg 'header',
	    by giving it a ROS std_msgs/Header, or a dict with the fields 'stamp'
	    and 'frame' or 'frame_id'."""
	if not kwargs.has_key('pose'):
		kwargs['pose'] = True
	return transform(data,data2,**kwargs)


_CanonicalTransformAttrs = [key for key in CanonicalTransform.__dict__.keys() if (key not in _PARENT_KEYS 
																				and key not in _CHILD_KEYS
																				and not key.startswith('_'))]
for attr in _CanonicalTransformAttrs:
	setattr(transform,attr,getattr(CanonicalTransform,attr))
	setattr(pose,attr,getattr(CanonicalTransform,attr))

setattr(transform,'parent',CanonicalTransform._frame_property)
setattr(transform,'from_frame',CanonicalTransform._frame_property)
setattr(transform,'frame',CanonicalTransform._frame_property)

setattr(transform,'child',CanonicalTransform._name_property)
setattr(transform,'to_frame',CanonicalTransform._name_property)
setattr(transform,'name',CanonicalTransform._name_property)

setattr(pose,'frame',CanonicalTransform._frame_property)
setattr(pose,'name',CanonicalTransform._name_property)

def point(*args,**kwargs):
	"""Create a CanonicalPoint object out of almost anything.
	
	If the data is already a CanonicalPoint object, it will	be returned 
	unless the keyword argument 'copy' is set to True.
	
	Keyword args:
	    The frame can be set with any of the following:
	    'frame','frame_id','parent'
	    
	    The stamp can be set with the keyword arg 'stamp'.
	    
	    The frame and the stamp can be set together with the keyword arg 'header',
	    by giving it a ROS std_msgs/Header, or a dict with the fields 'stamp'
	    and 'frame' or 'frame_id'."""
	if args and args[0] is None:
		return None
	copy = kwargs.pop('copy',False)
	if len(args) == 1 and isinstance(args[0],CanonicalPoint) and not copy:
		return args[0]
	else:
		return CanonicalPoint(*args,**kwargs)

vector = point

_CanonicalPointAttrs = [key for key in CanonicalPoint.__dict__.keys() if not key.startswith('_')]

for attr in _CanonicalPointAttrs:
	setattr(point,attr,getattr(CanonicalPoint,attr))

def rotation(*args,**kwargs):
	"""Create a CanonicalRotation object out of almost anything.
	
	If the data is already a CanonicalRotation object, it will	be returned 
	unless the keyword argument 'copy' is set to True.
	
	Keyword args:
	    The frame can be set with any of the following:
	    'frame','frame_id','parent'
	    
	    The stamp can be set with the keyword arg 'stamp'.
	    
	    The frame and the stamp can be set together with the keyword arg 'header',
	    by giving it a ROS std_msgs/Header, or a dict with the fields 'stamp'
	    and 'frame' or 'frame_id'."""
	if args and args[0] is None:
		return None
	copy = kwargs.pop('copy',False)
	if len(args) == 1 and isinstance(args[0],CanonicalRotation) and not copy:
		return args[0]
	else:
		return CanonicalRotation(*args,**kwargs)
	
quaternion = rotation

_CanonicalRotationAttrs = [key for key in CanonicalRotation.__dict__.keys() if not key.startswith('_')]

for attr in _CanonicalRotationAttrs:
	setattr(rotation,attr,getattr(CanonicalRotation,attr))

def rotation_tb(*args,**kwargs):
	"""Create a CanonicalRotation object from Tait-Bryan angles.
	
	See the tb_angles documentation for a description of the allowed input."""
	tb_kwargs = kwargs
	rot_kwargs = kwargs.copy()
	for key in ['deg','rad','yaw','y','pitch','p','roll','r']:
		rot_kwargs.pop(key,None)
	return CanonicalRotation(tb_angles(*args,**tb_kwargs),**rot_kwargs)

def rotation_euler(ai,aj,ak,axes,**kwargs):
	"""Create a CanonicalRotation object from Euler angles.
	
	The axes must be given as (r|s)<axes>, where 'r' indicates rotating axes
	and 's' indicates stationary axes. <axes> must be the axes to rotate about.
	For example, stationary xyz would be given as sxyz."""
	if isinstance(ai,basestring):
		ax = ai
		ai = aj
		aj = ak
		ak = axes
		axes = ax
	return CanonicalRotation(tft.quaternion_from_euler(ai, aj, ak, axes),**kwargs)

def inverse_tf(data,data2=None,**kwargs):
	"""Create a CanonicalTransform object from the input and return the inverse."""
	return transform(data,data2,**kwargs).inverse()

def identity_tf(**kwargs):
	"""Create a CanonicalTransform object with identity value, plus any 
	keyword arguments given."""
	return CanonicalTransform.identity(**kwargs)
def identity_rotation(**kwargs):
	"""Create a CanonicalRotation object with identity value, plus any 
	keyword arguments given."""
	return CanonicalRotation.identity(**kwargs)
	
def random_tf(scale=1,mean=0,dist='uniform',**kwargs):
	return CanonicalTransform.random(scale=scale,mean=mean,dist=dist,**kwargs)

def random_pose(scale=1,mean=0,dist='uniform',**kwargs):
	if not kwargs.has_key('pose'):
		kwargs['pose'] = True
	return CanonicalTransform.random(scale=scale,mean=mean,dist=dist,**kwargs)

def random_point(scale=1,mean=0,dist='uniform',**kwargs):
	return CanonicalPoint.random(scale=scale,mean=mean,dist=dist,**kwargs)
random_vector = random_point

def random_translation_tf(scale=1,mean=0,dist='uniform',**kwargs):
	return CanonicalTransform.random_translation(scale=scale,mean=mean,dist=dist,**kwargs)
def random_translation_pose(scale=1,mean=0,dist='uniform',**kwargs):
	if not kwargs.has_key('pose'):
		kwargs['pose'] = True
	return CanonicalTransform.random_translation(scale=scale,mean=mean,dist=dist,**kwargs)

def random_rotation(**kwargs):
	return CanonicalRotation.random(**kwargs)
random_quaternion = random_rotation
	
def random_rotation_tf(**kwargs):
	return CanonicalTransform.random_rotation(**kwargs)
def random_rotation_pose(**kwargs):
	if not kwargs.has_key('pose'):
		kwargs['pose'] = True
	return CanonicalTransform.random_rotation(**kwargs)

def _get_type(value,default_4_to_quat=False,default_4_to_pt=False):
	#base types: tf, ps, p, q, r, u, s
	#sub types: c, d, v#, a#, l#, t, msg[/h]
	
	def _is_rot_type(typ):
		base_type = typ.split('/')[0]
		return base_type == 'q' or base_type == 'r'
	
	def _is_pt_type(typ):
		base_type = typ.split('/')[0]
		return base_type == 'p'

	def _is_unknown_type(typ):
		base_type = typ.split('/')[0]
		return base_type == 'u'
	
	if value is None:
		return None
	if isinstance(value,numbers.Number):
		return 's'
	elif isinstance(value,CanonicalTransform):
		if value.pose:
			return 'ps/c'
		else:
			return 'tf/c'
	elif isinstance(value,CanonicalPoint):
		return 'p/c'
	elif isinstance(value,CanonicalRotation):
		return 'r/c'
	elif isinstance(value,tb_angles):
		return 'r/tb/c'
	elif isinstance(value,dict):
		return _get_dict_data_type(value)
	elif isinstance(value,numpy.ndarray):
		if value.shape == (4,4):
			return 'tf/a44'
		elif value.shape == (3,3):
			return 'r/a33'
		elif value.size == 3:
			return 'p/v3'
		elif value.size == 4:
			if value.shape == (4,1):
				return 'p/v4'
			elif value.shape == (4,) or value.shape == (1,4):
				if value.flat[3] != 1:
					return 'q/l4'
				elif abs(numpy.linalg.norm(value) - 1) > 1e-6:
					return 'p/l4' 
				else:
					if default_4_to_quat:
						return 'q/a4'
					elif default_4_to_pt:
						return 'p/a4'
					else:
						return 'u/a4'
			else:
				return 'u/a' + str(value.shape[0]) + str(value.shape[1])
		else:
			if value.ndim == 1:
				return 'u/a' + str(len(value))
			else:
				return 'u/a' + str(value.shape[0]) + str(value.shape[1])
	elif isinstance(value,collections.Sequence):
		if len(value) == 16:
			return 'tf/l16'
		elif len(value) == 9:
			return 'r/l9'
		elif len(value) == 3:
			if all([isinstance(val,collections.Sequence) and len(val) == 3 for val in value]):
				return 'r/l33'
			elif all([isinstance(val,collections.Sequence) and len(val) == 1 for val in value]):
				return 'p/l31'
			elif all([isinstance(val,numbers.Number) for val in value]):
				return 'p/l3'
			else:
				return 'u/l3'
		elif len(value) == 4:
			if all([isinstance(val,collections.Sequence) and len(val) == 4 for val in value]):
				return 'tf/l44'
			elif all([isinstance(val,collections.Sequence) and len(val) == 1 for val in value]):
				return 'p/l41'
			else:
				if isinstance(value,QuaternionTuple):
					return 'q/l4'
				elif value[3] != 1:
					return 'q/l4'
				elif abs(numpy.linalg.norm(value) - 1) > 1e-6:
					return 'p/l4' 
				else:
					if default_4_to_quat:
						return 'q/l4'
					elif default_4_to_pt:
						return 'p/l4'
					else:
						return 'u/l4'
		elif len(value) == 1:
			if isinstance(value[0],collections.Sequence):
				if len(value[0]) == 3:
					return 'p/l13'
				elif len(value[0]) == 4 and all([isinstance(val,numbers.Number) for val in value[0]]):
					if value[0][3] != 1:
						return 'q/l14'
					elif abs(numpy.linalg.norm(value[0]) - 1) > 1e-6:
						return 'p/l14' 
					else:
						if default_4_to_quat:
							return 'q/l14'
						elif default_4_to_pt:
							return 'p/l14'
						else:
							return 'u/l14'
				else:
					return 'u/l1'
			else:
				return 'u/l1'
		elif len(value) == 2:
			val1 = value[0]
			val2 = value[1]
			if isinstance(val1,numbers.Number) and isinstance(val2,collections.Sequence) and len(val2) == 3:
				return 'r/t/ax'
			elif isinstance(val2,numbers.Number) and isinstance(val1,collections.Sequence) and len(val1) == 3:
				return 'r/t/xa'
			elif isinstance(val1,numbers.Number) and isinstance(val2,numpy.ndarray) and val2.size == 3:
				return 'r/t/ax'
			elif isinstance(val2,numbers.Number) and isinstance(val1,numpy.ndarray) and val1.size == 3:
				return 'r/t/xa'
			else:
				type1 = _get_type(value[0])
				type2 = _get_type(value[1])
				if  _is_pt_type(type1) and _is_unknown_type(type2):
					type2 = _get_type(value[1],default_4_to_quat=True)
				elif _is_pt_type(type2) and _is_unknown_type(type1):
					type1 = _get_type(value[0],default_4_to_quat=True)
				if (_is_rot_type(type1) and _is_pt_type(type2)) or (_is_pt_type(type1) and _is_rot_type(type2)):
					return 'tf/t/[' + type1 + ',' + type2 + ']'
				else:
					return 'u/l2'
		else:
			return 'u/l' + str(len(value))
	elif _ROS and _ismsginstance(value,gm.Quaternion):
		return 'q/msg'
	elif _ROS and _ismsginstance(value,gm.QuaternionStamped):
		return 'q/msg/h'
	elif _ROS and _ismsginstance(value,gm.Pose):
		return 'ps/msg'
	elif _ROS and _ismsginstance(value,gm.PoseStamped):
		return 'ps/msg/h'
	elif _ROS and _ismsginstance(value,gm.PoseWithCovariance):
		return 'ps/msg/cov'
	elif _ROS and _ismsginstance(value,gm.PoseWithCovarianceStamped):
		return 'ps/msg/cov/h'
	elif _ROS and _ismsginstance(value,gm.Transform):
		return 'tf/msg/tf'
	elif _ROS and _ismsginstance(value,gm.TransformStamped):
		return 'tf/msg/tf/h'
	elif _ROS and _ismsginstance(value,(gm.Point,gm.Point32)):
		return 'p/msg/pt'
	elif _ROS and _ismsginstance(value,gm.PointStamped):
		return 'p/msg/pt/h'
	elif _ROS and _ismsginstance(value,gm.Vector3):
		return 'p/msg/v'
	elif _ROS and _ismsginstance(value,gm.Vector3Stamped):
		return 'p/msg/v/h'
	else:
		return 'u'

def _get_dict_data_type(value):
	if all([value.has_key(key) for key in 'xyz']):
		if value.has_key('w'):
			return 'q/d/x'
		else:
			return 'p/d'
	elif all([value.has_key(key) for key in ['qx','qy','qz','qw']]):
		return 'q/d/qx'
	elif all([value.has_key(key) for key in ['yaw','pitch','roll']]):
		return 'r/tb/d'
	elif any([value.has_key(key) for key in (_POS_KEYS + _ROT_AND_TB_KEYS)]):
		is_pose = None
		if value.has_key('is_pose'):
			is_pose = value['is_pose']
		elif value.has_key('is_transform'):
			is_pose = not value['is_transform']
		elif value.has_key('is_tf'):
			is_pose = not value['is_tf']
		if is_pose is not None:
			if is_pose:
				return 'ps/d'
			else:
				return 'tf/d'
		else:
			return 'tf/d'
	elif value.has_key('pose'):
		return 'ps/d/o'
	elif any([value.has_key(key) for key in ['transform','tf']]):
		return 'tf/d/o'
	return 'u/d'

def _get_data(data,default_4_to_quat=False,default_4_to_pt=False):
	data_type = _get_type(data,default_4_to_quat=default_4_to_quat,default_4_to_pt=default_4_to_pt)
	
	base_type = data_type.split('/')[0]
	
	if base_type in ['s','u']:
		return (data_type,data)
	
	sub_types = data_type.split('/')[1:]
	
	if base_type in ['tf','ps']:
		return (base_type,_get_tf_data(data,sub_types,is_pose=base_type=='ps',default_4_to_quat=default_4_to_quat,default_4_to_pt=default_4_to_pt))
	elif base_type == 'p':
		return (base_type,_get_pt_data(data,sub_types))
	elif base_type == 'q':
		return (base_type,_get_quat_data(data,sub_types))
	elif base_type == 'r':
		return (base_type,_get_rot_data(data,sub_types))

def _get_tf_data(data,sub_types,is_pose,default_4_to_quat=False,default_4_to_pt=False):
	if sub_types[0] == 'c':
		return numpy.asarray(data)[0:4,0:4]
	elif sub_types[0] == 't':
		vals = {}
		for v in data:
			vals.__setitem__(*_get_data(v,default_4_to_quat=default_4_to_quat,default_4_to_pt=default_4_to_pt))
		return vals
	elif sub_types[0] == 'd':
		for key in ['pose','transform','tf']:
			if data.has_key(key):
				return _get_tf_data(data[key],sub_types,is_pose,default_4_to_quat=default_4_to_quat,default_4_to_pt=default_4_to_pt)
		vals = {}
		for key in _POS_KEYS:
			if data.has_key(key):
				p_type, p_data = _get_data(data[key],default_4_to_quat=False,default_4_to_pt=True)
				if p_type.startswith('tf') or p_type.startswith('ps'):
					p_type, p_data = _get_data(p_data['p'])
				vals[p_type] = p_data
				break
		for key in _ROT_KEYS:
			if data.has_key(key):
				r_type, r_data = _get_data(data[key],default_4_to_quat=True,default_4_to_pt=False)
				if r_type.startswith('tf') or r_type.startswith('ps'):
					r_type, r_data = _get_data(r_data['r'])
				vals[r_type] = r_data
				break
		else:
			for key in _TB_KEYS:
				if data.has_key(key):
					data = tb_angles(data[key]).quaternion
				r_type, r_data = _get_data(data[key],default_4_to_quat=True,default_4_to_pt=False)
				if r_type.startswith('tf') or r_type.startswith('ps'):
					r_type, r_data = _get_data(r_data['r'])
				vals[r_type] = r_data
				break
		return vals
	elif sub_types[0] in ['a44','l44']:
		return numpy.asarray(data)
	elif sub_types[0] == 'l16':
		return numpy.asarray(data).reshape((4,4))
	elif sub_types[0] == 'msg':
		vals = {}
		if is_pose:
			if len(sub_types) > 1 and sub_types[1] == 'cov':
				data = data.pose
				del sub_types[1]
			if sub_types[-1] == 'h':
				data = data.pose
				del sub_types[-1]
			vals['p'] = _get_pt_data(data.position,sub_types)
			vals['q'] = _get_quat_data(data.orientation,sub_types)
		else:
			if sub_types[-1] == 'h':
				data = data.transform
				del sub_types[-1]
			vals['p'] = _get_pt_data(data.translation,sub_types)
			vals['q'] = _get_quat_data(data.rotation,sub_types)
		return vals

def _get_pt_data(data,sub_types):
	if sub_types[0] == 'c':
		return numpy.asarray(data).flat[0:3].reshape((3,1))
	elif any([sub_types[0].startswith(t) for t in ['a','v','l']]):
		return numpy.asarray(data).flat[0:3].reshape((3,1))
	elif sub_types[0].startswith('d'):
		return numpy.array([data.get('x',0.),data.get('y',0.),data.get('z',0.)]).reshape((3,1))
	elif sub_types[0] == 'msg':
		if sub_types[-1] == 'h':
			if sub_types[1] == 'pt':
				data = data.point
			elif sub_types[1] == 'v':
				data = data.vector
		return numpy.array([data.x,data.y,data.z]).reshape((3,1))

def _get_quat_data(data,sub_types):
	if sub_types[0] == 'c':
		return numpy.asarray(data).flat[0:4]
	if any([sub_types[0].startswith(t) for t in ['a','v','l']]):
		return numpy.asarray(data).flat[0:4]
	elif sub_types[0].startswith('d'):
		if sub_types[1] == 'x':
			return numpy.array([data['x'],data['y'],data['z'],data['w']])
		elif sub_types[1] == 'qx':
			return numpy.array([data['qx'],data['qy'],data['qz'],data['qw']])
	elif sub_types[0] == 'msg':
		if sub_types[-1] == 'h':
			data = data.quaternion
		return numpy.array([data.x,data.y,data.z,data.w])

def _get_rot_data(data,sub_types):
	if sub_types[0] == 'c':
		return numpy.asarray(data)
	if sub_types[0] == 'tb':
		return numpy.asarray(get_tb_angles(data).matrix)
	elif sub_types[0] in ['a33','l33']:
		return numpy.asarray(data)
	elif sub_types[0] == 'l9':
		return numpy.asarray(data).reshape((3,3))
	elif sub_types[0] == 't':
		if sub_types[1] in ['ax','xa']:
			axis, angle = _get_axis_angle(data[0], data[1])
			return tft.quaternion_matrix(tft.quaternion_about_axis(angle, axis))[0:3,0:3]

def _init_after_quat_mul(q1,q2,out,**kwargs):
	if q1.frame and q2.frame and q1.frame != q2.frame:
		raise FrameError("Trying to combine rotations from frames %s and %s" % (q1.frame,q2.frame))
	
	if kwargs.has_key('new_frame'):
		out.frame = kwargs['new_frame']
	else:
		if q1.frame:
			out.frame = q1.frame
		elif q2.frame:
			out.frame = q2.frame
		else:
			out.frame = None
	
	stamp_to_set = None
	if kwargs.has_key('stamp'):
		stamp_to_set = stamp(kwargs['stamp'])
	if kwargs.has_key('new_stamp'):
		stamp_to_set = stamp(kwargs['new_stamp'])
	elif q1.stamp:
		if not q2.stamp:
			stamp_to_set = q1.stamp
		elif q1.stamp > q2.stamp:
			stamp_to_set = q1.stamp
		else:
			stamp_to_set = q2.stamp
	elif q2.stamp:
		stamp_to_set = q2.stamp
	else:
		stamp_to_set = None
	out.stamp = copy.copy(stamp_to_set)
	
	return out

def _init_after_tf_pt_mul(tf,pt,newpt,**kwargs):
	if tf.pose:
		raise TypeError("Can't apply a pose to a vector")
	if kwargs.has_key('new_frame'):
		newpt.frame = kwargs['new_frame']
	else:
		if tf._child and pt.frame:
			if tf._child != pt.frame:
				if tf._child:
					raise FrameError('transform',tf.frame_str,'point',pt.frame)
				else:
					raise FrameError('transform',tf._child,'point',pt.frame)
			else:
				newpt.frame = tf._parent
		elif tf._child:
			#pt has no frame
			if tf._parent:
				newpt.frame = tf._parent
			else:
				pass #newpt.frame = tf._child
		elif tf._parent:
			#pt has frame, tf has no child frame
			#assume the transform child is the pt frame
			newpt.frame = tf._parent
		else:
			newpt.frame = pt.frame
	
	stamp_to_set = None
	if kwargs.has_key('stamp'):
		stamp_to_set = stamp(kwargs['stamp'])
	if kwargs.has_key('new_stamp'):
		stamp_to_set = stamp(kwargs['new_stamp'])
	elif tf.stamp:
		if not pt.stamp:
			stamp_to_set = tf.stamp
		elif tf.stamp > pt.stamp:
			stamp_to_set = tf.stamp
		else:
			stamp_to_set = pt.stamp
	elif pt.stamp:
		stamp_to_set = pt.stamp
	else:
		stamp_to_set = None
	newpt.stamp = copy.copy(stamp_to_set)
	
	return newpt
		

def _init_after_tf_mul(tf1,tf2,newtf,skip_pose_check=False,skip_frame_check=False,dont_set_pose=False,**kwargs):
	if not skip_pose_check and tf1.pose:
		raise TypeError("Can't left-multiply a pose!")
	if not skip_frame_check and tf1._child and tf2._parent and tf1._child != tf2._parent:
		raise FrameError('transform',tf1._frame_str,tf2._type_str,tf2._frame)
	
	frames = None
	if kwargs.has_key('new_frames'):
		frames = kwargs['new_frames']
	elif kwargs.has_key('frames'):
		frames = kwargs['frames']
	elif re.search(r'<-|->',kwargs.get('new_frame') or ''):
		frames = kwargs.pop('new_frame')
	elif re.search(r'<-|->',kwargs.get('frame') or ''):
		frames = kwargs.pop('frame')

	if frames:
		err = True
		try:
			if frames.find('<-') != -1:
				newtf._parent, newtf._child = frames.split('<-')
				err = False
			elif frames.find('->') != -1:
				newtf._name, newtf._frame = frames.split('->')
				err = False
		except:
			pass
		if err:
			raise ValueError('Frames %s cannot be parsed' % frames)
	else:
		if kwargs.get('keep_frames',False) and not tf1._parent and not tf1._child:
			newtf._parent = tf2._parent
			newtf._child = tf2._child
		elif kwargs.get('keep_frames',False) and not tf2._parent and not tf2._child and not tf2.pose:
			newtf._parent = tf1._parent
			newtf._child = tf1._child
		elif not tf2._parent and not tf2._child and tf2.pose and not dont_set_pose:
			newtf._parent = tf1._parent
			newtf._child = None
		else:
			for key in _PARENT_KEYS:
				if kwargs.has_key(key):
					newtf._parent = kwargs[key]
					break
				elif kwargs.has_key('new_' + key):
					newtf._parent = kwargs['new_' + key]
					break
			else:
				newtf._parent = tf1._parent
			
			for key in _CHILD_KEYS:
				if kwargs.has_key(key):
					newtf._child = kwargs[key]
					break
				elif kwargs.has_key('new_' + key):
					newtf._child = kwargs['new_' + key]
					break
			else:
				newtf._child = tf2._child
	if not dont_set_pose:
		newtf.pose = tf2.pose
	
	stamp_to_set = None
	if kwargs.has_key('stamp'):
		stamp_to_set = stamp(kwargs['stamp'])
	if kwargs.has_key('new_stamp'):
		stamp_to_set = stamp(kwargs['new_stamp'])
	elif tf1.stamp:
		if not tf2.stamp:
			stamp_to_set = tf1.stamp
		elif tf1.stamp > tf2.stamp:
			stamp_to_set = tf1.stamp
		else:
			stamp_to_set = tf2.stamp
	elif tf2.stamp:
		stamp_to_set = tf2.stamp
	else:
		stamp_to_set = None
	newtf.stamp = copy.copy(stamp_to_set)
		
	return newtf

def _canonical_mul(value1,value2,**kwargs):
	values = [value1,value2]
	types = [None,None]
	for i,value in enumerate(values):
		types[i] = _get_type(value)
	
	split_types = [t.split('/') for t in types]
	base_types = [t.split('/')[0] for t in types]
	
	if 's' in base_types and not 'p' in base_types:
		if 'tf' in base_types:
			raise TypeError("Can't multiply a transform by a scalar!")
		elif 'ps' in base_types:
			raise TypeError("Can't multiply a pose by a scalar!")
		elif 'q' in base_types or 'r' in base_types:
			raise TypeError("Can't multiply a rotation by a scalar!")
		elif base_types[0] == 's':
			raise TypeError("Can't multiply unknown data %s by a scalar!" % str(values[1]))
		elif base_types[1] == 's':
			raise TypeError("Can't multiply unknown data %s by a scalar!" % str(values[0]))
	elif base_types[0] == 's':
		scalar = values[0]
		pt = point(values[1])
		return CanonicalPoint(numpy.matrix.__rmul__(pt._mat_view,scalar),frame=pt._frame,stamp=pt.stamp)
	
	if base_types[0] == base_types[1] and base_types[0] == 'p':
		pt1 = numpy.asarray(values[0]).reshape(3,)
		pt2 = numpy.asarray(values[1]).reshape(3,)
		if isinstance(values[0],CanonicalPoint):
			if isinstance(values[1],CanonicalPoint):
				raise TypeError("Can't multiply two points!")
			else:
				return CanonicalPoint(numpy.ndarray.__mul__(pt1,pt2),frame=values[0]._frame,stamp=values[0].stamp)
		elif isinstance(values[1],CanonicalPoint):
			return CanonicalPoint(numpy.ndarray.__mul__(pt1,pt2),frame=values[1]._frame,stamp=values[1].stamp)
		else:
			return CanonicalPoint(numpy.ndarray.__mul__(pt1,pt2))
	
	if base_types[0] == 'p':
		if base_types[1] == 's':
			pt = point(values[0])
			scalar = values[1]
			return CanonicalPoint(numpy.matrix.__mul__(pt._mat_view,scalar),frame=pt._frame,stamp=pt.stamp)
		else:
			raise TypeError("Can't left-multiply a point!")
	
	if base_types[0] == 'ps':
		raise TypeError("Can't left-multiply a pose! Use .as_transform() if needed.")
	
	#All error checking done. At this point, we should only have the following cases:
	# 1. rot    * rot  =  rot
	# 2. tf/rot *  pt  =   pt
	# 3. tf/rot *   u  =   rot or pt, depending
	# 4. tf/rot *  tf  =   tf
	# 4. tf/rot *  ps  =   ps
	
	matrix_mul = lambda A,B: numpy.mat(numpy.matrix.__mul__(A, B))
	
	if base_types[0] == base_types[1] and base_types[0] != 'tf':
		if base_types[0] in ['q','r']:
			r1 = rotation(values[0])
			r2 = rotation(values[1])
			rnew = CanonicalRotation(matrix_mul(r1.matrix, r2.matrix))
			return _init_after_quat_mul(r1,r2,rnew)
		else:
			raise UnknownInputError(type1=types[0],type2=types[1])
	elif base_types[1] == 'p':
		l_value = transform(values[0])
		r_value = point(values[1])
		if l_value._parent == r_value._frame and r_value._frame:
			l_value = l_value.inverse()
		newpt = CanonicalPoint(matrix_mul(l_value.matrix, r_value.vector4))
		return _init_after_tf_pt_mul(l_value, r_value , newpt,**kwargs)
	elif base_types[1] == 'u' and re.match(r'^[lav]1?4$',split_types[1][1]):
		if isinstance(values[1],numpy.matrix):
			l_value = transform(values[0])
			r_value = point(values[1])
			if l_value._parent == r_value._frame and r_value._frame:
				l_value = l_value.inverse()
			newpt = CanonicalPoint(matrix_mul(l_value, r_value._mat_view))
			return _init_after_tf_pt_mul(l_value, r_value , newpt,**kwargs)
		else:
			if not kwargs.has_key('default_4_to_quat'):
				sys.stderr.write('WARNING: multiplication is interpreting 4 element array as quaternion\n')
			if kwargs.get('default_4_to_quat',True):
				l_value = transform(values[0])
				r_value = rotation(values[1]).as_tf()
				if l_value._parent == r_value._frame and r_value._frame:
					l_value = l_value.inverse()
				newtf = transform(matrix_mul(l_value.matrix, r_value.matrix))
				return _init_after_tf_mul(l_value, r_value, newtf,**kwargs)
			else:
				l_value = transform(values[0])
				r_value = point(values[1])
				if l_value._parent == r_value._frame and r_value._frame:
					l_value = l_value.inverse()
				newpt = CanonicalPoint(matrix_mul(l_value, r_value._mat_view))
				return _init_after_tf_pt_mul(l_value, r_value , newpt,**kwargs)
	else:
		l_value = transform(values[0])
		r_value = transform(values[1]) #will be a pose if data is pose
		if (base_types[1] != 'tf' or r_value.pose) \
				and l_value._parent == r_value._frame and r_value._frame:
			l_value = l_value.inverse()
		newtf = transform(matrix_mul(l_value.matrix, r_value.matrix))
		value = _init_after_tf_mul(l_value, r_value, newtf,**kwargs)
		if base_types[1] in ['q','r']: #if r_value is a rotation, output rotation
			value = value.rotation
		return value

def _get_canonical_tf_new(data,default_4_to_quat=True):
	T = numpy.identity(4, dtype=float)
	data_type, data = _get_data(data, default_4_to_quat=default_4_to_quat)
	if data_type in ['tf','ps']:
		if isinstance(data,dict):
			if data.has_key('p'):
				T[0:3,3:4] = data['p']
			if data.has_key('r'):
				T[0:3,0:3] = data['r']
			elif data.has_key('q'):
				T[0:3,0:3] = tft.quaternion_matrix(data['q'])[0:3,0:3]
		else:
			T = data
	elif data_type == 'p':
		T[0:3,3:4] = data
	elif data_type == 'r':
		T[0:3,0:3] = data
	elif data_type == 'q':
		T = tft.quaternion_matrix(data)
	return T
