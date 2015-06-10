
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
import readline
import tf
import tf.transformations as tft
from geometry_msgs.msg import *
from numpy import *
from math import *
from collections import namedtuple

import sys
import re
from optparse import OptionParser, OptionGroup
import numbers

from tb_angles import *
from tf_io import *

__version__ = '0.1'

def _get_tf_translation_tuple(tf):
	return tf.get_translation_tuple()

def _get_tf_quaternion_tuple(tf):
	return tf.get_quaternion_tuple()

class _Getch:
	"""Gets a single character from standard input.  Does not echo to the screen."""
	def __init__(self):
		try:
			self.impl = _GetchWindows()
		except ImportError:
			self.impl = _GetchUnix()

	def __call__(self): return self.impl()


class _GetchUnix:
	def __init__(self):
		import tty, sys

	def __call__(self):
		import sys, tty, termios
		fd = sys.stdin.fileno()
		old_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(sys.stdin.fileno())
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
		return ch


class _GetchWindows:
	def __init__(self):
		import msvcrt

	def __call__(self):
		import msvcrt
		return msvcrt.getch()
		
def _indent(text,base=None,initial=None,subsequent=None,width=80,strip=True,final_newline=None):
	if not base: 
		base = ''
	elif isinstance(base,numbers.Number):
		base = ' ' * base
	if not initial:
		initial = ''
	elif isinstance(initial,numbers.Number):
		initial = ' ' * initial
	if not subsequent:
		subsequent = ''
	elif isinstance(subsequent,numbers.Number):
		subsequent = ' ' * subsequent

	if final_newline is None:
		m = re.search(r'\n*$',text)
		final_newline = m.group()
		text = re.sub(r'\n*$','',text)
	elif isinstance(final_newline,bool):
		if final_newline:
			final_newline = '\n'
		else:
			final_newline = ''

	indented = ''

	m = re.search(r'^\n*',text)
	indented += m.group()
	text = re.sub(r'^\n*','',text)
		
	lines = text.splitlines()
	for line in lines:
		if strip:
			line = line.strip()
		first = True
		while True:
			if first:
				indent = base+initial
				first = False
			else:
				indent = base+subsequent
			lwidth = width-len(indent)
			if len(line) < lwidth:
				indented += indent + line + '\n'
				break
			ind = line.rfind(' ',0,lwidth)
			if ind == -1:
				indented += indent + line[0:lwidth] + '\n'
				del line[0:lwidth]
				continue
			indented += indent + line[0:ind].strip() + '\n'
			line = line[ind:].strip()
	indented = re.sub(r'\n*$','',indented) +  final_newline
	return indented
		

_param_name = lambda name: rospy.resolve_name('~' + name.replace('-','_'))

_has_param = lambda name: rospy.has_param(_param_name(name))
		
_str_param = lambda name, default: rospy.get_param(_param_name(name),default)

_str2bool = lambda val: str(val).lower() in ("true", "1")

_bool_param = lambda name, default: _str2bool(_str_param(name,default))

_prep_str = lambda val: re.sub(r' +',' ',re.sub(r'[\[\]\(\),]',' ',str(val))).strip()

KEYS = {
	'help': 'H',
	'x+':     'W', 'x-':     'S',
	'y+':     'A', 'y-':     'D',
	'z+':     'Q', 'z-':     'Z',
	'yaw+':   'Y', 'yaw-':   'U',
	'pitch+': 'P', 'pitch-': '0',
	'roll+':  'R', 'roll-':  'E',
	'print_params': 'V',
	'change_increment_pos':   'I',
	'change_increment_angle': 'K',
	'set': '=',
	'invert_increment': '-',
	'invert_transform': '/'}

def key_mapping_str():
	return \
			'PRINT THIS INFO: %s\n' % KEYS['help'] + \
			'axis    +  %s \t   -    \n' % ' ' + \
			'x     fwd  %s \t back  %s\n' % (KEYS['x+'],KEYS['x-']) + \
			'y     left %s \t right %s\n' % (KEYS['y+'],KEYS['y-']) + \
			'z     up   %s \t down  %s\n' % (KEYS['z+'],KEYS['z-']) + \
			'yaw   left %s \t right %s\n' % (KEYS['yaw+'],KEYS['yaw-']) + \
			'pitch down %s \t up    %s\n' % (KEYS['pitch+'],KEYS['pitch-']) + \
			'\n' + \
			'axis    -  %s  \t   +    \n' % ' ' + \
			'roll  left %s \t right %s\n' % (KEYS['roll-'],KEYS['roll+']) + \
			'\n' + \
			'view current params:            %s\n' % KEYS['print_params'] + \
			'view/change position increment: %s\n' % KEYS['change_increment_pos'] + \
			'view/change angle increment:    %s\n' % KEYS['change_increment_angle'] + \
			'set positions or angles:        %s\n' % KEYS['set'] + \
			'\n' + \
			'Flip transform increments:      %s\n' % KEYS['invert_increment'] + \
			'Flip transform:                 %s' % KEYS['invert_transform']

def _print_key_mapping():
	print ''
	print key_mapping_str()
	print '---------------------------------------'

class AdjustableStaticPublisher:

	def __init__(self,listener,defaulttype=None,useparams=False):
		self.useparams = useparams
		self.parser = None
		
		self.listener = listener
		self.sub = None
		self.pub = None
		self.br = None
		self.transform = None
		self.transform_frame = None
		
		self.x_offset = 0
		self.y_offset = 0
		self.z_offset= 0
		
		self.pos_increment = 0.005
		
		self.roll = 0
		self.pitch = 0
		self.yaw = 0
		
		self.angle_increment = 1. * pi / 180.
		
		self.invert = False
		self.invert_icrement = False
		
		self.defaulttype = defaulttype
		self.options = {}
		self.args = []
		
		self.interval = rospy.Duration(0.1)
		self.last_pub = None
		
		self.input_pose = None

	def get_Tmatrix(self,disableinvert=False):
		Tmatrix = dot(tft.translation_matrix([self.x_offset,self.y_offset,self.z_offset]),array(tb_to_tf(self.yaw, self.pitch, self.roll, rad=True)))
		
		if not disableinvert and self.invert:
			Tmatrix = tft.inverse_matrix(Tmatrix)
		
		return Tmatrix
		
	def set_from_matrix(self,T):
		pos = T[0:3,3]
		angles = get_tb_angles(T)
		self.x_offset = pos[0]
		self.y_offset = pos[1]
		self.z_offset = pos[2]
		self.yaw = angles.yaw
		self.pitch = angles.pitch
		self.roll = angles.roll
	
	def update_pos(self,icr_x,icr_y,icr_z):
		x = icr_x * self.pos_increment
		y = icr_y * self.pos_increment
		z = icr_z * self.pos_increment
		if not self.invert_icrement:
			self.x_offset += x
			self.y_offset += y
			self.z_offset += z
		else:
			p = tft.inverse_matrix(tft.translation_matrix([x,y,z]))
			self.set_from_matrix( dot( self.get_Tmatrix(), p ) )
	
	def update_angles(self,icr_y,icr_p,icr_r):
		y = icr_y * self.angle_increment
		p = icr_p * self.angle_increment
		r = icr_r * self.angle_increment
		if not self.invert_icrement:
			self.yaw += y
			self.pitch += p
			self.roll += r
		else:
			T = tft.inverse_matrix(array(tb_to_tf(y,p,r)))
			self.set_from_matrix( dot( T, self.get_Tmatrix() ) )
	
	def pose_callback(self,msg):
		self.input_pose = msg
		self.publish()
	
	def get_Tpose(self):
		if self.options.listen:
			if not self.input_pose: return None,None,None
			stamp = self.input_pose.header.stamp
			if not self.options.frame:
				frame_id = self.input_pose.header.frame_id
			else:
				frame_id = self.options.frame
			pose = self.input_pose.pose
			p = tft.translation_matrix([pose.position.x,pose.position.y,pose.position.z])
			rot = tft.quaternion_matrix([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
			Tinput_pose = dot(p,rot)
			if self.options.invert_input_pose:
				Tinput_pose = tft.inverse_matrix(Tinput_pose)
				frame_id = self.options.frame
			Tpose = dot(Tinput_pose, self.get_Tmatrix())
		else:
			frame_id = self.options.frame
			stamp = rospy.Time.now()
			Tpose = self.get_Tmatrix()
		
		if self.options.invert_output:
			Tpose = tft.inverse_matrix(Tpose)
		
		if self.options.tf:
			if self.options.invert_tf:
				from_frame = self.options.tf
				to_frame = frame_id
			else:
				from_frame = frame_id
				to_frame = self.options.tf
		else:
			from_frame = None
			to_frame = None
		
		frames = (frame_id,from_frame,to_frame)
		
		if self.options.invert_tf:
			Ttf = tft.inverse_matrix(Tpose)
		else:
			Ttf = Tpose
		
		return Tpose, Ttf, frames, stamp

	def publish(self):
		if self.last_pub and (rospy.Time.now() - self.last_pub) < self.interval: return
		
		(Tpose,Ttf,frames,stamp) = self.get_Tpose()
		frame_id = frames[0]
		if Tpose is None: return
		
		self.last_pub = rospy.Time.now()
		
		Tpose_p = Tpose[0:3,3]
		Tpose_q = tft.quaternion_from_matrix(Tpose)
			
		if self.options.pose:
			pub_msg = PoseStamped()
			pub_msg.header.stamp = stamp
			pub_msg.header.frame_id = frame_id
			pub_msg.pose.position = Point(*(Tpose_p.tolist()))
			pub_msg.pose.orientation = Quaternion(*(Tpose_q.tolist()))
			
			self.pub.publish(pub_msg)
		
		if self.options.tf:
			from_frame = frames[1]
			to_frame = frames[2]
			tf_p = Ttf[0:3,3]
			tf_q = tft.quaternion_from_matrix(Ttf)
			
			if self.options.tf_always_new:
				stamp = rospy.Time.now()
			
			self.br.sendTransform(tf_p,tf_q,stamp,to_frame,from_frame)
		

	def parse_args(self):
		base_indent = 9
		subsequent_indent = 3
		if self.defaulttype == 'pose':
			usage = 'Usage: %prog [options] [FRAME] TOPIC'
			pose_frame = lambda pose,tf,frame,listen: '%s %s' % (frame,pose)
			pose_listen = lambda pose,tf,frame,listen: '--listen %s %s' % (listen,pose)
			tf_frame = lambda pose,tf,frame,listen: None
			tf_listen = lambda pose,tf,frame,listen: None
			pose_tf_frame = lambda pose,tf,frame,listen: '--tf %s %s %s' % (tf,frame,pose)
			pose_tf_listen = lambda pose,tf,frame,listen: '--listen %s --tf %s %s' % (listen,tf,pose)
		elif self.defaulttype == 'tf':
			usage = 'Usage: %prog [options] PARENT CHILD'
			pose_frame = lambda pose,tf,frame,listen: None
			pose_listen = lambda pose,tf,frame,listen: None
			tf_frame = lambda pose,tf,frame,listen: '%s %s' % (frame,tf) 
			tf_listen = lambda pose,tf,frame,listen: None
			pose_tf_frame = lambda pose,tf,frame,listen: '--pose %s %s %s' % (pose,frame,tf)
			pose_tf_listen = lambda pose,tf,frame,listen: None
		else:
			usage = 'Usage: %prog [options] INPUT OUTPUT\n' + \
			_indent('where INPUT  = --listen TOPIC *OR* --frame PARENT\n' +\
			        '  and OUTPUT = --pose TOPIC   *OR* --tf CHILD\n',
			        base=base_indent,strip=False)
			pose_frame = lambda pose,tf,frame,listen: '--frame %s --pose %s' % (frame,pose)
			pose_listen = lambda pose,tf,frame,listen: '--listen %s --pose %s' % (listen,pose)
			tf_frame = lambda pose,tf,frame,listen: '--frame %s --tf %s' % (frame,tf) 
			tf_listen = lambda pose,tf,frame,listen: '--listen %s --tf %s' % (listen,tf)
			pose_tf_frame = lambda pose,tf,frame,listen: '--frame %s --tf %s --pose %s' % (frame,tf,pose)
			pose_tf_listen = lambda pose,tf,frame,listen: '--listen %s --tf %s --pose %s' % (listen,tf,pose)
		if self.useparams:
			usage = usage + _indent('\nLoads params from ROS, command line options will override\n',base=base_indent)
		
		usage = usage + _indent(base=base_indent,text="""
Publishes pose and tf. The pose can be given on the command line, loaded from a file, or taken from the ROS parameter server.
The pose can be published in isolation, or added to a pose received on a given topic.
There is also an interactive mode that allows the pose to be adjusted.
         
This utility holds a (modifiable) transform, which can be published as a pose and/or a transform. If a pose is published, the topic must be given. If a transform is published, the child frame must be given.
The parent frame for both pose and tf can be given with the --frame option.
Alternatively, the transform can be added to an existing published pose using the --listen option. In this case, the parent frame is is the frame of that pose. This utility with an identity transform can therefore be used to convert a pose into a tf frame.
The internal transform can be specified on the command line or loaded from a file, and all command line options can be additionally be given through ROS params.
""")

		#usage += '\n' + initial_indent + 'EXAMPLES:' + '\n'
		usage += _indent('\nEXAMPLES\n',base=base_indent)

		#dict of pose,tf,frame,listen, and text
		# -or-
		#call_fn, options, description
		examples = [ \
			{'pose':'/actuator_pose','tf':'/actuator_frame','frame':'/robot_frame','listen':'/robot_pose','text':lambda p,tf,f,l: 'Assume a pose is published on %s with frame %s' %(l,f)},
			(pose_frame,"",lambda p,tf,f,l: "Publish (interactive) pose on topic %s with frame %s" % (p,f)),
			(pose_listen,"",lambda p,tf,f,l: "Publish on topic %s the pose computed as the (interactive) pose added to the pose on topic %s" % (p,l)),
			(tf_frame,"",lambda p,tf,f,l: "Publish (interactive) tf with parent %s and child %s" % (f,tf)),
			(tf_listen,"",lambda p,tf,f,l: "Publish tf with child %s, using the parent frame from %s and transform computed as the (interactive) transform added to the pose from that topic" % (tf,l)),
			(pose_tf_frame,"",lambda p,tf,f,l: "Publish with parent frame %s the (interactive) transform as a pose on %s and as a tf with child frame %s" % (f,p,tf)),
			(pose_tf_listen,"",lambda p,tf,f,l: "Publish pose on %s and tf with child frame %s as the (interactive) transform added to the pose from topic %s and using the parent frame from that pose" % (p,tf,l)),
			{'pose':'N/A','tf':'/robot_frame','frame':'N/A/','listen':'/robot_pose','text':lambda p,tf,f,l: 'Assume a sensor detects the robot and publishes the pose in the sensor frame on the topic %s, and that there is tf from the world frame to the robot frame'%(l)},
			(tf_listen,"--invert-tf",lambda p,tf,f,l: "Publish the sensor frame in tf. Since the pose is in the sensor frame, the tf as normally published would have the sensor as parent and the robot as child, but to maintain the tf tree, the tf needs to be flipped to have the robot as parent and the sensor as child")]

		for example in examples:
			if isinstance(example,dict):
				example_pose = example['pose']
				example_tf = example['tf']
				example_frame = example['frame']
				example_listen = example['listen']
				usage += _indent('\n' + example['text'](example_pose,example_tf,example_frame,example_listen)+'\n',base=base_indent,subsequent=subsequent_indent)
				continue
			call_text = example[0](example_pose,example_tf,example_frame,example_listen)
			if call_text:
				example_call_str = "%prog " + "%s %s" % (example[1],call_text)
				usage += _indent('\n' + example_call_str + '\n',base=base_indent,subsequent=subsequent_indent) #+ \
						#initial_indent + subsequent_indent + example[2](example_pose,example_tf,example_frame,example_listen) + '\n'
				usage += _indent(example[2](example_pose,example_tf,example_frame,example_listen)+'\n',base=base_indent+subsequent_indent,subsequent=subsequent_indent)

		parser = OptionParser(usage=usage)
		self.parser = parser
		
		add_str_option = lambda name,the_default: parser.add_option('--' + name,action='store_true',default=_str_param(name,the_default))
		
		add_bool_option = lambda name: parser.add_option('--' + name,action='store_true',default=_bool_param(name,False))
		
		parser.add_option('--use-params',dest='params',action='store_true',default=self.useparams,help='Load options from ROS params')
		parser.add_option('--save-params',action='store_true',default=False,help='Save options to ROS params for this node')
		
		parser.add_option('-i','--load',help='Load tf and/or pose from FILE',metavar='FILE')
		parser.add_option("--load-only",choices=('tf','pose'),help="OPT='tf' or 'pose': Load only that data (automatically converts tf to pose and vice-versa if needed)",metavar='OPT')
		
		parser.add_option("-o", "--save",help="write final values to FILE", metavar="FILE")
		parser.add_option('--save-in-frame',help='Save pose and tf from parent FRAME',metavar='FRAME')
		parser.add_option('--save-pose-in-frame',help='Save pose from parent FRAME',metavar='FRAME')
		parser.add_option('--save-tf-in-frame',help='Save tf from parent FRAME',metavar='FRAME')
		parser.add_option("--save-only",choices=('tf','pose'),help="OPT='tf' or 'pose': Save only that data to file",metavar='OPT')
		
		parser.add_option('--static',action='store_true',help='Disable interactive mode')
		parser.add_option('--dynamic',dest='static',action='store_false',help='Force interactive mode')
		
		parser.add_option('--invert',action='store_true',default=False,help='Invert initial transform')
		parser.add_option('--invert-output',action='store_true',default=False)
		
		parser.add_option('--listen',help='listen for pose on TOPIC',metavar='TOPIC')
		parser.add_option('--invert-input-pose',action='store_true',default=False)
		#add_bool_option('invert-input-pose')
		
		
		parser.add_option('--pose',help='publish pose on TOPIC',metavar='TOPIC')
		
		parser.add_option('--tf',help='publish transform with frame FRAME',metavar='FRAME')
		parser.add_option('--invert-tf',action='store_true',default=False,help='Flip tf parent and child for publishing; use to maintain tf tree structure')
		#add_bool_option('invert-tf')
		parser.add_option('--tf-always-new',action='store_true',default=False,help='Always update timestamp on tf')
		#add_bool_option('tf-always-new')
		
		parser.add_option('--frame',default='',help='use FRAME as parent of transform and pose',metavar='FRAME')
		
		parser.add_option('--interval',default=0.1,type=float,help='The maximum interval for publishing')
		
		
		init_tf_group = OptionGroup(parser,'Initial transform')
		init_tf_group.add_option('-p','--pos',nargs=3,help='Translation',metavar='x y z')
		init_tf_group.add_option('-q','--quaternion',nargs=4,metavar='Qx Qy Qz Qw')
		
		
		
		init_tf_group.add_option('-e','--zyx',nargs=3,help='Fixed-axis Euler angles ZYX (ROS style)',metavar='z y x')
		parser.add_option_group(init_tf_group)
		
		init_tf_tb_group = OptionGroup(parser,'Tait-Bryan angles (airplane-style intrinsic rotations)')
		init_tf_tb_group.add_option('--tb',nargs=3,help='TB angles in degrees',metavar='yaw pitch roll')
		init_tf_tb_group.add_option('--tb-deg',nargs=3,help='TB angles in degrees',metavar='yaw pitch roll')
		init_tf_tb_group.add_option('--tb-rad',nargs=3,help='TB angles in radians',metavar='yaw pitch roll')
		parser.add_option_group(init_tf_tb_group)
		
		parser.add_option('-v','--version',action='store_true',default=False)
		
		(self.options, self.args) = parser.parse_args(args=rospy.myargv()[1:])
		
		if self.options.version:
			print __version__
			sys.exit(0)
		
		any_rot = lambda: self.options.quaternion or self.options.tb or self.options.tb_deg or self.options.zyx
		
		if self.options.params:
			rospy.loginfo('Setting option defaults from ros params')
			if self.options.static is None:
				self.options.static = True
			opts = parser.option_list
			for g in parser.option_groups:
				opts += g.option_list
			for opt in opts:
				if not opt.dest or not _has_param(opt.dest): continue
				if opt.action == 'store_true' or opt.action == 'store_false':
					setattr(self.options,opt.dest,_bool_param(opt.dest,opt.default))
				elif opt.type is float:
					setattr(self.options,opt.dest,float(_str_param(opt.dest,opt.default)))
				else:
					if opt.nargs > 1:
						val = _prep_str(_str_param(opt.dest,opt.default))
						if val:
							setattr(self.options,opt.dest,val.split(' '))
					else:
						setattr(self.options,opt.dest,_str_param(opt.dest,opt.default))
		else:
			if self.options.static is None:
				self.options.static = False
		
		
		if self.options.load:
			tf_file_data = TfFileData()
			tf_file_data.load(self.options.load,pose_frame_default_to_tf_parent=True)
			rospy.loginfo('Loading data from %s...' % self.options.load)
			if tf_file_data.tf and not self.options.load_only == 'pose':
				rospy.loginfo('Loaded tf')
				if tf_file_data.tf.from_frame and not self.options.frame:
					self.options.frame = tf_file_data.tf.from_frame
				if tf_file_data.tf.to_frame and not self.options.tf:
					self.options.tf = tf_file_data.tf.to_frame
				if not self.options.pos:
					self.options.pos = _prep_str(_get_tf_translation_tuple(tf_file_data.tf.transform)).split(' ')
				if not any_rot():
					self.options.quaternion = _prep_str(_get_tf_quaternion_tuple(tf_file_data.tf.transform)).split(' ')
			elif self.options.load_only == 'tf' and tf_file_data.pose:
				rospy.loginfo('Loaded tf from pose')
				if tf_file_data.pose.frame and not self.options.frame:
					self.options.frame = tf_file_data.pose.frame
				if tf_file_data.pose.name and not self.options.tf:
					self.options.tf = tf_file_data.pose.name
				if not self.options.pos:
					self.options.pos = _prep_str(_get_tf_translation_tuple(tf_file_data.pose.pose)).split(' ')
				if not any_rot():
					self.options.quaternion = _prep_str(_get_tf_quaternion_tuple(tf_file_data.pose.pose)).split(' ')
			
			if tf_file_data.pose and not self.options.load_only == 'tf':
				rospy.loginfo('Loaded pose')
				if tf_file_data.pose.frame and not self.options.frame:
					self.options.frame = tf_file_data.pose.frame
				if tf_file_data.pose.name and not self.options.pose:
					self.options.pose = tf_file_data.pose.name
				if not self.options.pos:
					self.options.pos = _prep_str(_get_tf_translation_tuple(tf_file_data.pose.pose)).split(' ')
				if not any_rot():
					self.options.quaternion = _prep_str(_get_tf_quaternion_tuple(tf_file_data.pose.pose)).split(' ')
			elif self.options.load_only == 'pose' and tf_file_data.tf:
				rospy.loginfo('Loaded pose from tf')
				if tf_file_data.tf.from_frame and not self.options.frame:
					self.options.frame = tf_file_data.tf.from_frame
				if tf_file_data.tf.from_frame and not self.options.pose:
					self.options.pose = tf_file_data.tf.to_frame
				if not self.options.pos:
					self.options.pos = _prep_str(_get_tf_translation_tuple(tf_file_data.tf.transform)).split(' ')
				if not any_rot():
					self.options.quaternion = _prep_str(_get_tf_quaternion_tuple(tf_file_data.tf.transform)).split(' ')
			rospy.loginfo('Done loading')
		
		if self.args:
			if self.defaulttype == 'pose':
				if len(self.args) > 1:
					self.options.frame = self.args[0]
					del self.args[0]
				if self.args:
					self.options.pose = self.args[0]
					del self.args[0]
				if not self.options.pose:
					parser.error('No pose given!')
			elif self.defaulttype == 'tf':
				if not self.options.frame:
					if self.args:
						self.options.frame = self.args[0]
						del self.args[0]
					else:
						parser.error('Must input parent tf frame!')
				if self.args:
					self.options.tf = self.args[0]
					del self.args[0]
				if not self.options.tf:
					parser.error('No tf child frame given!')
		
		#print self.options
		#print self.args
		
		if not self.options.pose and not self.options.tf:
			if rospy.has_param(rospy.resolve_name('~pose')):
				self.options.pose = rospy.get_param(rospy.resolve_name('~pose'))
			elif rospy.has_param(rospy.resolve_name('~tf')):
				self.options.tf = rospy.get_param(rospy.resolve_name('~tf'))
			else:
				parser.error('must select either pose or tf!')
		
		if not self.options.listen and not self.options.frame:
			if rospy.has_param(rospy.resolve_name('~listen')):
				self.options.listen = rospy.get_param(rospy.resolve_name('~listen'))
			elif rospy.has_param(rospy.resolve_name('~frame')):
				self.options.frame = rospy.get_param(rospy.resolve_name('~frame'))
			else:
				parser.error('no parent!')
		
		if self.options.pose:
			pose_info_str = 'Publishing pose on topic %s' % self.options.pose
			if self.options.listen:
				pose_info_str = pose_info_str + ', listening to %s' % self.options.listen
			if self.options.frame:
				pose_info_str = pose_info_str + ', with frame %s' % self.options.frame
			rospy.loginfo(pose_info_str)
		
		if self.options.tf:
			if self.options.frame:
				parent_frame = self.options.frame
			else:
				parent_frame = 'from topic %s' % self.options.listen
			child_frame = self.options.tf
			
			if self.options.invert_tf:
				parent_frame_tmp = child_frame
				child_frame = parent_frame
				parent_frame = parent_frame_tmp
			
			rospy.loginfo('Publishing tf with parent %s and child %s' % (parent_frame,child_frame))
				
		
		self.interval = rospy.Duration(self.options.interval)
		
		self.frame_id = self.options.frame
		
		self.invert = self.options.invert
		
		if self.options.pose:
			self.pub = rospy.Publisher(self.options.pose,PoseStamped)
		
		if self.options.tf:
			self.br = tf.TransformBroadcaster()
		
		if self.options.save_in_frame:
			self.options.save_tf_in_frame = self.options.save_in_frame;
			self.options.save_pose_in_frame = self.options.save_in_frame;
		
		if self.options.pos:
			self.x_offset = float(_prep_str(self.options.pos[0]))
			self.y_offset = float(_prep_str(self.options.pos[1]))
			self.z_offset = float(_prep_str(self.options.pos[2]))
		
		if self.options.quaternion:
			qx = float(_prep_str(self.options.quaternion[0]))
			qy = float(_prep_str(self.options.quaternion[1]))
			qz = float(_prep_str(self.options.quaternion[2]))
			qw = float(_prep_str(self.options.quaternion[3]))
			tb = get_tb_angles([qx,qy,qz,qw])
			self.yaw = tb.yaw_rad
			self.pitch = tb.pitch_rad
			self.roll = tb.roll_rad
		elif self.options.tb_rad:
			self.yaw = float(_prep_str(self.options.tb[0]))
			self.pitch = float(_prep_str(self.options.tb[1]))
			self.roll = float(_prep_str(self.options.tb[2]))
		elif self.options.tb or self.options.tb_deg:
			self.yaw = float(_prep_str(self.options.tb_deg[0])) * pi  / 180.
			self.pitch = float(_prep_str(self.options.tb_deg[1])) * pi  / 180.
			self.roll = float(_prep_str(self.options.tb_deg[2])) * pi  / 180.
		elif self.options.zyx:
			parser.error('zyx not implemented yet')
	
	def print_params(self):
		print ''
		if self.invert:
			print "********INVERTED*******"
		if self.invert_icrement:
			print "**INVERTED INCREMENTS**"
		print "Position offset:         ({0:.3f}, {1:.3f}, {2:.3f})".format(self.x_offset,self.y_offset,self.z_offset)
		print "Position increment:       {0}".format(self.pos_increment)
		print "Angles (yaw,pitch,roll): ({0:.1f}, {1:.1f}, {2:.1f})".format(self.yaw * 180 / pi,self.pitch * 180 / pi,self.roll * 180 / pi)
		print "Angle    increment:       {0:.1f} deg".format(self.angle_increment * 180 / pi)
		
		(Tpose,Ttf,frame_id,stamp) = self.get_Tpose()

		Tpose_p = Tpose[0:3,3]
		Tpose_tb = get_tb_angles(Tpose)
		
		print "Current pose: (%.3f, %.3f, %.3f) [yaw: %.1f, pitch: %.1f, roll: %.1f]" % (
				Tpose_p[0],Tpose_p[1],Tpose_p[2],
				Tpose_tb.yaw_deg, Tpose_tb.pitch_deg, Tpose_tb.roll_deg)
		
		print '---------------------------------'
	
	def run(self):
		self.parse_args()
	
		rospy.loginfo('initial params: (%f,%f,%f), tb:ypr (%f, %f, %f)' % (
				self.x_offset,self.y_offset,self.z_offset,
				self.yaw * 180 / pi, self.pitch * 180 / pi, self.roll * 180 /pi))
		
		if self.options.listen:
			self.sub = rospy.Subscriber(self.options.listen,PoseStamped,self.pose_callback)
			rospy.loginfo('waiting for message on %s...' % self.options.listen)
			while not rospy.is_shutdown():
				try:
					rospy.wait_for_message(self.options.listen,PoseStamped,1)
					break;
				except Exception, e:
					pass
		
		if rospy.is_shutdown():
			return
		
		timer = rospy.Timer(self.interval,lambda timerevent: self.publish())
		
		rospy.loginfo('ready: %s' % rospy.get_name())
		
		if self.options.static:
			rospy.spin()
		else:
			_print_key_mapping()
			while not rospy.is_shutdown():
				try:
					#self.pub_tf()
					getch = _Getch()
					#cmd = '_';
					cmd = getch().lower()
					if ord(cmd) == 3:
						rospy.signal_shutdown('user terminated')
					else:
						if cmd == KEYS['help'].lower():
							_print_key_mapping()
						elif cmd == KEYS['x+'].lower():
							self.update_pos( 1, 0, 0)
						elif cmd == KEYS['x-'].lower():
							self.update_pos(-1, 0, 0)
						elif cmd == KEYS['y+'].lower():
							self.update_pos( 0, 1, 0)
						elif cmd == KEYS['y-'].lower():
							self.update_pos( 0,-1, 0)
						elif cmd == KEYS['z+'].lower():
							self.update_pos( 0, 0, 1)
						elif cmd == KEYS['z-'].lower():
							self.update_pos( 0, 0,-1)
						elif cmd == KEYS['yaw+'].lower():
							self.update_angles( 1, 0, 0)
						elif cmd == KEYS['yaw-'].lower():
							self.update_angles(-1, 0, 0)
						elif cmd == KEYS['pitch+'].lower():
							self.update_angles( 0, 1, 0)
						elif cmd == KEYS['pitch-'].lower():
							self.update_angles( 0,-1, 0)
						elif cmd == KEYS['roll+'].lower():
							self.update_angles( 0, 0, 1)
						elif cmd == KEYS['roll-'].lower():
							self.update_angles( 0, 0,-1)
						elif cmd == KEYS['print_params'].lower():
							self.print_params()
						elif cmd == KEYS['change_increment_pos'].lower():
							print 'Current pos increment: {0}'.format(self.pos_increment)
							l = raw_input('Enter value or hit return: ');
							if not l:
								try:
									self.pos_increment = float(l)
								except Exception, e:
									pass
							print 'Value saved: {0}'.format(self.pos_increment)
						elif cmd == KEYS['change_increment_angle'].lower():
							print 'Current angle increment: {0} deg'.format(self.angle_increment * 180 / pi)
							l = raw_input('Enter value in degrees or hit return: ');
							if l:
								try:
									self.angle_increment = float(l) * pi / 180.
								except Exception, e:
									pass
							print 'Value saved: {0} deg'.format(self.angle_increment * 180 / pi)
						elif cmd == KEYS['invert_increment'].lower():
							self.invert_icrement = not self.invert_icrement
							if self.invert_icrement:
								print 'Transform increments are now inverted'
							else:
								print 'Transform increments are now normal'
						elif cmd == KEYS['invert_transform'].lower():
							self.invert = not self.invert
							if self.invert:
								print 'Transform is now inverted'
							else:
								print 'Transform is now normal'
						elif cmd == KEYS['set'].lower():
							POS_KEY = 'P'
							ANGLE_KEY = 'A'
							print 'Choose position (%s) or angles (%s)' % (POS_KEY.upper(), ANGLE_KEY.upper())
							cmd = getch().lower()
							try:
								if cmd == POS_KEY.lower():
									l = raw_input('Enter new offsets or enter field: ')
									l = l.lower().strip()
									if l:
										if l[0] == 'x':
											self.x_offset = float(l[1:])
										elif l[0] == 'y':
											self.y_offset = float(l[1:])
										elif l[0] == 'z':
											self.z_offset = float(l[1:])
										else:
											nums = _prep_str(l).split(' ')
											self.x_offset = float(nums[0])
											self.y_offset = float(nums[1])
											self.z_offset = float(nums[2])
								elif cmd == ANGLE_KEY.lower():
									RADIAN_PREFIX='*'
									l = raw_input(
											'Enter new rotation:\n' + \
											'  For tb angles, enter \'[%s] <yaw> <pitch> <roll>\'\n' % RADIAN_PREFIX + \
											'    or \'[%s] <field> <val>\'  (prefix with * for radians)\n' % RADIAN_PREFIX + \
											'  For quaternion, enter \'q <qx> <qy> <qz> <qw>\n')
									l = l.lower().strip()
									if l:
										conv = pi/180
										if l[0] == 'q':
											if not re.match(r'[-+0-9]',l[1]):
												ind = l.find(' ')
											else:
												ind = 1
											nums = _prep_str(l[ind:]).split(' ')
											qmat = tft.quaternion_matrix([float(nums[0]),float(nums[1]),float(nums[2]),float(nums[3])])
											angles = get_tb_angles(qmat)
											print "setting angles to ({0},{1},{2})".format(angles.yaw_deg,angles.pitch_deg,angles.roll_deg)
											self.yaw = angles.yaw
											self.pitch = angles.pitch
											self.roll = angles.roll
										else:
											if l[0] == RADIAN_PREFIX:
												conv = 1
												l = l[1:].strip()
											if re.match(r'[-+0-9]',l[0]):
												nums = _prep_str(l).split(' ')
												self.yaw = float(nums[0]) * conv
												self.pitch = float(nums[1]) * conv
												self.roll = float(nums[2]) * conv
											else:
												field = l.split(' ')[0]
												l = l[l.find(' '):].strip()
												if l[0] == RADIAN_PREFIX:
													conv = 1
													l = l[1:].strip()
												if field[0] == 'y':
													self.yaw = float(l) * conv
												elif field[0] == 'p':
													self.pitch = float(l) * conv
												elif field[0] == 'r':
													self.roll = float(l) * conv
												else:
													raise Exception('Unrecognized field!')
								self.print_params()
							except Exception, e:
								print 'Could not set params:', e
				except Exception, e:
					print e
				rospy.sleep(0.1)
		rospy.loginfo('Shutting down...')
		if self.options.save:
			filename = self.options.save
			rospy.loginfo('Saving to %s...' % filename)
			(Tpose,Ttf,frames,stamp) = self.get_Tpose()
			tf_file_data = TfFileData()
			if self.options.tf or self.options.save_only == 'tf':
				from_frame=frames[1]
				to_frame=frames[2]
				if self.options.save_tf_in_frame:
					rospy.loginfo('waiting for transform from %s to %s' % (self.options.save_tf_in_frame,frames[1]))
					self.listener.waitForTransform(self.options.save_tf_in_frame,frames[1],rospy.Time(),rospy.Duration(5))
					T_ = transform(self.listener.lookupTransform(self.options.save_tf_in_frame,frames[1],rospy.Time()))
					Ttf = T_ * Ttf
					from_frame = self.options.save_tf_in_frame
				tf_file_data.set_tf(from_frame,to_frame,transform=Ttf,stamp=stamp)
			if self.options.pose or self.options.save_only == 'pose':
				if self.options.save_pose_in_frame:
					if self.options.save_pose_in_frame != self.options.save_tf_in_frame:
						rospy.loginfo('waiting for transform from %s to %s' % (self.options.save_pose_in_frame,frames[0]))
						self.listener.waitForTransform(self.options.save_pose_in_frame,frames[0],rospy.Time(),rospy.Duration(5))
					T_ = transform(self.listener.lookupTransform(self.options.save_pose_in_frame,frames[0],rospy.Time()))
					Tpose = T_ * Tpose
					frame = self.options.save_pose_in_frame
				tf_file_data.set_pose(frame=frames[0],pose=Tpose,stamp=stamp)
			rospy.loginfo('writing...')
			tf_file_data.write(filename)
			rospy.loginfo('done')
		if self.options.save_params:
			rospy.loginfo('Saving params...')
			for opt in self.parser.option_list:
				if not opt.dest or getattr(self.options,opt.dest) is None: continue
				if opt.dest == 'params' or opt.dest == 'save_params' or opt.dest == 'static' or opt.dest == 'load' or opt.dest == 'save':
					continue
				rospy.set_param(_param_name(opt.dest),str(getattr(self.options,opt.dest)))
