#!/usr/bin/env python

import roslib
roslib.load_manifest('tfx')

from tfx.tb_angles import *
from tfx.tb_angles import _FLOAT_CLOSE_ENOUGH
import unittest
import datetime, time
import rospy
import std_msgs.msg
import geometry_msgs.msg
import numpy as np
import operator
from math import *
from collections import defaultdict
import tf.transformations as tft
import random
import re

class TestTbAngles(unittest.TestCase):
	def setUp(self):
		self.yaw = [0,30,45,60,90,180,360]
		self.yaw += [-v for v in self.yaw[1:]]
		self.pitch = [0,30,45,60,90,180,360]
		self.pitch += [-v for v in self.pitch[1:]]
		self.roll = [0,30,45,60,90,180,360]
		self.roll += [-v for v in self.roll[1:]]
		pass
	
	def compare_deg(self,tb,yaw,pitch,roll):
		self.assertAlmostEqual(tb.yaw_deg, yaw)
		self.assertAlmostEqual(tb.pitch_deg, pitch)
		self.assertAlmostEqual(tb.roll_deg, roll)
		self.assertAlmostEqual(tb.yaw_rad, yaw * pi / 180.)
		self.assertAlmostEqual(tb.pitch_rad, pitch * pi / 180.)
		self.assertAlmostEqual(tb.roll_rad, roll * pi / 180.)
		
	
	def compare_rad(self,tb,yaw,pitch,roll):
		self.assertAlmostEqual(tb.yaw_rad, yaw)
		self.assertAlmostEqual(tb.pitch_rad, pitch)
		self.assertAlmostEqual(tb.roll_rad, roll)
		self.assertAlmostEqual(tb.yaw_deg, yaw * 180. / pi)
		self.assertAlmostEqual(tb.pitch_deg, pitch * 180. / pi)
		self.assertAlmostEqual(tb.roll_deg, roll * 180. / pi)

	def assertMsgEqual(self,msg1,msg2):
		self.assertTrue(
					np.allclose([msg1.x,msg1.y,msg1.z,msg1.w],[msg2.x,msg2.y,msg2.z,msg2.w]) or \
					np.allclose([msg1.x,msg1.y,msg1.z,msg1.w],[-msg2.x,-msg2.y,-msg2.z,-msg2.w]),
					msg='msgs not equal! %s\n\n%s' % (msg1,msg2))
	
	def assertTbEqual(self,tb,yaw,pitch,roll):
		q = tb.quaternion
		q2 = tft.quaternion_from_euler(yaw * pi / 180,pitch * pi / 180,roll * pi / 180,'rzyx')
		eq = np.allclose(q,q2) or np.allclose(-q,q2)
		msg = '%s not (%f,%f,%f)' % (tb.tostring(),yaw,pitch,roll)
		
		self.assertTrue(eq,msg=msg)
	
	def getMsg(self,q):
		msg = geometry_msgs.msg.Quaternion()
		msg.x = q[0]
		msg.y = q[1]
		msg.z = q[2]
		msg.w = q[3]
		return msg
		
	#********************** create **************************
	
	def test_create(self):
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					self.compare_deg(tb, yaw, pitch, roll)
					tb = tb_angles(yaw,pitch,roll,deg=True)
					self.compare_deg(tb, yaw, pitch, roll)
					tb = tb_angles(yaw,pitch,roll,rad=False)
					self.compare_deg(tb, yaw, pitch, roll)
					
					yaw_rad = yaw * pi / 180.
					pitch_rad = pitch * pi / 180.
					roll_rad = roll * pi / 180.
					tb = tb_angles(yaw_rad,pitch_rad,roll_rad,rad=True)
					self.compare_rad(tb, yaw_rad, pitch_rad, roll_rad)
					tb = tb_angles(yaw_rad,pitch_rad,roll_rad,deg=False)
					self.compare_rad(tb, yaw_rad, pitch_rad, roll_rad)
					
	
	def test_create_dict(self):
		vals = {'yaw':self.yaw[2],'pitch':self.pitch[4],'roll':self.roll[9]}
		for yaw_key in ['yaw','y',None]:
			for pitch_key in ['pitch','p',None]:
				for roll_key in ['roll','r',None]:
					d = {}
					v = defaultdict(float)
					for key1,key2 in zip(['yaw','pitch','roll'],[yaw_key,pitch_key,roll_key]):
						if key2 is None:
							continue
						d[key2] = vals[key1]
						v[key1] = vals[key1]
					
					if not d:
						continue
					tb = tb_angles(d)
					self.compare_deg(tb, v['yaw'], v['pitch'], v['roll'])
					
					tb = tb_angles(**d)
					self.compare_deg(tb, v['yaw'], v['pitch'], v['roll'])
	
	#msg
	def test_create_msg(self):
		for _ in xrange(50):
			q = tft.random_quaternion()
			q_msg = self.getMsg(q)
			tb = tb_angles(q_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
			self.assertMsgEqual(q_msg, tb.msg)
			self.assertMsgEqual(q_msg, tb.to_quaternion_msg())
			
			qs_msg = geometry_msgs.msg.QuaternionStamped()
			qs_msg.quaternion = q_msg
			tb = tb_angles(qs_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
			
			pose_msg = geometry_msgs.msg.Pose()
			pose_msg.orientation = q_msg
			tb = tb_angles(pose_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
			
			pose_stamped_msg = geometry_msgs.msg.PoseStamped()
			pose_stamped_msg.pose.orientation = q_msg
			tb = tb_angles(pose_stamped_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
			
			tf_msg = geometry_msgs.msg.Transform()
			tf_msg.rotation = q_msg
			tb = tb_angles(tf_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
			
			tf_stamped_msg = geometry_msgs.msg.TransformStamped()
			tf_stamped_msg.transform.rotation = q_msg
			tb = tb_angles(tf_stamped_msg)
			self.assertTrue(np.allclose(q, tb.quaternion) or np.allclose(-q, tb.quaternion),msg='%s and %s not close!' % (list(q),tb.to_quaternion_list()))
	
	def test_create_matrix(self):
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					m = tft.euler_matrix(yaw * pi / 180., pitch * pi / 180., roll * pi / 180., 'rzyx')
					tb = tb_angles(m)
					self.assertTbEqual(tb, yaw, pitch, roll)
					
					m2 = m[0:3,0:3]
					tb = tb_angles(m2)
					self.assertTbEqual(tb, yaw, pitch, roll)
					
					m3 = np.mat(m)
					tb = tb_angles(m3)
					self.assertTbEqual(tb, yaw, pitch, roll)
					
					m4 = m.tolist()
					tb = tb_angles(m4)
					self.assertTbEqual(tb, yaw, pitch, roll)
	
	def test_create_quat(self):
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					q = tft.quaternion_from_euler(yaw * pi / 180., pitch * pi / 180., roll * pi / 180., 'rzyx')
					tb = tb_angles(q)
					self.assertTbEqual(tb, yaw, pitch, roll)
					
					q2 = list(q)
					tb = tb_angles(q2)
					self.assertTbEqual(tb, yaw, pitch, roll)
	
	def test_invalid_sequences(self):
		with self.assertRaises(ValueError):
			tb_angles([1,2,3])
		
		with self.assertRaises(ValueError):
			tb_angles([[1,0,0],[0,1,0],[0,0]])
			
	
	#axis-angle
	def test_create_axis_angle(self):
		axis = tft.random_vector(3)
		angle = random.random() * 2 * pi
		
		q = tft.quaternion_about_axis(angle, axis)
		
		q2 = tb_angles(axis,angle).quaternion
		
		self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close!' % (list(q),list(q2)))
		
		q2 = tb_angles(angle,axis).quaternion
		
		self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close!' % (list(q),list(q2)))
		
		q2 = tb_angles((axis,angle)).quaternion
		
		self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close!' % (list(q),list(q2)))
		
		q2 = tb_angles((angle,axis)).quaternion
		
		self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close!' % (list(q),list(q2)))
		
		for _ in xrange(1000):
			axis = tft.random_vector(3)
			angle = random.random() * 2 * pi
			
			q = tft.quaternion_about_axis(angle, axis)
			q2 = tb_angles(axis,angle).quaternion
			
			self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close! for axis %s and angle %f' % (list(q),list(q2),tuple(axis),angle))
	
	#********************** convert **************************
	
	#quaternion
	def test_to_quat(self):
		tb = tb_angles(45,-5,24)
		q1 = tb.quaternion
		self.assertFalse(q1.flags.writeable)
		q1a = tb.quaternion
		self.assertIs(q1, q1a)
		
		q2 = tb.to_quaternion()
		self.assertTrue(q2.flags.writeable)
		self.assertIsNot(q1,q2)
		
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					q = tb.quaternion
					self.assertAlmostEqual(np.linalg.norm(q), 1)
					self.assertGreater(q[3], 0) #normalized w > 0
					q2 = tft.quaternion_from_euler(yaw * pi / 180, pitch * pi / 180, roll * pi / 180, 'rzyx')
					self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg='%s and %s not close!' % (list(q),list(q2)))
	
	#matrix
	def test_to_mat(self):
		tb = tb_angles(45,-5,24)
		m1 = tb.matrix
		self.assertFalse(m1.flags.writeable)
		m1a = tb.matrix
		self.assertIs(m1, m1a)
		
		m2 = tb.to_matrix()
		self.assertTrue(m2.flags.writeable)
		self.assertIsNot(m1,m2)
		
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					m = tb.matrix
					self.assertAlmostEqual(np.linalg.det(m), 1)
					m2 = tft.euler_matrix(yaw * pi / 180, pitch * pi / 180, roll * pi / 180, 'rzyx')[0:3,0:3]
					self.assertTrue(np.allclose(m, m2),msg='%s and %s not close!' % (m,m2))
	
	#tf
	def test_to_tf(self):
		tb = tb_angles(45,-5,24)
		self.assertTrue(tb.to_tf().flags.writeable)
		
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					m = tb.to_tf()
					m2 = tft.euler_matrix(yaw * pi / 180, pitch * pi / 180, roll * pi / 180, 'rzyx')
					self.assertTrue(np.allclose(m, m2),msg='%s and %s not close!' % (m,m2))
	
	
	#axis-angle
	def test_to_axis_angle(self):
		for _ in xrange(1000):
			axis = tft.random_vector(3)
			axis = axis / np.linalg.norm(axis)
			for angle in list(np.linspace(-pi, pi, 10)) + [0]:
			
				q = tft.quaternion_about_axis(angle, axis)
				axis2,angle2 = tb_angles(q).axis_angle
				q2 = tft.quaternion_about_axis(angle2, axis2)
				
				self.assertTrue(np.allclose(q, q2) or np.allclose(-q, q2),msg="axis %s and angle %f don't match %s, %f" % (tuple(axis),angle,tuple(axis2),angle2))
	
	#********************** string methods **************************
	
	def test_str(self):
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					for brackets in ['[]','()']:
						for deg in [None,True,False]:
							for rad in [None,True,False]:
								if deg is False and rad is False:
									continue
								for short_names in [None,True,False]:
									if short_names:
										y_str = 'y'
										p_str = 'p'
										r_str = 'r'
									else:
										y_str = 'yaw'
										p_str = 'pitch'
										r_str = 'roll'
										
									for fixed_width in [None,True,False]:
										if fixed_width:
											deg_str = r'([ -]\d{3}| [ -]\d{2}|  [ -]\d{1})\.\d{1}'
											rad_str = r'[ -]\d{1}\.\d{3}'
										else:
											deg_str = r'-?\d{1,3}\.\d{1}'
											rad_str = r'-?\d{1}\.\d{3}'
										
										def add_num(regex):
											if deg or (deg is None and rad is None):
												regex += deg_str
											if deg is True:
												regex += ' deg'
											if rad:
												if deg:
													regex += r' \('
												regex += rad_str + ' rad'
												if deg:
													regex += r'\)'
											return regex
										
										regex = '^\\' + brackets[0]
										regex += y_str + ':'
										regex = add_num(regex)
										regex += ', '
										regex += p_str + ':'
										regex = add_num(regex)
										regex += ', '
										regex += r_str + ':'
										regex = add_num(regex)
										regex += '\\' + brackets[1] + '$'
										
										if deg is False and rad is False:
											with self.assertRaises(ValueError):
												s = tb.tostring(brackets=brackets,deg=deg,rad=rad,short_names=short_names,fixed_width=fixed_width)
										else:
											s = tb.tostring(brackets=brackets,deg=deg,rad=rad,short_names=short_names,fixed_width=fixed_width)
											self.assertRegexpMatches(s, regex,msg = '%s' % [brackets,deg,rad,short_names,fixed_width])											
							
	
	def test_format(self):
		for yaw in self.yaw:
			for pitch in self.pitch:
				for roll in self.roll:
					tb = tb_angles(yaw,pitch,roll)
					for (deg,deg_fmt) in zip([True,False],['d','']):
						for (rad,rad_fmt) in zip([True,False],['r','']):
							for (short_names,short_names_fmt) in zip([True,False],['s','']):
								if short_names:
									y_str = 'y'
									p_str = 'p'
									r_str = 'r'
								else:
									y_str = 'yaw'
									p_str = 'pitch'
									r_str = 'roll'
									
								for (fixed_width,fixed_width_fmt) in zip([True,False],['f','']):
									if fixed_width:
										deg_str = r'([ -]\d{3}| [ -]\d{2}|  [ -]\d{1})\.\d{1}'
										rad_str = r'[ -]\d{1}\.\d{3}'
									else:
										deg_str = r'-?\d{1,3}\.\d{1}'
										rad_str = r'-?\d{1}\.\d{3}'
									
									def add_num(regex):
										if deg or (deg is False and rad is False):
											regex += deg_str
										if deg is True:
											regex += ' deg'
										if rad:
											if deg:
												regex += r' \('
											regex += rad_str + ' rad'
											if deg:
												regex += r'\)'
										return regex
									
									regex = '^\\' + '['
									regex += y_str + ':'
									regex = add_num(regex)
									regex += ', '
									regex += p_str + ':'
									regex = add_num(regex)
									regex += ', '
									regex += r_str + ':'
									regex = add_num(regex)
									regex += '\\' + ']' + '$'
									
									fmt_str = '{:' + deg_fmt + rad_fmt + short_names_fmt + fixed_width_fmt + '}'
									
									s = fmt_str.format(tb)
									self.assertRegexpMatches(s, regex,msg = '%s' % [deg,rad,short_names,fixed_width])

if __name__ == '__main__':
	unittest.main()