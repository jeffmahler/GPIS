from operator import add
from tfx import pose, transform, vector, rotation
from DexConstants import DexConstants
from PyControl import PyControl

class DexRobotZeke:
	'''
	Abstraction for a robot profile. Contains all information specific
	to the Zeke robot, including its physical dimensions, joints
	accepted poses, etc. 
	'''

	NUM_STATES = 6
	#TODO: find actual physical values
	# Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
	MIN_STATES = [0,.02,.01,0.183086039735,-.01,0.197775641646]
	MAX_STATES = [6.58991792126, .3, 0.3, 6.61606370861, 0.0348490572685, 6.83086639225]
	
	RESET_STATE = [2.3931, 1.4635, 1.4513, 3.4204, 0.0348, 2.842]
	ZEKE_LOCAL_T = transform(
											vector(-10, 0, 0), 
											rotation.identity(), 
											parent=DexConstants.WORLD_FRAME,
											frame="ZEKE_LOCAL")
	
	def __init__(self, comm, baudrate, time):
		self.comm = comm 
		self.baudrate = baudrate
		self.time = time
	
	def reset(self):
		self.setState(DexRobotZeke.RESET_STATE)
		return
	
	def _execute_once(self, action):
		return_val = None
		zeke = PyControl(self.comm, self.baudrate, self.time)
		zeke.stop()
		try:
			return_val = action(zeke)
		except Exception, e:
			print e
		finally:
			zeke.ser.close()
		return return_val
	
	def move(self, controls):
		def action_move(zeke):
			print "Before: ", zeke.getState()
			print "Controlling: ", controls
			zeke.control(controls)	
			print "After: ", zeke.getState()
		self._execute_once(action_move)
	
	def stop(self):
		self._execute_once(lambda zeke: zeke.stop())

	def setState(self, state):
		def action_setState(zeke):
		
			for i in range(DexRobotZeke.NUM_STATES):
				if not (DexRobotZeke.MIN_STATES[i] <= state[i] <= DexRobotZeke.MAX_STATES[i]):
					raise Exception("State " + str(i) + " is out of zeke bounds: " + str(state[i]))
			print "Before: ", zeke.getState()
			print "Sent State: ", state
			zeke.sendStateRequest(state)
		self._execute_once(action_setState)
	
	def changeState(self, deltas):
		cur_state = self.getState()
		target_state = map(add, cur_state, deltas)
		self.setState(target_state)
	
	def getState(self):
		return self._execute_once(lambda zeke: zeke.getState())
		
	def _pose_IK(self, pose):
		'''
		Takes in a pose and returns the following list of joint settings:
		Elevation
		Rotation about Z axis
		Extension of Arm
		Rotation of gripper
		'''
		settings = {}
		settings["elevation"] = pose.position.z
		
		#calculate rotation about z axis
		x = pose.position.x
		y = pose.position.y
		theta = 0
		if x == 0:
			if y > 0:
				theta = np.pi / 2
			else: 
				theta = - np.pi / 2
		else:
			theta_ref = abs(np.arctan(y/x))
			if x >= 0 and y >= 0:
				theta = theta_ref
			elif y >= 0 and x < 0:
				theta = np.pi - theta_ref
			elif y < 0 and x < 0:
				theta = np.pi + theta_ref
			else:
				theta = 2*np.pi - theta_ref
		
		settings["rot_z"] = theta
		settings["extension"] = sqrt(pow(x, 2) + pow(y, 2))
		settings["rot_y"] = pose.rotation.euler['sxyz'][1]
		
		return settings
		
	def _settings_to_state(self, settings):
		'''
		Takes in a list of joint settings and concats them into one single 
		final target state. Basically forward kinematics
		'''
		# Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
		state = [0] * 6
		state[0] = settings["rot_z"]
		state[1] = settings["elevation"]
		state[2] = settings["extension"]
		state[3] = settings["rot_z"]
		state[4] = MIN_STATES[4] #TODO: verify this is open gripper
		state[5] = MIN_STATES[5]
		
	def transform(self, target_pose):
		target_pose = ZEKE_LOCAL_T * target_pose
		
		if abs(target_pose.rotation.euler['sxyz'][0]) >= 0.0001:
			raise Exception("Can't perform rotation about x-axis on Zeke's gripper")

		joint_settings = self._pose_IK(target_pose)
		target_state = self._settings_to_state()
		
		self.setState(target_state)