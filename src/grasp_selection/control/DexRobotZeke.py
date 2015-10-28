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

	#TODO: find actual physical values
	RESET_STATE = [2.3931, 1.4635, 1.4513, 3.4204, 0.0348, 2.842]
	ZEKE_LOCAL_T = transform(
											vector(0, 0, 0), 
											DexConstants.I_ROT, 
											frame=DexConstants.WORLD_FRAME)
	
	def __init__(self, comm, baudrate, time):
		self.comm = comm 
		self.baudrate = baudrate
		self.time = time
	
	def reset(self):
		self.setState(DexRobotZeke.RESET_STATE)
		return
	
	def _execute_once(self, action):
		zeke = PyControl(self.comm, self.baudrate, self.time)
		zeke.stop()
		return_val = action(zeke)
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
		
	def _process(self, pose):
		
		
		
	def transform(self, target_pose):
		target_pose = ZEKE_LOCAL_T * target_pose
		
		