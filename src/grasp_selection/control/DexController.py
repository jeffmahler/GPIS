from DexRobotZeke import DexRobotZeke
from DexConstants import DexConstants
import tfx
class DexController:
	'''Transformation Controller class. Currently robot defaults to Zeke
	Usage Pattern: To be instantiated once per robot
	Goal: Takes in a target pose and controls the Zeke robot to achieve that pose
	'''
	
	def __init__(self, 	robot = None,
			comm = "COM3", baudrate=115200, timeout=.01):
		#instantiating variables
		if robot is None:
			robot = DexRobotZeke(comm, baudrate, timeout)
		self.robot = robot

	def transform(self, target_pose):
		'''
		target_pose is a tfx.pose object
		'''
		if target_pose.frame is not DexConstants.WORLD_FRAME:
			raise Exception("Given target_pose is not in WORLD frame")
		self.robot.transform(target_pose)
		
	def reset(self):
		self.robot.reset()
		
	def stop(self):
		self.robot.stop()
		
ctrl = DexController()
origin = tfx.pose(DexConstants.ORIGIN, tfx.rotation.identity(), frame = DexConstants.WORLD_FRAME)