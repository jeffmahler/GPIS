class DexControlInterface:
	'''
	Acts as intermediary interface between DexController the specific robot.
	This interface relies on the controller in the RobotProfile to control the actual robot
	'''
	
	def __init__(self, robot, comm, baudrate, timeout):
		self._robot = robot
		self.comm = comm
		self.baudrate = baudrate
		self.timeout = timeout
		
	def reset(self):
		self._robot.reset()
		