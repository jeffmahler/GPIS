from DexRobotZeke import DexRobotZeke
from DexRobotTurntable import DexRobotTurntable
from DexConstants import DexConstants
from tfx import pose, rotation, rotation_euler
from numpy import pi
class DexController:
    '''Transformation Controller class. Currently robot defaults to Zeke
    Usage Pattern: To be instantiated once per robot
    Goal: Takes in a target pose and controls the Zeke robot to achieve that pose
    '''
    
    def __init__(self, robot = None, comm = DexConstants.COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        #instantiating variables
        if robot is None:
            robot = DexRobotZeke(comm, baudrate, timeout)
        self._robot = robot

    def transform(self, target_pose, name = None, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        #target_pose is a tfx.pose object
        if target_pose.frame is not DexConstants.WORLD_FRAME:
            raise Exception("Given target_pose is not in WORLD frame")
        self._robot.transform(target_pose, rot_speed, tra_speed, name)
        
    def gotoState(self, state, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        self._robot.gotoState(state, rot_speed, tra_speed)
        
    def reset(self, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        self._robot.reset(rot_speed, tra_speed)
        
    def pause(self, s):
        self._robot.maintainState(s)
        
    def stop(self):
        self._robot.stop()
        
    def getState(self):
        return self._robot.getState()
        
    def grip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self._robot.grip(tra_speed)
    
    def unGrip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self._robot.unGrip(tra_speed)
        
    def plot(self):
        self._robot.plot()

origin = pose(DexConstants.ORIGIN, DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)
raised = pose((0, 0, 0.15), DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)

def test():
    target = pose((0.05, 0.05, 0.1), rotation_euler(0, pi/2, pi, 'sxyz'), frame = DexConstants.WORLD_FRAME)
    t = DexController(DexRobotTurntable())
    t.reset()
    t.transform(target)
    t.plot()
    t.stop()

'''
def angle(ctrl, angle):
    ctrl.gotoState(ctrl.getState().set_gripper_rot(angle))
    
def dAngle(ctrl, delta):
    ctrl.gotoState(ctrl.getState().set_gripper_rot(ctrl.getState().gripper_rot + delta))
'''