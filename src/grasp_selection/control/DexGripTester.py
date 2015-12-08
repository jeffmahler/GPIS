from tfx import point, pose, rotation
from numpy import pi
from DexConstants import DexConstants
from DexController import DexController
from ZekeState import ZekeState
from DexRobotZeke import DexRobotZeke
from Logger import Logger

class DexGripTester:

    def __init__(self, tra_mult = 0.7, rot_mult = 0.7, robot = None, comm = DexConstants.COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._ctrl = DexController(robot, comm, baudrate, timeout)
        self.set_tra_speed(tra_mult)
        self.set_rot_speed(rot_mult)
        
    def set_tra_speed(self, val):
        val = min(abs(val), 1)
        self._tra_speed = val * DexConstants.MAX_TRA_SPEED        

    def set_rot_speed(self, val):
        val = min(abs(val), 1)        
        self._rot_speed = val * DexConstants.MAX_ROT_SPEED

    def _moveArm(self, state, n):        
        retract_state = state.copy().set_arm_ext(DexConstants.MIN_STATE.arm_ext + 0.05)
        extend_state = state.copy().set_arm_ext(DexConstants.MAX_STATE.arm_ext * 0.8)

        for _ in range(n):
            self._ctrl.gotoState(retract_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(extend_state, self._rot_speed, self._tra_speed)
        
    def _rotateArm(self, state, n):
        rot_angle = pi / 6
        state.set_arm_rot(max(state.arm_rot, rot_angle))
        
        ccw_state = state.copy().set_arm_rot(state.arm_rot + rot_angle)
        cw_state = state.copy().set_arm_rot(state.arm_rot - rot_angle)
        
        for _ in range(n):
            self._ctrl.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(cw_state, self._rot_speed, self._tra_speed)            
            
    def _rotateWrist(self, state, n):
        rot_angle = pi / 2
        
        neutral_angle = pi / 2 + DexRobotZeke.THETA
        ccw_state = state.copy().set_gripper_rot(neutral_angle + rot_angle)
        cw_state = state.copy().set_gripper_rot(neutral_angle - rot_angle)
        
        for _ in range(n):
            self._ctrl.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(state, self._rot_speed, self._tra_speed)           
            self._ctrl.gotoState(cw_state, self._rot_speed, self._tra_speed)            
            self._ctrl.gotoState(state, self._rot_speed, self._tra_speed)
                   
    def _wristArmCombo(self, state, n):  
        original_angle = state.gripper_rot
        
        state.set_gripper_rot(pi + DexRobotZeke.THETA)
        self._ctrl.gotoState(state)
        
        self._rotateArm(state, n)
        
        state.set_gripper_rot(original_angle)
        self._ctrl.gotoState(state)
                   
    def plot(self):
        self._ctrl.plot()
            
    def testGrip(self, target_pose, n):
        self._ctrl.reset()
        self._ctrl.unGrip()
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        self._ctrl.transform(target_pose, "Initial Grasp State")
        self._ctrl.grip()
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        local_target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose
        neutral_state = DexRobotZeke.pose_to_state(local_target_pose, self._ctrl.getState().set_gripper_grip(DexConstants.MIN_STATE.gripper_grip))
        neutral_state.set_arm_elev(DexConstants.MAX_ELEVATION * 0.5)
        
        self._ctrl.gotoState(neutral_state)
        self._moveArm(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl.gotoState(neutral_state)
        self._rotateArm(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl.gotoState(neutral_state)
        self._rotateWrist(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl.gotoState(neutral_state)
        self._wristArmCombo(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl.transform(target_pose, "Initial Grasp State")    
                

raised = pose((0.1, 0, 0.13), DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)
local_raised = DexRobotZeke.ZEKE_LOCAL_T * raised
'''
def test(s, n):
    Logger.start()
    t = DexGripTester(s, s)
    t.testGrip(raised, n)
    t.plot()
    t._ctrl.stop()
s'''