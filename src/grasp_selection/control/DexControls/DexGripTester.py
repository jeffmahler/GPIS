from tfx import point, pose, rotation, rotation_tb
from numpy import pi
from DexConstants import DexConstants
from DexController import DexController
from ZekeState import ZekeState
from TurntableState import TurntableState
from DexRobotZeke import DexRobotZeke
from Logger import Logger

class DexGripTester:

    def __init__(self, tra_mult = 0.7, rot_mult = 0.7):
        self._ctrl = DexController()
        self.set_tra_speed(tra_mult)
        self.set_rot_speed(rot_mult)
        
    def set_tra_speed(self, val):
        val = min(abs(val), 1)
        self._tra_speed = val * DexConstants.MAX_TRA_SPEED        

    def set_rot_speed(self, val):
        val = min(abs(val), 1)        
        self._rot_speed = val * DexConstants.MAX_ROT_SPEED

    def _moveArm(self, state, n):        
        retract_state = state.copy().set_arm_ext(ZekeState.MIN_STATE().arm_ext + 0.05)
        extend_state = state.copy().set_arm_ext(ZekeState.MAX_STATE().arm_ext * 0.8)

        for _ in range(n):
            self._ctrl._zeke.gotoState(retract_state, self._rot_speed, self._tra_speed)
            self._ctrl._zeke.gotoState(extend_state, self._rot_speed, self._tra_speed)
        
    def _rotateArm(self, state, n):
        rot_angle = pi / 6
        state.set_arm_rot(max(state.arm_rot, rot_angle))
        
        ccw_state = state.copy().set_arm_rot(state.arm_rot + rot_angle)
        cw_state = state.copy().set_arm_rot(state.arm_rot - rot_angle)
        
        for _ in range(n):
            self._ctrl._zeke.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl._zeke.gotoState(cw_state, self._rot_speed, self._tra_speed)            
            
    def _rotateWrist(self, state, n):
        rot_angle = pi / 2
        
        neutral_angle = pi / 2 + DexRobotZeke.THETA
        ccw_state = state.copy().set_gripper_rot(neutral_angle + rot_angle)
        cw_state = state.copy().set_gripper_rot(neutral_angle - rot_angle)
        
        for _ in range(n):
            self._ctrl._zeke.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl._zeke.gotoState(state, self._rot_speed, self._tra_speed)           
            self._ctrl._zeke.gotoState(cw_state, self._rot_speed, self._tra_speed)            
            self._ctrl._zeke.gotoState(state, self._rot_speed, self._tra_speed)
                   
    def _wristArmCombo(self, state, n):  
        original_angle = state.gripper_rot
        
        state.set_gripper_rot(pi + DexRobotZeke.THETA)
        self._ctrl._zeke.gotoState(state)
        
        self._rotateArm(state, n)
        
        state.set_gripper_rot(original_angle)
        self._ctrl._zeke.gotoState(state)
                   
    def plot(self):
        self._ctrl.plot()
            
    def testGrip(self, target_pose_unprocessed, n):
        self._ctrl._table.reset()
        target_pose = self._ctrl.do_grasp(target_pose_unprocessed)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        local_target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose
        neutral_state = DexRobotZeke.pose_to_state(local_target_pose, self._ctrl.getState()[0].set_gripper_grip(ZekeState.MIN_STATE().gripper_grip))
        neutral_state.set_arm_elev(DexConstants.MAX_ELEVATION * 0.5)
        neutral_state.set_arm_rot(pi + DexRobotZeke.PHI)
        
        self._ctrl._zeke.gotoState(neutral_state)
        self._moveArm(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl._zeke.gotoState(neutral_state)
        self._rotateArm(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl._zeke.gotoState(neutral_state)
        self._rotateWrist(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl._zeke.gotoState(neutral_state)
        self._wristArmCombo(neutral_state.copy(), n)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._ctrl._zeke.transform(target_pose, "Initial Grasp State")    
                
raised = pose((0.1, 0, 0.13), DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)
local_raised = DexRobotZeke.ZEKE_LOCAL_T * raised

def test(s, n, phi = 45):
    target = pose((0.05, 0.05, 0.05), rotation_tb(phi, 90, 0), frame = DexConstants.WORLD_FRAME)
    t = DexGripTester(s, s)
    t.testGrip(target, n)
    t._ctrl.plot_approach_angle()
    t._ctrl._zeke.plot()
    t._ctrl.stop()