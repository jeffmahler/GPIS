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
            
    def _moveHeight(self, target_pose, n):
        x = target_pose.position[0]
        y = target_pose.position[1]
        z = target_pose.position[2]
    
        hi = point(x, y, DexConstants.MAX_ELEVATION * 0.9)
        pose_hi = pose(hi, rotation.identity(), frame=DexConstants.WORLD_FRAME)
        
        lo = point(x, y, z)
        pose_lo = pose(lo, rotation.identity(), frame=DexConstants.WORLD_FRAME)
        
        for _ in range(n):
            self._ctrl.transform(pose_hi, self._rot_speed, self._tra_speed)
            self._ctrl.transform(pose_lo, self._rot_speed, self._tra_speed)

    def _moveArm(self, target_pose, n):
        state = DexRobotZeke.pose_to_state(target_pose, self._ctrl.getState().set_gripper_grip(DexConstants.MIN_STATE.gripper_grip))
        state.set_arm_elev(DexConstants.MAX_ELEVATION * 0.5)
        
        retract_state = state.copy().set_arm_ext(DexConstants.MIN_STATE.arm_ext + 0.05)
        extend_state = state.copy().set_arm_ext(DexConstants.MAX_STATE.arm_ext * 0.8)

        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        self._ctrl.gotoState(state)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        for _ in range(n):
            self._ctrl.gotoState(retract_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(extend_state, self._rot_speed, self._tra_speed)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
    def _rotateArm(self, target_pose, n):
        rot_angle = pi / 6
        state = DexRobotZeke.pose_to_state(target_pose, self._ctrl.getState().set_gripper_grip(DexConstants.MIN_STATE.gripper_grip))
        state.set_arm_rot(max(state.arm_rot, rot_angle))
        state.set_arm_elev(DexConstants.MAX_STATE.arm_elev * 0.5)
        
        ccw_state = state.copy().set_arm_rot(state.arm_rot + rot_angle)
        cw_state = state.copy().set_arm_rot(state.arm_rot - rot_angle)
        
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        self._ctrl.gotoState(state)
        for _ in range(n):
            self._ctrl.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(cw_state, self._rot_speed, self._tra_speed)            
        self._ctrl.gotoState(state, self._rot_speed, self._tra_speed)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
            
    def _rotateWrist(self, target_pose, n):
        rot_angle = pi / 4
        state = DexRobotZeke.pose_to_state(target_pose, self._ctrl.getState().set_gripper_grip(DexConstants.MIN_STATE.gripper_grip))
        state.set_arm_elev(DexConstants.MAX_STATE.arm_elev * 0.5)
        
        original_angle = state.gripper_rot
        ccw_state = state.copy().set_gripper_rot(original_angle + rot_angle)
        cw_state = state.copy().set_gripper_rot(original_angle - rot_angle)
        
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        self._ctrl.gotoState(state)
        for _ in range(n):
            self._ctrl.gotoState(ccw_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(state, self._rot_speed, self._tra_speed)           
            self._ctrl.gotoState(cw_state, self._rot_speed, self._tra_speed)            
            self._ctrl.gotoState(state, self._rot_speed, self._tra_speed)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
                   
    def plot(self):
        self._ctrl.plot()
            
    def testGrip(self, target_pose, n):
        self._ctrl.reset()
        self._ctrl.unGrip()
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        self._ctrl.transform(target_pose, "Initial Grasp State")
        self._ctrl.grip()
        local_target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose
        
        self._moveArm(local_target_pose, n)
        self._ctrl.transform(target_pose)
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._rotateArm(local_target_pose, n)
        self._ctrl.transform(target_pose, "Initial Grasp State")
        self._ctrl.pause(DexConstants.PAUSE_DELAY)
        
        self._rotateWrist(local_target_pose, n)
        self._ctrl.transform(target_pose, "Initial Grasp State")    
                
        
raised = pose((0.1, 0, 0.04), rotation.identity(), frame = DexConstants.WORLD_FRAME)
local_raised = DexRobotZeke.ZEKE_LOCAL_T * raised

def test(s, n):
    Logger.start()
    t = DexGripTester(s, s)
    t.testGrip(raised, n)
    t.plot()
    t._ctrl.stop()