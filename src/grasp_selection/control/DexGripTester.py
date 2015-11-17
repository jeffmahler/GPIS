from DexConstants import DexConstants
from DexController import DexController
from tfx import point, pose, rotation
from ZekeState import ZekeState
from DexRobotZeke import DexRobotZeke

class DexGripTester:

    def __init__(self, robot = None, comm = DexConstants.COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT, tra_speed = 1, rot_speed = 1):
        self._ctrl = DexController(robot, comm, baudrate, timeout)
        self.set_tra_speed(tra_speed)
        self.set_rot_speed(rot_speed)
        
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
            self._ctrl.transform(pose_hi)
            self._ctrl.transform(pose_lo)

    def _moveArm(self, target_pose, n):
        state = DexRobotZeke.pose_to_state(target_pose, self._ctrl.getState())
        state.set_arm_elev(DexConstants.MAX_ELEVATION * 0.5)
        
        retract_state = state.copy().set_arm_ext(DexConstants.MIN_STATE.arm_ext)
        extend_state = state.copy().set_arm_ext(DexConstants.MAX_STATE.arm_ext * 0.5)
        
        for _ in range(n):
            self._ctrl.gotoState(retract_state, self._rot_speed, self._tra_speed)
            self._ctrl.gotoState(extend_state, self._rot_speed, self._tra_speed)
                   
    def plot(self):
        self._ctrl.plot()
            
    def testGrip(self, target_pose, n):
        self._ctrl.reset()
        self._ctrl.unGrip()
        self._ctrl.transform(target_pose)
        self._ctrl.grip()
        #self._moveHeight(target_pose, n)
        self._moveArm(target_pose, n)
        '''
        self._rotateArm(pose)
        self._rotateWrist(pose)
        '''
        
raised = pose((0, 0, 0.15), rotation.identity(), frame = DexConstants.WORLD_FRAME)