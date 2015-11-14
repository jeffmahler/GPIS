from DexConstants import DexConstants
from DexController import DexController
from tfx import point, pose, rotation
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
            
    def _moveHeight(self, pose, n):
        x = pose.position[0]
        y = pose.position[1]
        z = pose.position[2]
    
        hi = point(x, y, DexConstants.MAX_ELEVATION * 0.9)
        pose_hi = pose(hi, rotation.identity(), frame=DexConstants.WORLD_FRAME)
        
        lo = point(x, y, z)
        pose_lo = pose(lo, rotation.identity(), frame=DexConstants.WORLD_FRAME)
        
        for _ in range(n):
            self._ctrl.transform(pose_lo)
            self._ctrl.transform(pose_hi)
            
    def testGrip(self, pose, n):
        self._ctrl.transform(pose)
        self._ctrl.grip()
        self._moveHeight(pose)
        '''
        self._moveArm(pose)
        self._rotateArm(pose)
        self._rotateWrist(pose)
        '''