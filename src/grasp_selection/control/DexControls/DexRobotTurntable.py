from DexConstants import DexConstants
from DexSerial import DexSerialInterface
from TurntableState import TurntableState
from Logger import Logger
from numpy import pi, cos, sin
from numpy.linalg import norm
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class DexRobotTurntable:
    '''
    Abstraction for a robot profile. Contains all information specific
    to the Turntable robot
    '''

    #actual angle = desired angle + OFFSET
    THETA = 0.4819 #turntable rotation 0 degree offset.
    _RADIUS = 5
    
    RESET_STATE = TurntableState([THETA])

    def __init__(self, comm = DexConstants.TABLE_COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._turntable= DexSerialInterface(TurntableState, comm, baudrate, timeout)      
        self._turntable.start()
        self._target_state = self.getState()
        Logger.clear(TurntableState.NAME)
    
    def reset(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.gotoState(DexRobotTurntable.RESET_STATE, rot_speed, tra_speed, "Reset Turntable")
            
    def stop(self):
        self._turntable.stop()
        
    def getState(self):
        return self._turntable.getState()
        
    def gotoState(self, target_state, rot_speed, tra_speed, name = None):
        def _boundTurntableRot(rot):
            if rot is None:
                return None
            if abs(rot - TurntableState.MAX_STATE().table_rot) <= 0.01:
                return TurntableState.MAX_STATE().table_rot
            if abs(rot - TurntableState.MIN_STATE().table_rot) <= 0.01:
                return TurntableState.MIN_STATE().table_rot
            if rot > TurntableState.MAX_STATE().table_rot:
                return TurntableState.MAX_STATE().table_rot
            if rot < TurntableState.MIN_STATE().table_rot:
                return TurntableState.MIN_STATE().table_rot
            return rot
                
        target_state.set_table_rot(_boundTurntableRot(target_state.table_rot))
        self._turntable.gotoState(target_state, rot_speed, tra_speed, name)
        self._target_state = target_state.copy()

    @staticmethod
    def pose_to_state(target_pose, angles = None):
        #ANGLES using yaw
        if angles is None:
            phi = target_pose.rotation.tb_angles.yaw_rad
        else:
            phi = angles.yaw
        return TurntableState().set_table_rot(phi + DexRobotTurntable.THETA)
        
    def _state_FK(self, state):
        return (DexRobotTurntable._RADIUS * cos(state.table_rot), DexRobotTurntable._RADIUS * sin(state.table_rot), 0)
        
    def transform(self, target_pose, name, angles = None, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        target_state = DexRobotTurntable.pose_to_state(target_pose, angles)
        
        self.gotoState(target_state, rot_speed, tra_speed, name)
        
    def maintainState(self, s):
        self._turntable.maintainState(s)
        
    def is_action_complete(self):
        return self._turntable.is_action_complete()
        
    def plot(self):
        hist = self._turntable.state_hist
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x = []
        y = []
        z = []
        
        for state in hist:
            pos = self._state_FK(state)
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        
        ax.plot(x, y, z, c="g", marker="o")

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.show()