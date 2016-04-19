from DexConstants import DexConstants
from DexSerial import DexSerialInterface
from TurntableState import TurntableState
from DexNumericSolvers import DexNumericSolvers
from Logger import Logger
from numpy import pi, cos, sin
from numpy.linalg import norm
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import logging

class DexRobotTurntable:
    '''
    Abstraction for a robot profile. Contains all information specific
    to the Turntable robot
    '''
    
    #only for visualization purposes.
    _RADIUS = 5
    
    RESET_STATE = TurntableState([pi + TurntableState.THETA])

    def __init__(self, comm = DexConstants.TABLE_COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._turntable= DexSerialInterface(TurntableState, comm, baudrate, timeout)      
        self._turntable.start()
        self._target_state = self.getState()
        print self._target_state
        Logger.clear(TurntableState.NAME)
    
    def reset(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.gotoState(DexRobotTurntable.RESET_STATE, rot_speed, tra_speed, "Reset Turntable", block=False)
            
    def stop(self):
        self._turntable.stop()
        
    def getState(self):
        return self._turntable.getState()

    def reset_fishing(self):
        self._turntable.queueArbitraryRequest('r')
        sleep(DexConstants.RESET_FISHING_DELAY)
        
    def reset_fishing(self):
        self._turntable.queueArbitraryRequest('r')
        sleep(DexConstants.RESET_FISHING_DELAY)
        
    def gotoState(self, target_state, rot_speed, tra_speed, name = None, block=True):
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
        self._turntable.gotoState(target_state, rot_speed, tra_speed, name, block=block)
        self._target_state = target_state.copy()
 
    def _state_FK(self, state):
        return (DexRobotTurntable._RADIUS * cos(state.table_rot), DexRobotTurntable._RADIUS * sin(state.table_rot), 0)
         
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
