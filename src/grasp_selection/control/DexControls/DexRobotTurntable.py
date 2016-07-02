from DexConstants import DexConstants
from DexSerial import DexSerialInterface
from TurntableState import TurntableState
from DexNumericSolvers import DexNumericSolvers
from Logger import Logger
from numpy import pi, cos, sin, mean, std, abs
from numpy.random import uniform
from numpy.linalg import norm
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import logging

import IPython

class DexRobotTurntable:
    '''
    Abstraction for a robot profile. Contains all information specific
    to the Turntable robot
    '''
    
    #only for visualization purposes.
    _RADIUS = 5
    
    RESET_STATE = TurntableState([TurntableState.THETA])

    def __init__(self, comm = DexConstants.TABLE_COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._turntable= DexSerialInterface(TurntableState, comm, baudrate, timeout)      
        self._turntable.start()
        self._target_state = self.getState()
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
        
    def gotoState(self, target_state, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED, name = None, block=True):
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

def test_turntable(n=20):
    t = DexRobotTurntable()
    sleep(0.5)
    mid = t.getState()

    targets = []
    actuals = []
    diffs = []
    for i in range(n):
        target = mid.copy().set_table_rot(mid.table_rot + uniform(-1, 1))
        t.gotoState(target)
        while not t.is_action_complete():
            continue
        sleep(0.2)
        actual = t.getState()
        
        target_val = target.table_rot
        actual_val = actual.table_rot
        diff_val = actual_val - target_val

        targets.append(target_val)
        actuals.append(actual_val)
        diffs.append(diff_val)

        print 'Target {0}. Actual {1}. Diff {2}'.format(target_val, actual_val, diff_val)

    mean_abs_diff = mean(abs(diffs))
    std_abs_diff = std(abs(diffs))

    print 'Mean abs diff {0}. Std abs diff {1}'.format(mean_abs_diff, std_abs_diff)

    t.stop()

if __name__ == '__main__':
    test = False
    if test:
        test_turntable()
