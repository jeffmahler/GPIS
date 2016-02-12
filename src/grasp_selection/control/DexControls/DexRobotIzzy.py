from tfx import transform, vector, rotation
from DexConstants import DexConstants
from DexSerial import DexSerialInterface
from DexNumericSolvers import DexNumericSolvers
from IzzyState import IzzyState
from Logger import Logger
from math import sqrt
from numpy import pi, arctan, cos, sin
from numpy.linalg import norm
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import logging
import numpy as np

class DexRobotIzzy:
    '''
    Abstraction for a robot profile for Izzy.
    '''
    
    RESET_STATES = {"GRIPPER_SAFE_RESET" : IzzyState([IzzyState.PHI, 0.00556, 0.0185, None, 0.01]),
                    #TODO: change wrist rot to IzzyState.THETA + pi/2 when gripper wrist is working
                    "GRIPPER_RESET" : IzzyState([None, None, None, None, None]),
                    "OBJECT_RESET" : IzzyState([-pi / 2 + IzzyState.PHI, 0.1, 0.0, None, None]),
                    "IZZY_RESET_SHUTTER_FREE" : IzzyState([None, 0.00556, None, None, None]), 
                    "IZZY_RESET" : IzzyState([None, None, 0.0185, None, None]),
                    "IZZY_RESET_CLEAR_TABLE" : IzzyState([-pi/2 + IzzyState.PHI, None, None, None, None])}
    
    IZZY_LOCAL_T = transform(
                                            vector(-IzzyState.IZZY_ARM_ORIGIN_OFFSET, 0, 0),
                                            rotation.identity(), 
                                            parent=DexConstants.IZZY_LOCAL_FRAME,
                                            frame=DexConstants.WORLD_FRAME)

    def __init__(self, comm = DexConstants.IZZY_COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._ser_int = DexSerialInterface(IzzyState, comm, baudrate, timeout)      
        self._ser_int.start()
        self._target_state = self.getState()
        Logger.clear(IzzyState.NAME)
    
    def reset(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.gotoState(DexRobotIzzy.RESET_STATES["GRIPPER_SAFE_RESET"], rot_speed, tra_speed, "Reset Gripper Safe")
            
    def reset_clear_table(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.reset(rot_speed, tra_speed)
            
    def reset_object(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.gotoState(DexRobotIzzy.RESET_STATES["OBJECT_RESET"], rot_speed, tra_speed, "Reset Object")

    def stop(self):
        self._ser_int.stop()
        
    def getState(self):
        return self._ser_int.getState()       
        
    def grip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        state = self._target_state.copy()
        state.set_gripper_grip(IzzyState.MIN_STATE().gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed, "Gripping")
        
    def unGrip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        state = self._target_state.copy()
        state.set_gripper_grip(IzzyState.MAX_STATE().gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed, "Ungripping")
                    
    @staticmethod
    def pose_to_state(target_pose, prev_state, angles = None):
        '''
        Takes in a pose w/ respect to izzy and returns the state using IK
        '''
        if target_pose.frame is not DexConstants.IZZY_LOCAL_FRAME:
            raise Exception("Given target_pose is not in IZZY LOCAL frame")
                
        #calculate rotation about z axis
        x = target_pose.position.x
        y = target_pose.position.y
        theta = DexNumericSolvers.get_cartesian_angle(x, y)
        
        logging.debug('Computed robot pose wrt robot base frame: X = %f, Y = %f, Theta = %f' %(x, y, theta))

        # snap theta to valid range
        target_theta = theta + IzzyState.PHI
        if target_theta > 2 * np.pi:
            logging.info('Decreasing computed rotation') 
            target_theta = target_theta - 2 * np.pi

        # snap grasp to valid range
        target_elev = target_pose.position.z - IzzyState.DELTA_Z
        if target_elev < 0.0:
            logging.info('Snapping to valid height') 
            target_elev = 0.0

        state = IzzyState()
        state.set_arm_rot(target_theta)
        state.set_arm_elev(target_elev)
        state.set_arm_ext(norm([x, y]) - IzzyState.IZZY_ARM_TO_GRIPPER_TIP_LENGTH)

        #ANGLES using pitch
        if angles is None:
            psi = target_pose.rotation.tb_angles.pitch_rad
        else:
            psi = angles.pitch
        state.set_gripper_rot(psi + IzzyState.THETA)
        state.set_gripper_grip(prev_state.gripper_grip)
        
        return state
        
    def gotoState(self, target_state, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED, name = None):

        #TODO: Retest after gripper wrist working
        def _boundGripperRot(rot):
            if rot is None:
                return None
            if abs(rot - IzzyState.MAX_STATE().gripper_rot) <= 0.01:
                return IzzyState.MAX_STATE().gripper_rot
            if abs(rot - IzzyState.MIN_STATE().gripper_rot) <= 0.01:
                return IzzyState.MIN_STATE().gripper_rot
            if rot > IzzyState.MAX_STATE().gripper_rot:
                return IzzyState.MAX_STATE().gripper_rot
            if rot < IzzyState.MIN_STATE().gripper_rot:
                return IzzyState.MIN_STATE().gripper_rot
            return rot
                
        #TODO: need bound to take into account of offsets
        def _boundArmRot(rot):
            if rot is None:
                return None
            return rot
                
        target_state.set_gripper_rot(target_state.gripper_rot)
        target_state.set_arm_rot(_boundArmRot(target_state.arm_rot))
        self._ser_int.gotoState(target_state, rot_speed, tra_speed, name)

        self._target_state = target_state.copy()

    def transform(self, target_pose, name, angles = None, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        target_pose = DexRobotIzzy.IZZY_LOCAL_T * target_pose

        #ANGLES using roll
        if angles is None:
            gamma = target_pose.rotation.tb_angles.roll_rad
        else:
            gamma = angles.roll
        if abs(gamma) >= DexConstants.ROLL_THRESH:
            raise Exception("Can't perform rotation about x-axis on Izzy's gripper")
            
        target_state = DexRobotIzzy.pose_to_state(target_pose, self._target_state, angles)
        
        self.gotoState(target_state, rot_speed, tra_speed, name)
        
    def transform_aim_extend_grip(self, target_pose, name, angles = None, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        target_pose = DexRobotIzzy.IZZY_LOCAL_T * target_pose

        #ANGLES using roll
        if angles is None:
            gamma = target_pose.rotation.tb_angles.roll_rad
        else:
            gamma = angles.roll
        if abs(gamma) >= DexConstants.ROLL_THRESH:
            raise Exception("Can't perform rotation about x-axis on Izzy's gripper: "  + str(target_pose.rotation.euler))
            
        target_state = DexRobotIzzy.pose_to_state(target_pose, self._target_state, angles)
        target_state.set_gripper_grip(IzzyState.MAX_STATE().gripper_grip)

        logging.info('Izzy goal gripper state:  %s' %(str(target_state)))
        aim_state = target_state.copy().set_arm_ext(IzzyState.MIN_STATE().arm_ext)
        
        logging.info('Opening grippers')
        self.unGrip()
        while not self.is_action_complete():
            sleep(0.01)
        sleep(2)

        logging.info('Aiming Izzy toward goal state')
        self.gotoState(aim_state, rot_speed, tra_speed, name + "_aim")
        while not self.is_action_complete():
            sleep(0.01)
        sleep(2)

        logging.info('Moving gripper to goal state')
        self.gotoState(target_state, rot_speed, tra_speed, name + "_grasp")

        while not self.is_action_complete():
            sleep(0.01)
        sleep(2)
            
        logging.info('Closing grippers')
        self.grip()
        while not self.is_action_complete():
            sleep(0.01)
        sleep(2)
        
    def _state_FK(self, state):
        arm_angle = state.arm_rot - IzzyState.PHI
        z = state.arm_elev
        r = state.arm_ext
        x = r * cos(arm_angle)
        y = r * sin(arm_angle)
        
        return (x, y, z)
        
    def maintainState(self, s):
        self._ser_int.maintainState(s)
        
    def is_action_complete(self):
        return self._ser_int.is_action_complete()
        
    def plot(self):
        hist = self._ser_int.state_hist
        
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
