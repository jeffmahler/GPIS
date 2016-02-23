from tfx import transform, vector, rotation
from DexConstants import DexConstants
from DexSerial import DexSerialInterface
from DexNumericSolvers import DexNumericSolvers
from ZekeState import ZekeState
from Logger import Logger
from math import sqrt
from numpy import pi, arctan, cos, sin
from numpy.linalg import norm
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import IPython
import logging
import numpy as np

class DexRobotZeke:
    '''
    Abstraction for a robot profile for Zeke
    '''
    
    RESET_STATES = {"GRIPPER_SAFE_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.1, 0.02, ZekeState.THETA + pi, 0.036]),
                    "GRIPPER_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.05, 0.02, ZekeState.THETA + pi, None]),
                    "OBJECT_RESET" : ZekeState([3 * pi / 2 + ZekeState.PHI, 0.01, 0.02, ZekeState.THETA + pi, None]),
                    "ZEKE_RESET_SHUTTER_FREE" : ZekeState([None, 0.01, None, None, None]), 
                    "ZEKE_RESET" : ZekeState([None, None, 0.01, None, None]),
                    "ZEKE_RESET_CLEAR_TABLE" : ZekeState([3 * pi /2 + + ZekeState.PHI, None, None, None, None])}
    
    ZEKE_LOCAL_T = transform(
                            vector(-ZekeState.ZEKE_ARM_ORIGIN_OFFSET, 0, 0),
                            rotation.identity(), 
                            parent=DexConstants.ZEKE_LOCAL_FRAME,
                            frame=DexConstants.WORLD_FRAME)

    def __init__(self, comm = DexConstants.ZEKE_COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._ser_int= DexSerialInterface(ZekeState, comm, baudrate, timeout, read_sensors=True)      
        self._ser_int.start()
        self._target_state = self.getState()
        Logger.clear(ZekeState.NAME)
    
    def reset(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        #self.gotoState(DexRobotZeke.RESET_STATES["GRIPPER_SAFE_RESET"], rot_speed, tra_speed, "Reset Gripper Safe")
        self.gotoState(DexRobotZeke.RESET_STATES["GRIPPER_RESET"], rot_speed, tra_speed, "Gripper Reset")
        #self.gotoState(DexRobotZeke.RESET_STATES["ZEKE_RESET_SHUTTER_FREE"], rot_speed, tra_speed, "Reset Shutter Free")
        #self.gotoState(DexRobotZeke.RESET_STATES["ZEKE_RESET"], rot_speed, tra_speed, "Reset Complete")
            
    def reset_clear_table(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.reset(rot_speed, tra_speed)
        #self.gotoState(DexRobotZeke.RESET_STATES["ZEKE_RESET_CLEAR_TABLE"], rot_speed, tra_speed, "Reset Clear Table")

    def reset_object(self, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self.gotoState(DexRobotZeke.RESET_STATES["OBJECT_RESET"], rot_speed, tra_speed, "Reset Object")
            
    def stop(self):
        self._ser_int.stop()
        
    def getState(self):
        return self._ser_int.getState()       

    def getSensors(self):
        sensor_vals = self._ser_int.getSensors()       
        if isinstance(sensor_vals, str):
            return None
        return sensor_vals
        
    def grip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        state = self._target_state.copy()
        state.set_gripper_grip(ZekeState.MIN_STATE().gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed, "Gripping", block=False)
        
        sensor_vals = self.getSensors()
        current_state = self.getState()
        duration = 0.0
        while (sensor_vals is None or sensor_vals.gripper_force < DexConstants.GRIPPER_CLOSE_FORCE_THRESH) and \
                current_state.gripper_grip > ZekeState.MIN_STATE().gripper_grip + DexConstants.GRIPPER_CLOSE_EPS and \
                duration < DexConstants.GRIPPER_CLOSE_TIMEOUT:
            sleep(0.1)
            sensor_vals = self.getSensors()
            current_state = self.getState()
            duration = duration + 0.1

        print sensor_vals
        print 'Duration', duration

        self.gotoState(current_state, DexConstants.DEFAULT_ROT_SPEED, tra_speed, "Gripping")
        
    def unGrip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        state = self._target_state.copy()
        state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed, "Ungripping")
                    
    @staticmethod
    def pose_to_state(target_pose, prev_state, angles = None):
        '''
        Takes in a pose w/ respect to zeke and returns the state using IK
        '''
        if target_pose.frame is not DexConstants.ZEKE_LOCAL_FRAME:
            raise Exception("Given target_pose is not in ZEKE LOCAL frame")
                
        # calculate finger position and arm rotation about z axis
        x = target_pose.position.x
        y = target_pose.position.y
        theta = DexNumericSolvers.get_cartesian_angle(x, y)

        logging.debug('Computed robot pose wrt robot base frame: X = %f, Y = %f, Theta = %f' %(x, y, theta))

        # get wrist rotation and compute offset from fingers to wrist
        if angles is None:
            raise ValueError('Use of deprecated tb angles')
        else:
            # TODO: update Zeke with kinematics - this angle should really be computed from the pose in Zeke itself, NOT in the controller
            psi = angles.psi
            if psi < pi/2:
                psi = psi + pi
            elif psi > 3*pi/2:
                psi = psi - pi
            
        delta_h_wrist_fingers = ZekeState.WRIST_TO_FINGER_RADIUS * np.cos(psi) # the differential height of the fingers due to the wrist rotation
        delta_xy_wrist_fingers = ZekeState.WRIST_TO_FINGER_RADIUS * -np.sin(psi) # the differential xy pos of the fingers due to the wrist rotation
        
        # compute the x and y pos of the wrist to get the fingers to the desired position
        finger_r = np.array([x, y])
        finger_r_hat = finger_r / np.linalg.norm(finger_r)
        finger_orth_dir = np.array([-finger_r_hat[1], finger_r_hat[0]])
        delta_r = delta_xy_wrist_fingers * finger_orth_dir
        wrist_r = finger_r - delta_r
        x = wrist_r[0]
        y = wrist_r[1]

        # snap theta to valid range
        target_theta = theta + ZekeState.PHI
        if target_theta > 2 * np.pi:
            logging.info('Decreasing computed rotation') 
            target_theta = target_theta - 2 * np.pi

        # snap grasp height to valid range
        target_elev = target_pose.position.z - ZekeState.DELTA_Z - delta_h_wrist_fingers
        if target_elev < 0.0:
            logging.info('Snapping to valid height') 
            target_elev = 0.0
        
        print 'X', x
        print 'Y', y
        print 'Z', target_elev
        print 'Pose X', target_pose.position.x
        print 'Pose Y', target_pose.position.x
        print 'Pose Z', target_pose.position.z
        print 'Delta h', delta_h_wrist_fingers
        print 'Psi', psi

        state = ZekeState()
        state.set_arm_rot(target_theta)
        state.set_arm_elev(target_elev)
        state.set_arm_ext(norm(wrist_r) - ZekeState.ZEKE_ARM_TO_GRIPPER_TIP_LENGTH)

        #ANGLES using pitch
        state.set_gripper_rot(psi + ZekeState.THETA)
        state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)

        return state
        
    def gotoState(self, target_state, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED,
                  name = None, block=True):
        def _boundGripperRot(rot):
            if rot is None:
                return None
            if abs(rot - ZekeState.MAX_STATE().gripper_rot) <= 0.01:
                return ZekeState.MAX_STATE().gripper_rot
            if abs(rot - ZekeState.MIN_STATE().gripper_rot) <= 0.01:
                return ZekeState.MIN_STATE().gripper_rot
            if rot > ZekeState.MAX_STATE().gripper_rot:
                return ZekeState.MAX_STATE().gripper_rot
            if rot < ZekeState.MIN_STATE().gripper_rot:
                return ZekeState.MIN_STATE().gripper_rot
            return rot
                
        target_state.set_gripper_rot(_boundGripperRot(target_state.gripper_rot))
        self._ser_int.gotoState(target_state, rot_speed, tra_speed, name, block=block)
                
        self._target_state = target_state.copy()

    def transform(self, target_pose, name, angles = None, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose

        #ANGLES using roll
        if angles is None:
            gamma = target_pose.rotation.tb_angles.roll_rad
        else:
            gamma = angles.roll
        if abs(gamma) >= DexConstants.ROLL_THRESH:
            raise Exception("Can't perform rotation about x-axis on Zeke's gripper")
            
        target_state = DexRobotZeke.pose_to_state(target_pose, self._target_state, angles)
        
        self.gotoState(target_state, rot_speed, tra_speed, name)
        
    def transform_aim_extend_grip(self, target_pose, name, angles = None, rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED, sleep_val=1.0):
        target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose

        #ANGLES using roll
        if angles is None:
            gamma = target_pose.rotation.tb_angles.roll_rad
        else:
            gamma = angles.roll
        if abs(gamma) >= DexConstants.ROLL_THRESH:
            raise Exception("Can't perform rotation about x-axis on Zeke's gripper: "  + str(target_pose.rotation.euler))
            
        target_state = DexRobotZeke.pose_to_state(target_pose, self._target_state, angles)

        logging.info('Target state: %s' %(str(target_state)))

        aim_state = target_state.copy().set_arm_ext(ZekeState.MIN_STATE().arm_ext)
        aim_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
        
        #logging.info('Opening grippers')
        #self.unGrip()

        logging.info('Aiming Zeke toward goal state: %s' %(str(aim_state)))
        self.gotoState(aim_state, rot_speed, tra_speed, name + "_aim")

        logging.info('Moving gripper to goal state: %s' %(str(target_state)))
        self.gotoState(target_state, rot_speed, tra_speed, name + "_grasp")

        logging.info('Closing grippers')
        self.grip()

        return target_state
        
    def _state_FK(self, state):
        arm_angle = state.arm_rot - ZekeState.PHI
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
