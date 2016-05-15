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
from pneumaticbox import io, api

import IPython
import logging
import numpy as np

class DexRobotZeke:
    '''
    Abstraction for a robot profile for Zeke
    '''
    PNEUMATIC_BOX_ADDRESS = 'beagle12.local'
    SOFTHAND_SYNERGY = [0.4, 0.6, 0.2, 0.8, 0.3, 0.2]
    
    RESET_STATES = {"GRIPPER_SAFE_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.1, 0.02, ZekeState.THETA + 3 * pi/2, 0.036]),
                    "GRIPPER_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.05, 0.02, ZekeState.THETA + 3 * pi/2, None]),
                    "OBJECT_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.1, 0.02, ZekeState.THETA + 3 * pi/2, 0.065]),
                    #"OBJECT_RESET" : ZekeState([np.pi + ZekeState.PHI, 0.25, 0.02, ZekeState.THETA + pi, 0.00]),
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
        self._setup_channels()
              
        Logger.clear(ZekeState.NAME)
    
    def _setup_channels(self):
        # init variables
        self.airserver = io.Airserver(TCP_IP=DexRobotZeke.PNEUMATIC_BOX_ADDRESS,
                                      doReset=False)
        self.channels = np.array([2, 3, 4, 5, 6, 7]) # max partition, two thumb chambers are connected
        self.sensors  = np.array([0, 1, 2, 3, 4, 5])
        
        assert(self.channels.size == self.sensors.size)
        
        self.watchdog_levels = np.array([90, 90, 85, 90, 90, 90, 90, 90])
        self.limiter_levels = np.array([60, 60, 55, 60, 60, 60, 60, 60])-10 #offset by 10kPa as the valves take some time to turn off
        
        # Configure all channels for Threshold controller
        for i in self.channels:
            self.airserver.submit(api.MsgConfigurationControllerThreshold(i))
        
        # Configure pressure watchdogs for all channels, additionally configure pressure limiter that only limit channel pressure
        for i, s in zip(self.channels, self.sensors):
            msg_watchdog = api.MsgConfigurationWatchdogPressure(i, id_offset = 20, max_pressure=self.watchdog_levels[i])
            msg_limiter = api.MsgConfigurationControllerPressureLimiter(i, id_offset = 60, limit=self.limiter_levels[i])

            msg_limiter.signal_to_watch = api.BLOCK_SIGNALS_SENSOR + s
            msg_watchdog.signal_to_watch = api.BLOCK_SIGNALS_SENSOR + s

            self.airserver.submit([msg_watchdog, msg_limiter])

        # Activate the threshold controllers
        self.airserver.submit(map(api.MsgControllerActivate, self.channels))

        # Activate watchdogs and pressure limiter controllers
        self.airserver.submit(map(api.MsgControllerActivate, self.channels + 20))
        self.airserver.submit(map(api.MsgControllerActivate, self.channels + 60))

    def _command_softhand(self, signed_delta):
        if not type(signed_delta) is list:
            signed_delta = [signed_delta] * len(self.channels)
        assert (len(signed_delta) == len(self.channels))
        valve_open = [api.MsgSignalEvent(api.BLOCK_SIGNALS_CLIENT + i, inflate_or_deflate, 0.0) for i, inflate_or_deflate in zip(self.channels.tolist(), np.sign(signed_delta))]
        valve_close  = [api.MsgSignalEvent(api.BLOCK_SIGNALS_CLIENT + i, 0.0, delta) for i, delta in zip(self.channels.tolist(), np.abs(signed_delta))]
        self.airserver.submit(valve_open + valve_close)

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
        self._command_softhand(DexRobotZeke.SOFTHAND_SYNERGY)
        sleep(1.0)

    def unGrip(self, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        self._command_softhand(-1.0)
        sleep(1.0)
                    
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
            psi = angles.pitch - np.pi / 2
            if psi < 0:
                psi = psi + 2 * np.pi
            elif psi > 2*np.pi:
                psi = psi - 2 * np.pi                
            print psi + ZekeState.THETA

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
        print 'Delta h', delta_h_wrist_fingers
        target_elev = target_pose.position.z# - ZekeState.DELTA_Z - delta_h_wrist_fingers
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
        #if abs(gamma) >= DexConstants.ROLL_THRESH:
        #    raise Exception("Can't perform rotation about x-axis on Zeke's gripper: "  + str(target_pose.rotation.euler))
            
        target_state = DexRobotZeke.pose_to_state(target_pose, self._target_state, angles)

        logging.info('Target state: %s' %(str(target_state)))

        aim_state = target_state.copy().set_arm_ext(target_state.arm_ext)
        pitch = angles.pitch % (2 * np.pi)
        logging.info('Pitch %f' %pitch)
        if pitch > np.pi / 2 and pitch < 3 * np.pi / 2 :
            aim_state.set_arm_rot(ZekeState.PHI + 10*np.pi / 8)
        else:
            aim_state.set_arm_rot(ZekeState.PHI + 6*np.pi / 8)            
        aim_state.set_gripper_grip(ZekeState.MAX_STATE().gripper_grip)
        
        #logging.info('Opening grippers')
        #self.unGrip()

        logging.info('Aiming Zeke toward goal state: %s' %(str(aim_state)))
        self.gotoState(aim_state, rot_speed, tra_speed, name + "_aim")

        logging.info('Moving gripper to goal state: %s' %(str(target_state)))
        self.gotoState(target_state, rot_speed / 3.0, tra_speed / 3.0, name + "_grasp")

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
