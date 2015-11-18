from operator import add
from tfx import pose, transform, vector, rotation
from DexConstants import DexConstants
from ZekeSerial import ZekeSerialInterface
from ZekeState import ZekeState
from math import sqrt
from numpy import pi, arctan, cos, sin
from time import sleep
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class DexRobotZeke:
    '''
    Abstraction for a robot profile. Contains all information specific
    to the Zeke robot, including its physical dimensions, joints
    accepted poses, etc. 
    '''

    #For the two offsets below, actual angle = desired angle + OFFSET
    PHI = 0.3 #zeke arm rotation angle offset to make calculations easier.
    THETA = 3.0967 #zeke wrist rotation 0 degree offset.
    
    RESET_STATES = {"GRIPPER_SAFE_RESET": ZekeState([pi + PHI, 0.1, 0.01, THETA, 0.036, 0]),
                                "ZEKE_RESET": ZekeState([pi + PHI, 0.01, 0.01, THETA, 0.036, 0])}
    
    ZEKE_LOCAL_T = transform(
                                            vector(-0.22, 0, 0), 
                                            rotation.identity(), 
                                            parent=DexConstants.ZEKE_LOCAL_FRAME,
                                            frame=DexConstants.WORLD_FRAME)
    
    def __init__(self, comm = DexConstants.COMM, baudrate = DexConstants.BAUDRATE, timeout = DexConstants.SER_TIMEOUT):
        self._zeke= ZekeSerialInterface(comm, baudrate, timeout)      
        self._zeke.start()
        self._target_state = self.getState()
    
    def reset(self, rot_speed, tra_speed):
        self.gotoState(DexRobotZeke.RESET_STATES["GRIPPER_SAFE_RESET"], rot_speed, tra_speed)
        self.gotoState(DexRobotZeke.RESET_STATES["ZEKE_RESET"], rot_speed, tra_speed)
            
    def stop(self):
        self._zeke.stop()
        
    def getState(self):
        return self._zeke.getState()       
        
    def grip(self, tra_speed):
        state = self._target_state.copy()
        state.set_gripper_grip(DexConstants.MIN_STATE.gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed)
        
    def unGrip(self, tra_speed):
        state = self._target_state.copy()
        state.set_gripper_grip(DexConstants.MAX_STATE.gripper_grip)
        self.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed)
    
    @staticmethod
    def _pose_IK(pose):
        '''
        Takes in a pose w/ respect to zeke and returns the following list of joint settings:
        Elevation
        Rotation about Z axis
        Extension of Arm
        Rotation of gripper
        '''
        settings = {}
        settings["elevation"] = pose.position.z
        
        #calculate rotation about z axis
        x = pose.position.x
        y = pose.position.y
        
        theta = 0
        if x == 0:
            if y >= 0:
                theta = pi / 2
            else: 
                theta = - pi / 2
        else:
            theta_ref = abs(arctan(y/x))
            if theta_ref > pi/2:
                theta_ref = pi - theta_ref

            if x >= 0 and y >= 0:
                theta = theta_ref
            elif y >= 0 and x < 0:
                theta = pi - theta_ref
            elif y < 0 and x < 0:
                theta = pi + theta_ref
            else:
                theta = 2*pi - theta_ref
                
        settings["rot_z"] = theta
        settings["extension"] = sqrt(pow(x, 2) + pow(y, 2))
        settings["rot_y"] = pose.rotation.euler['sxyz'][1]
        
        return settings
        
    @staticmethod
    def _settings_to_state(settings, prev_state):
        '''
        Takes in a list of joint settings and concats them into one single 
        final target state. Basically forward kinematics
        '''
        # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
        state_vals = [0] * 6
        state_vals[0] = settings["rot_z"] + DexRobotZeke.PHI
        state_vals[1] = settings["elevation"]
        state_vals[2] = settings["extension"]
        state_vals[3] = settings["rot_y"] + DexRobotZeke.THETA
        state_vals[4] = prev_state.gripper_grip
        state_vals[5] = prev_state.table_rot

        return ZekeState(state_vals)
        
    @staticmethod
    def pose_to_state(target_pose, prev_state):
        if target_pose.frame is not DexConstants.ZEKE_LOCAL_FRAME:
            raise Exception("Given target_pose is not in ZEKE LOCAL frame")
        joint_settings = DexRobotZeke._pose_IK(target_pose)
        target_state = DexRobotZeke._settings_to_state(joint_settings, prev_state)
        return target_state
        
    def gotoState(self, target_state, rot_speed, tra_speed):
        self._zeke.gotoState(target_state, rot_speed, tra_speed)
        self._target_state = target_state.copy()

    def transform(self, target_pose, rot_speed, tra_speed):
        target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose

        if abs(target_pose.rotation.euler['sxyz'][0]) >= 0.0001:
            raise Exception("Can't perform rotation about x-axis on Zeke's gripper")
            
        target_state = DexRobotZeke.pose_to_state(target_pose, self._target_state)
        
        self.gotoState(target_state, rot_speed, tra_speed)
        
    def _state_FK(self, state):
        # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
        arm_angle = state.arm_rot - DexRobotZeke.PHI
        z = state.arm_elev
        r = state.arm_ext
        x = r * cos(arm_angle)
        y = r * sin(arm_angle)
        
        return (x, y, z)
        
    def plot(self):
        hist = self._zeke.state_hist
        
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