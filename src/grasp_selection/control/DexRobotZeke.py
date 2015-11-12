from operator import add
from tfx import pose, transform, vector, rotation
from DexConstants import DexConstants
from ZekeSerial import ZekeSerialInterface
from math import sqrt
from numpy import pi, arctan
from time import sleep

class DexRobotZeke:
    '''
    Abstraction for a robot profile. Contains all information specific
    to the Zeke robot, including its physical dimensions, joints
    accepted poses, etc. 
    '''

    NUM_STATES = 6
    PHI = 0.35 #zeke arm rotation angle offset to make calculations easier
    
    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    MIN_STATES = [0 , 0.008, 0.008, 0.1831, 0.001, 0]
    MAX_STATES = [2*pi, 0.3, 0.3, 2*pi, 0.05, 2*pi]
    RESET_STATE = [pi + PHI, 0.01, 0.01, 0.5076, 0, 0]
    
    ZEKE_LOCAL_T = transform(
                                            vector(0.22, 0, 0), 
                                            rotation.identity(), 
                                            parent=DexConstants.WORLD_FRAME,
                                            frame="ZEKE_LOCAL")
    
    def __init__(self, comm, baudrate, time):
        self._zeke= ZekeSerialInterface(comm, baudrate, time)      
        self._zeke.start()
        sleep(DexConstants.INIT_DELAY)
    
    def reset(self, rot_speed, tra_speed):
        self._zeke.gotoState(DexRobotZeke.RESET_STATE, rot_speed, tra_speed)
            
    def stop(self):
        self._zeke.stop()
        
    def getState(self):
        return self._zeke.getState()       
    
    def grip(self, tra_speed):
        state = self._zeke.getState()
        state[4] = DexRobotZeke.MIN_STATES[4]
        self._zeke.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed)
        
    def unGrip(self, tra_speed):
        state = self._zeke.getState()
        state[4] = DexRobotZeke.MAX_STATES[4]
        self._zeke.gotoState(state, DexConstants.DEFAULT_ROT_SPEED, tra_speed)
    
    def _pose_IK(self, pose):
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
        
    def _settings_to_state(self, settings):
        '''
        Takes in a list of joint settings and concats them into one single 
        final target state. Basically forward kinematics
        '''
        # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
        state = [0] * 6
        state[0] = settings["rot_z"] + DexRobotZeke.PHI
        state[1] = settings["elevation"]
        state[2] = settings["extension"]
        state[3] = settings["rot_y"]
        state[4] = DexRobotZeke.MIN_STATES[4] #TODO: verify this is open gripper
        state[5] = DexRobotZeke.MIN_STATES[5]

        return state
        
    def transform(self, target_pose, rot_speed, tra_speed):
        target_pose = DexRobotZeke.ZEKE_LOCAL_T * target_pose

        if abs(target_pose.rotation.euler['sxyz'][0]) >= 0.0001:
            raise Exception("Can't perform rotation about x-axis on Zeke's gripper")

        joint_settings = self._pose_IK(target_pose)
        target_state = self._settings_to_state(joint_settings)
        self._zeke.gotoState(target_state, rot_speed, tra_speed)