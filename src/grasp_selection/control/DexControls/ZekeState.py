from numpy import pi
class ZekeState:

    NUM_STATES = 5
    NAME = "Zeke"
    
    #For the two offsets below, actual angle = desired angle + OFFSET
    PHI = 0.29 # zeke arm rotation angle offset to make calculations easier.
    THETA = 1.15 # zeke wrist rotation 0 degree offset.
    DELTA_Z = 0.028 # zeke arm 0 elevation in world coordinates
    WRIST_TO_FINGER_RADIUS = 0.006 # the radius of the fingers from the center of wrist rotation
    
    ZEKE_ARM_ORIGIN_OFFSET = 0.59
    ZEKE_ARM_TO_GRIPPER_TIP_LENGTH = 0.41
    
    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    
    @staticmethod
    def INIT_STATE():
        return ZekeState([3.49, 0.01, 0.01, 0.53, 0, 0])
        
    @staticmethod
    def MIN_STATE():
        return ZekeState([0 , 0.008, 0.008, 0.1665, 0.0005, 0])
        
    @staticmethod
    def MAX_STATE():
        return ZekeState([2*pi, 0.3, 0.3, 2*pi, 0.068, 2*pi])
    
    @staticmethod
    def is_rot(i):
        return i in (0, 3)
        

    def to_dict(self):
        d = {}
        d['arm_rot'] = self.state[0]
        d['arm_elev'] = self.state[1]
        d['arm_ext'] = self.state[2]
        d['gripper_rot'] = self.state[3]
        d['gripper_grip'] = self.state[4]
        return d

    @property
    def speeds_ids(self):
        return (1, 0, 0, 1, 0)
        
    # Rotation, Elevation, Extension, Wrist rotation, Grippers
    def __init__(self, vals = [None] * NUM_STATES):
        self.state = vals[::]

    def __str__(self):
        return "Rot: {0}, Ele: {1}, Ext: {2}, Wrist: {3}, Grip: {4}".format(
            self.arm_rot, self.arm_elev, self.arm_ext, self.gripper_rot, self.gripper_grip)
            
    def __repr__(self):
        return "ZekeState([{0}, {1}, {2}, {3}, {4}])".format(
            self.arm_rot, self.arm_elev, self.arm_ext, self.gripper_rot, self.gripper_grip)
        
    @property
    def arm_rot(self):
        return self.state[0]
        
    def set_arm_rot(self, val):
        self.state[0] = val
        return self
        
    @property
    def arm_elev(self):
        return self.state[1]
        
    def set_arm_elev(self, val):
        self.state[1] = val
        return self
        
    @property
    def arm_ext(self):
        return self.state[2]
        
    def set_arm_ext(self, val):
        self.state[2] = val
        return self
        
    @property
    def gripper_rot(self):
        return self.state[3]
        
    def set_gripper_rot(self, val):
        self.state[3] = val
        return self
        
    @property
    def gripper_grip(self):
        return self.state[4]
        
    def set_gripper_grip(self, val):
        self.state[4] = val
        return self
        
    def copy(self):
        return ZekeState(self.state[::])
