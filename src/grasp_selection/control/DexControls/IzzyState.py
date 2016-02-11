from numpy import pi
class IzzyState:

    NUM_STATES = 5
    NAME = "Izzy"
        
    #For the two offsets below, actual angle = desired angle + OFFSET
    PHI = 3.59  #izzy arm rotation angle offset to make calculations easier.
    THETA = 1.06 #izzy wrist rotation 0 degree offset.
    DELTA_Z = 0.049 #izzy arm 0 elevation in world coordinates
        
    IZZY_ARM_ORIGIN_OFFSET = -0.502
    IZZY_ARM_TO_GRIPPER_TIP_LENGTH = 0.395

    # Rotation, Elevation, Extension, Wrist rotation, Grippers
    @staticmethod
    def INIT_STATE():
        return IzzyState([3.58, 0.00556, 0.0185, 2.4, 0.01, 0])
        
    @staticmethod
    def MIN_STATE():
        return IzzyState([1.29, 0.00556, 0.0185, 1.06, 0.003, 0])
        
    @staticmethod
    def MAX_STATE():
        return IzzyState([5.6, 0.29, 0.34, 7.07, 0.035, 2*pi])
    
    @staticmethod
    def is_rot(i):
        return i in (0, 3)
        
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
        return "IzzyState([{0}, {1}, {2}, {3}, {4}])".format(
            self.arm_rot, self.arm_elev, self.arm_ext, self.gripper_rot, self.gripper_grip)
        
    @property
    def arm_rot(self):
        return self.state[0]
        
    def set_arm_rot(self, val):
        if val is not None:
            self.state[0] = val
        else:
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
        if val is not None:
            self.state[3] = val
        else:
            self.state[3] = val
        return self
        
    @property
    def gripper_grip(self):
        return self.state[4]
        
    def set_gripper_grip(self, val):
        self.state[4] = val
        return self
        
    def copy(self):
        return IzzyState(self.state[::])
