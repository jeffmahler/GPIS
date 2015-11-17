class ZekeState:

    @staticmethod
    def is_rot(i):
        return i in (0, 3, 5)

    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    def __init__(self, vals):
        self.state = vals[::]

    def __str__(self):
        return "Rot: {0}, Ele: {1}, Ext: {2}, Wrist: {3}, Grip: {4}, Table: {5}".format(
            self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5])
        
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
        
    @property
    def table_rot(self):
        return self.state[5]
        
    def set_table_rot(self, val):
        self.state[5] = val
        return self
        
    def copy(self):
        return ZekeState(self.state[::])