from numpy import pi
class TurntableState:

    NUM_STATES = 1
    NAME = "Tuntable"
                
    @staticmethod
    def INIT_STATE():
        return TurntableState([0.4819])

    @staticmethod
    def MIN_STATE():
        return TurntableState([0.4819])
        
    @staticmethod
    def MAX_STATE():
        return TurntableState([0.4819 + 2 * pi])
        
    @staticmethod
    def is_rot(i):
        return i in (0,)
        
    @property
    def speeds_ids(self):
        return (1,)

    def __init__(self, vals = [None] * NUM_STATES):
        self.state = vals[::]

    def __str__(self):
        return "Table Rot: {0}".format(self.table_rot)
            
    def __repr__(self):
        return "TurntableState([{0}])".format(self.table_rot)
        
    @property
    def table_rot(self):
        return self.state[0]
        
    def set_table_rot(self, val):
        self.state[0] = val
        return self
        
    def copy(self):
        return TurntableState(self.state[::])