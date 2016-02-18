class DexSensorReadings:
    NUM_SENSORS = 1

    def __init__(self, values=None):
        self.values_ = values

    @property
    def gripper_force(self):
        """ Return the force reading """
        # TODO: convert to actual force values
        return self.values_[0]
    
    def __str__(self):
        return "Force: {0}".format(self.gripper_force)

