from serial import Serial
from DexConstants import DexConstants
from time import sleep, time
from ZekeState import ZekeState
from TurntableState import TurntableState
from IzzyState import IzzyState
from DexSensorReadings import DexSensorReadings

class DexStateViewer:

    def __init__(self, State, comm, baudrate=115200, timeout=.01, print_sensors=DexConstants.READ_FORCE_SENSORS):
        self.ser = Serial(comm, baudrate)
        #self.ser.setTimeout(timeout)
        self._State = State
        self._print_sensors = print_sensors
        sleep(DexConstants.INIT_DELAY)
    
    def monitor(self, period = 10):
        start = time()
        while time() - start < period:
             print self._getState()
             if self._print_sensors:
                 print self._getSensors()
             sleep(DexConstants.INTERP_TIME_STEP)
    
    def _getState(self):
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        
        num_vals = self._State.NUM_STATES
        
        if self._State.NAME == "Zeke" or self._State.NAME == "IZZY":
            num_vals = 6
        
        for i in range(num_vals):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        return self._State(sensorVals)

    def _getSensors(self):
        self.ser.flushInput()
        self.ser.write("f")
        sensorVals = []
        
        num_vals = DexSensorReadings.NUM_SENSORS
        
        for i in range(num_vals):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        return DexSensorReadings(sensorVals)

    @staticmethod
    def viewZeke(period = 300, comm = DexConstants.ZEKE_COMM):
        viewer = DexStateViewer(ZekeState, comm, print_sensors=False)
        viewer.monitor(period)
        
    @staticmethod
    def viewIzzy(period = 300, comm = DexConstants.IZZY_COMM):
        viewer = DexStateViewer(IzzyState, comm, print_sensors=False)
        viewer.monitor(period)
        
    @staticmethod
    def viewTable(period = 300, comm = DexConstants.TABLE_COMM):
        viewer = DexStateViewer(TurntableState, comm)
        viewer.monitor(period)

if __name__ == '__main__':
    DexStateViewer.viewZeke()
    #DexStateViewer.viewTable()
