from serial import Serial
from DexConstants import DexConstants
from time import sleep, time
from ZekeState import ZekeState

class ZekeStateViewer:

    def __init__(self, comm = "COM3", baudrate=115200, timeout=.01):
        self.ser = Serial(comm, baudrate)
        self.ser.setTimeout(timeout)
        sleep(DexConstants.INIT_DELAY)
    
    def monitor(self, period = 10):
        start = time()
        while time() - start < period:
             print self._getState()
             sleep(DexConstants.INTERP_TIME_STEP)
    
    def _getState(self):
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(6):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        return ZekeState(sensorVals)