from serial import Serial
from multiprocessing import Process, Queue
from time import sleep, time
from DexConstants import DexConstants
from DexInterpolater import DexInterpolater

class _ZekeSerial(Process):    
    #Private class that abstracts continuous serial communication with Zeke

    INIT_DELAY = 3 #magic 3 seconds initial wait for serial connection to stablize
    NUM_STATES = 6
    
    def __init__(self, flags_q, states_q, state_read_q, comm, baudrate, timeout):
        Process.__init__(self)
        self._comm = comm
        self._baudrate = baudrate
        self._timeout = timeout
        
        self._flags_q = flags_q
        self._states_q = states_q
        self._state_read_q = state_read_q
        
        self._flags = {
            "stopping" : False,
            "pausing" : False,
            "reading" : False,
        }

    def run(self):
        #Main run function that constantly sends the current state to the robot
        self.ser = Serial(self._comm,self._baudrate)
        self.ser.setTimeout(self._timeout)
        self._stop_robot()
        sleep(_ZekeSerial.INIT_DELAY)
        
        self._current_state = self._getStateSerial()
        
        self._states_q.put(self._current_state)
        while True:
            if not self._flags_q.empty():
                flag = self._flags_q.get();
                self._flags[flag[0]] = flag[1]
                                    
            if self._flags["stopping"]:
                self._stop_robot()
                self.ser.close()
                break

            if self._flags["reading"]:
                self._state_read_q.put(self._getStateSerial())
                self._flags["reading"] = False
            
            #sending the current state
            self._sendSingleStateRequest(self._current_state)
            
            sleep(DexConstants.INTERP_TIME_STEP)
            
            #only update current state if state queue is not empty and we're not pausing
            if not self._states_q.empty() and not self._flags["pausing"]:
                self._current_state = self._states_q.get()

    def _isValidState(self, state):
        is_rot = lambda x : x in (0, 3, 5)
        for i in range(_ZekeSerial.NUM_STATES):
            if is_rot(i):
                bound = DexConstants.INTERP_MAX_RAD
            else:
                bound = DexConstants.INTERP_MAX_M
            if abs(state[i] - self._current_state[i]) >= bound:
                return False
        return True                   

    def _sendSingleStateRequest(self, requests):
        if not self._isValidState(requests):
            raise Exception("State is invalid or out of bounds for safety reasons")
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for thing in requests:
            val = int(thing*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))

    def _sendControls(self, requests):
        self.ser.flushOutput()
        PWMs = []
        for req in requests:
            if req >= 0:
                PWMs.append(int(req))
                PWMs.append(0)
            else:
                PWMs.append(0)
                PWMs.append(int(abs(req)))
        for e in PWMs:
            self.ser.write(chr(e))
            
    def _control(self, requests):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self._sendControls(requests)
        
    def _stop_robot(self):
        self._control([0,0,0,0,0,0])
        
    def _getStateSerial(self):
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,6):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        return sensorVals
        
class ZekeSerialInterface:
    
    def __init__(self, comm = "COM3", baudrate=115200, timeout=.01):
        self._comm = comm
        self._baudrate = baudrate
        self._timeout = timeout
        self._reset()
        
    def _reset(self):
        self._flags_q = Queue()
        self._states_q = Queue()
        self._state_read_q = Queue()
        self._zeke_serial = _ZekeSerial(self._flags_q, self._states_q, self._state_read_q, self._comm, self._baudrate, self._timeout)
        
    def start(self):
        self._zeke_serial.start()
        
    def stop(self):
        self._flags_q.put(("stopping", True))
        self._reset()

    def pause(self):
        self._flags_q.put(("pausing", True))
            
    def resume(self):
        self._flags_q.put(("resume", True))
        
    def getState(self):
        self._flags_q.put(("reading", True))
        
        started = time()
        while self._state_read_q.empty():
            if time() - started > DexConstants.ROBOT_OP_TIMEOUT:
                raise Exception("Get State timed out")

        return self._state_read_q.get()
        
    def _queueState(self, state):
        self._states_q.put(state[::])
        
    def gotoState(self, target_state, rot_speed, tra_speed):
        speeds_ids = (1, 0, 0, 1, 0, 1)
        speeds = (tra_speed, rot_speed)
        
        states = DexInterpolater.interpolate(self.getState(), target_state[::], speeds_ids, speeds, DexConstants.INTERP_TIME_STEP)
        for state in states:
            self._queueState(state)