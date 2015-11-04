import numpy as np
from serial import Serial
from multiprocessing import Process, Pipe
from Queue import Queue
from time import sleep, time
from DexConstants import DexConstants

class ZekeSerial(Process):    

    INIT_DELAY = 3 #magic 3 seconds initial wait for serial connection to stablize
    NUM_STATES = 6
    
    def __init__(self, flags_pipe, states_pipe, comm, baudrate, timeout):
        print "hi 1"
        Process.__init__(self)
        self._comm = comm
        self._baudrate = baudrate
        self._timeout = timeout
        self._flags_pipe = flags_pipe
        self._states_pipe = states_pipe
        
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
        sleep(ZekeSerial.INIT_DELAY)
        
        self._current_state = self._getStateSerial()
        self._read_state = self._current_state[::]
        
        self._state_queue = Queue()
        self._state_queue.put(self._current_state)
        while True:
            print "hi"
            flags = self._flags_pipe.recv()
            print "hi"
            if flags:
                for flag, value in flags.items():
                    self._flags[flag] = value
                    
            new_state = self._states_pipe.recv()
            if new_state:
                #queues new state for the robot to follow
                self._state_queue.put(state[::])
                
            if self._flags["stopping"]:
                self._stop_robot()
                self.ser.close()
                break

            if self._flags["reading"]:
                self._read_state = self._getStateSerial()
                self._flags["reading"] = False
            
            #sending the current state
            self._sendSingleStateRequest(self._current_state)
            
            sleep(DexConstants.INTERP_TIME_STEP)
            
            #only update current state if state queue is not empty and we're not pausing
            if not self._state_queue.empty() and not self._flags["pausing"]:
                self._current_state = self._state_queue.get()

    def _isValidState(self, state):
        is_rot = lambda x : x in (0, 3, 5)
        for i in range(ZekeSerial.NUM_STATES):
            if is_rot(i):
                bound = DexConstants.INTERP_MAX_RAD
            else:
                bound = DexConstants.INTERP_MAX_M
            if abs(state[i] - self._current_state[i]) >= bound:
                return False
        return True      

    def _sendSingleStateRequest(self, requests):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for thing in requests:
            val = int(thing*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))

    def _sendControls(self,requests):
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
            
    def control(self,requests):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self._sendControls(requests)
        
    def _stop_robot(self):
        self.control([0,0,0,0,0,0])

    def getState(self):
        '''
        Returns current state of the robot by asking the thread to read the state using _getStateSerial
        '''
        self._flags["reading"] = True
        started = time()
        while self._flags["reading"]:
            if time() - started > 2:
                self._flags["reading"] = False
                raise Exception("Get State timed out")
        return self._read_state
        
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
        flags_pipe_send, flags_pipe_recv = Pipe()
        states_pipe_send, states_pipe_recv = Pipe()
        self._zeke_serial = ZekeSerial(flags_pipe_recv, states_pipe_recv, comm, baudrate, timeout)
        self._flags_pipe = flags_pipe_send
        self._states_pipe = states_pipe_recv
        
    def start(self):
        self._zeke_serial.start()
        
    def stop(self):
        self._flags_pipe.send({"stopping", True})

    def pause(self):
        self._flags_pipe.send({"pausing", True})
            
    def resume(self):
        self._flags_pipe.send({"resume", True})
        
    def queueState(self, state):
        if not self._isValidState(state):
            raise Exception("State is invalid or out of bounds for safety reasons")
        self._state_pipe.send(state)
        
z  = ZekeSerialInterface()