from serial import Serial
from multiprocessing import Process, Queue
from DexConstants import DexConstants
from DexNumericSolvers import DexNumericSolvers
from DexSensorReadings import DexSensorReadings
from Logger import Logger

import copy
import logging
import IPython
import numpy as np
import time

class _DexSerial(Process):
    #Private class that abstracts continuous serial communication with DexRobots    
    def __init__(self, State, flags_q, states_q, state_read_q, comm, baudrate, timeout, sensor_read_q=None):
        Process.__init__(self)
        
        self._State = State
        
        self._comm = comm
        self._baudrate = baudrate
        self._timeout = timeout
        
        self._flags_q = flags_q
        self._states_q = states_q
        self._state_read_q = state_read_q
        self._sensor_read_q = sensor_read_q
        
        self._updated_state = False

        self._flags = {
            "stopping" : False,
            "pausing" : False,
            "reading" : False,
        }

    def run(self):
        self._current_state = self._State.INIT_STATE()
        
        #Main run function that constantly sends the current state to the robot
        if not DexConstants.DEBUG:
            self.ser = Serial(self._comm,self._baudrate)
            self._stop_robot()
            time.sleep(DexConstants.INIT_DELAY)
            self._current_state = self._getStateSerial()
        
        self._states_q.put({"Type" : "State", "Data" : self._current_state})
        while True:
            if not self._flags_q.empty():
                flag = self._flags_q.get();
                self._flags[flag[0]] = flag[1]
                                    
            if self._flags["stopping"]:
                self._stop_robot()
                if not DexConstants.DEBUG:
                    self.ser.close()
                break

            if self._flags["reading"]:
                self._state_read_q.put(self._getStateSerial())
                if self._sensor_read_q is not None:
                    self._sensor_read_q.put(self._getSensorsSerial())
                self._flags["reading"] = False

            if self._sensor_read_q is None:
                Logger.log("empty sensor", self._current_state, self._State.NAME)

            #sending the current state
            if self._updated_state:
                self._sendSingleStateRequest(self._current_state)
                #self._updated_state = False

            time.sleep(DexConstants.INTERP_TIME_STEP)

            #only update current state if state queue is not empty and we're not pausing
            if not self._states_q.empty() and not self._flags["pausing"]:
                item = self._states_q.get()
                type, data = item["Type"], item["Data"]
                if type == "Label" and data:
                    Logger.log("Going to state ", data, self._State.NAME)
                elif type == "State":
                    self._current_state = data
                    self._updated_state = True
                elif type == "*":
                    self._sendArbitraryRequest(data)

    def _isValidState(self, state):
        for i in range(self._State.NUM_STATES):
            if self._State.is_rot(i):
                bound = DexConstants.INTERP_MAX_RAD
            else:
                bound = DexConstants.INTERP_MAX_M
                
            if abs(state.state[i] - self._current_state.state[i]) > bound:
                return False
        return True          

    def _sendArbitraryRequest(self, msg):
        Logger.log("Sent Arbitrary Message", msg, self._State.NAME)

        if DexConstants.DEBUG:
            return
            
        self.ser.write(msg)

    def _sendSingleStateRequest(self, state):
        Logger.log("Sent State", state, self._State.NAME)
   
        if DexConstants.DEBUG:
            self._current_state = state
            return
            
        if not self._isValidState(state):
            raise Exception("State is invalid or out of bounds for safety reasons")
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for x in state.state:
            val = int(x*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))
        
        if self._State.NAME == "Zeke" or self._State.NAME == "Izzy":
            val = 0
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
        if DexConstants.DEBUG:
            return

        self._control([0,0,0,0,0,0])
        
    def _getStateSerial(self):
        if DexConstants.DEBUG:
            return self._current_state

        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        
        num_vals = self._State.NUM_STATES
        
        if self._State.NAME == "Zeke":
            num_vals = 6
        
        for i in range(num_vals):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'

        if self._State.NAME == "Zeke":
            sensorVals = sensorVals[:-1]
            
        return self._State(sensorVals)

    def _getSensorsSerial(self):
        if DexConstants.DEBUG:
            return self._current_sensors

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

        
class DexSerialInterface:
    
    def __init__(self, State, comm, baudrate=115200, timeout=.01, read_sensors=False):
        self._comm = comm
        self._baudrate = baudrate
        self._timeout = timeout
        self._State = State
        self._read_sensors = read_sensors
        
        self._reset()
        self._target_state = None
        self.state_hist = []
        
    def _reset(self):
        self._flags_q = Queue()
        self._states_q = Queue()
        self._state_read_q = Queue()
        self._sensor_read_q = None
        if self._read_sensors:
            self._sensor_read_q = Queue()
        self._dex_serial = _DexSerial(self._State, self._flags_q, self._states_q, self._state_read_q, self._comm, self._baudrate, self._timeout,
                                      sensor_read_q=self._sensor_read_q)
        
    def start(self):
        self._dex_serial.start()
        if not DexConstants.DEBUG:
            time.sleep(DexConstants.INIT_DELAY)
        
    def stop(self):
        self._flags_q.put(("stopping", True))
        self._reset()

    def pause(self):
        self._flags_q.put(("pausing", True))
            
    def resume(self):
        self._flags_q.put(("resume", True))
        
    def getState(self):
        self._flags_q.put(("reading", True))

        # wait for states to show up
        started = time.time()
        while self._state_read_q.empty():
            if time.time() - started > DexConstants.ROBOT_OP_TIMEOUT:
                raise Exception("Get State timed out")

        # get latest reading
        cur_state = self._state_read_q.get()
        while not self._state_read_q.empty():
            cur_state = self._state_read_q.get()
        return cur_state

    def getSensors(self):
        if self._read_sensors:
            self._flags_q.put(("reading", True))

            # wait for sensors to show up
            started = time.time()
            while self._sensor_read_q.empty():
                if time.time() - started > DexConstants.ROBOT_OP_TIMEOUT:
                    raise Exception("Get Sensors timed out")

            # get latest sensor reading
            cur_sensor = self._sensor_read_q.get()
            while not self._sensor_read_q.empty():
                cur_sensor = self._sensor_read_q.get()
            return cur_sensor
        return None

    def _queueState(self, state):
        self._states_q.put({"Type" : "State", "Data" : state.copy()})
        
    def _queueLabel(self, label):
        self._states_q.put({"Type" : "Label", "Data" : label})

    def queueArbitraryRequest(self, data):
        self._states_q.put({"Type": "*", "Data": data})
        
    def maintainState(self, s):
        num_pauses = s / DexConstants.INTERP_TIME_STEP
        self._queueLabel("Maintaining State")
        state = self._target_state.copy()
        for _ in range(int(num_pauses)):
            self._queueState(state)
            self.state_hist.append(state) 
    
    def is_action_complete(self):
        return self._states_q.empty()
    
    def gotoState(self, target_state, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED,
                  name = None, block=True, timeout=3.0, state_eps=0.01, diff_eps=0.0001, time_delta=0.01):
        speeds_ids = target_state.speeds_ids
        speeds = (tra_speed, rot_speed)
                
        if self._target_state is None:
            self._target_state = self.getState()

        if rot_speed > DexConstants.MAX_ROT_SPEED or tra_speed > DexConstants.MAX_TRA_SPEED:
            raise Exception("Rotation or translational speed too fast.\nMax: {0} rad/sec, {1} m/sec\nGot: {2} rad/sec, {3} m/sec ".format(
                                    DexConstants.MAX_ROT_SPEED, DexConstants.MAX_TRA_SPEED, rot_speed, tra_speed))
        
        for i in range(len(target_state.state)):
            if target_state.state[i] is None:
                target_state.state[i] = self._target_state.state[i]
        
        states_vals = DexNumericSolvers.interpolate(self._State.NUM_STATES, 
                                                                    self._target_state.state, 
                                                                    target_state.copy().state, 
                                                                    speeds_ids, 
                                                                    speeds, 
                                                                    DexConstants.INTERP_TIME_STEP)
        #states_vals = [target_state.copy().state[::]]

        self._queueLabel(name)
        for state_val in states_vals:
            state = self._State(state_val)
            self._queueState(state)
            self.state_hist.append(state)

        # block until the target state is reached or a timeout occurs
        if block:
            move_thresh = 1.0
            duration = 0.0
            cur_state = np.array(self.getState().state)
            prev_state = np.zeros(cur_state.shape)

            # stop when either (a) target state is reached or (b) a timeout
            while np.linalg.norm(cur_state - np.array(states_vals[-1])) > state_eps and duration < timeout and \
                    (duration < move_thresh or np.linalg.norm(cur_state - prev_state) > diff_eps):
                time.sleep(time_delta)
                prev_state = copy.copy(cur_state)
                cur_state = np.array(self.getState().state)
                duration = duration + time_delta

                #print
                #print 'Cur', cur_state
                #print 'Target', states_vals[-1]
            print 'Target State', target_state
            print 'Current State', cur_state
            print 'Prev Diff', np.linalg.norm(cur_state - prev_state)
            print 'Target Diff', np.linalg.norm(cur_state - np.array(states_vals[-1]))
            print 'Duration', duration

        self._target_state = target_state.copy()
