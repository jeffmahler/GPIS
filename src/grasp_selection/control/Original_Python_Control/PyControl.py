import time
import serial
from time import sleep
import numpy as np
class PyControl:

    STEP_ROT = np.pi / 50
    STEP_TRA = 0.005
    STEP_TIME = 0.05
    NUM_STATES = 6
    
    def __init__(self, comm = "COM3",baudrate=115200,timeout=.01):
        # initialize Serial Connection
        self.ser = serial.Serial(comm,baudrate)
        self.ser.setTimeout(timeout)
        time.sleep(2)
        
    def stop(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls([0,0,0,0,0,0])
        return
        
    def _sendSingleStateRequest(self, requests):
        print "about to send ", requests
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for thing in requests:
            val = int(thing*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))
            
    def _interpolate(self, initial, target):
        initial = initial[::] #copies the array
        target = target[::]
        
        states = []
        is_rot = lambda x : x in (0, 3, 5)
                        
        def allTrue(lst):
            for b in lst:
                if not b:
                    return False
            return True
                
        sgn = [0] * PyControl.NUM_STATES
        for i in range(PyControl.NUM_STATES):
            if target[i] > initial[i]:
                sgn[i] = 1
            elif target[i] == initial[i]:
                sgn[i] = 0
            else:
                sgn[i] = -1
            
        current = initial
        reached = [False] * PyControl.NUM_STATES
        while not allTrue(reached):
            for i in range(PyControl.NUM_STATES):
                if not reached[i]:
                    if is_rot(i):
                        next = current[i] + sgn[i] * PyControl.STEP_ROT
                    else:
                        next = current[i] + sgn[i] * PyControl.STEP_TRA
                        
                    if sgn[i] == 0:
                        reached[i] = True
                        current[i] = target[i]
                    elif sgn[i] == 1:
                        if next >= target[i]:
                            reached[i] = True
                            current[i] = target[i]
                        else:
                            current[i] = next
                    else:
                        if next <= target[i]:
                            reached[i] = True
                            current[i] = target[i]
                        else:
                            current[i] = next
            
            states.append(current[::])
        return states

    def sendStateRequest(self, target):
        current = self.getState() 
        states = self._interpolate(current, target)
        for state in states:
            self._sendSingleStateRequest(state)
            sleep(PyControl.STEP_TIME)

    def sendControls(self,requests):
        # Converts an array of requests to an array of PWM signals sent to the robot
        # Checks out of bounds 
        self.ser.flushOutput()
        PWMs = []
        for req in requests:
            if req >= 0:
                PWMs.append(int(req))
                PWMs.append(0)
            else:
                PWMs.append(0)
                PWMs.append(int(abs(req)))
        # send PWMs
        for e in PWMs:
            self.ser.write(chr(e))
        
    def control(self,requests):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls(requests)

    def getState(self):
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,6):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        return sensorVals