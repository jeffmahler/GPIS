import time
import serial
import numpy as np
class PyControl:
    def __init__(self, comm = "COM3",baudrate=115200,timeout=.01):
        # initialize Serial Connection
        self.ser = serial.Serial(comm,baudrate)
        self.ser.setTimeout(timeout)
        time.sleep(1)
        
    def stop(self):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls([0,0,0,0,0,0])
        return
        
    def sendStateRequest(self,requests):
        self.ser.flushInput()
        self.ser.flushOutput()
        self.ser.write("a")
        for thing in requests:
            val = int(thing*10000000)
            self.ser.write(chr((val>>24) & 0xff))
            self.ser.write(chr((val>>16) & 0xff))
            self.ser.write(chr((val>>8) & 0xff))
            self.ser.write(chr(val & 0xff))

    def sendControls(self,requests):
        # Converts an array of requests to an array of PWM signals sent to the robot
        # Checks out of bounds 
        self.ser.flushOutput()
        PWMs = []
        for i in range(0,len(requests)):
            req = requests[i]
            if req >= 0:
#                if self.state[i+7]>self.maxStates[i]:
#                    req = 0
                PWMs.append(int(req))
                PWMs.append(0)
                
            else:
#                if self.state[i+7]<self.minStates[i]:
#                    req = 0
                PWMs.append(0)
                PWMs.append(int(abs(req)))
        # send PWMs
        for elem in PWMs:
            self.ser.write(chr(elem))
            
        return
        
    def control(self,requests):
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write("s")
        self.sendControls(requests)
        return


    def getState(self):
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,6):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                return 'Comm Failure'
            
        # timeout gives time for arduino to catch up no matter what
        # encoders 1-3, encoder velocities 1-3, potentiometers 1-6, currents 1-4
        # rotation elevation extension
        # rotation elevation extension wrist closure turntable
        # rotation elevation extension closure       
        return sensorVals
        
    def getPositonState(self):
        self.ser.flushInput()
        self.ser.write("c")
        sensorVals = []
        for i in range(0,6):
            thing = self.ser.readline();
            if type(thing)!=float:
                return
            sensorVals.append(float(self.ser.readline()))
        # timeout gives time for arduino to catch up no matter what
        # encoders 1-3, encoder velocities 1-3, potentiometers 1-6, currents 1-4
        # rotation elevation extension
        # rotation elevation extension wrist closure turntable
        # rotation elevation extension closure       
        return sensorVals