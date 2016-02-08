
from PID import *
import time
import serial
import numpy as np
import math as m


class PyControl:
    def __init__(self, comm = "COM6",baudrate=115200,timeout=.04, offsets=[.505, .2601, .234-.0043, 0.0164,-.2996], gripperParams=[.12, 0, -.08]):
        # initialize Serial Connection
        self.ser = serial.Serial(comm,baudrate)
        self.ser.setTimeout(timeout)
        time.sleep(1)
        
        self.offsets = offsets;
        self.gripperParams = gripperParams
        

    def xyz(self):
        try:
            state = self.getState()
            r = self.offsets[0]
            thetaAOffset = self.offsets[1]
            aOffset = self.offsets[2]
            zOffset = self.offsets[3]
            thetaGOffset = self.offsets[4]
            gX = self.gripperParams[0]
            gY = self.gripperParams[1]
            gZ = self.gripperParams[2]
            
            thetaA = state[0]+thetaAOffset
            z = state[1] + zOffset
            a = state[2]+aOffset
            thetaG = state[3]+thetaGOffset
        except:
            return "Comm Failure"

        X = (m.cos(thetaA)*gX - m.sin(thetaA)*m.cos(thetaG)*gY + 
            m.sin(thetaA)*m.sin(thetaG)*gZ + r+a*m.cos(thetaA))
        Y = (m.sin(thetaA)*gX + m.cos(thetaA)*m.cos(thetaG)*gY 
            - m.cos(thetaA)*m.sin(thetaG)*gZ + a*m.sin(thetaA))
        Z = m.sin(thetaG)*gY+m.cos(thetaG)*gZ + z;
        
        
        return [X,Y,Z]
        
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
        # Returns Array: Rotation, Elevation,Extension,Wrist,Jaws,Turntable
        self.ser.flushInput()
        self.ser.write("b")
        sensorVals = []
        for i in range(0,6):
            try:
                sensorVals.append(float(self.ser.readline()))
            except:
                time.sleep(.5)
                return 'Comm Failure'
        return sensorVals
        
        
    def getPots(self):
        self.ser.flushInput()
        self.ser.write("p")
        sensorVals = []
        for i in range(0,6):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'  
                
        return sensorVals
        
    def getCurrents(self):
        self.ser.flushInput()
        self.ser.write("c")
        sensorVals = []
        for i in range(0,4):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'   
        return sensorVals
        
    def getEncoders(self):
        self.ser.flushInput()
        self.ser.write("e")
        sensorVals = []
        for i in range(0,3):
            try:
                sensorVals.append(int(self.ser.readline()))
            except:
                return 'Comm Failure'   
        return sensorVals
    
        
        
    
        
    
    

    
        
