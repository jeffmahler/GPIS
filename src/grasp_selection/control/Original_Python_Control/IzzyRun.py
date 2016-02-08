# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 14:37:41 2015

@author: David
"""

from TurnTableControl import * 
from PyControl import * 
from xboxController import *
import numpy as np

t = TurnTableControl() # the com number may need to be changed. Default of com7 is used
izzy = PyControl("COM3",115200, .04, [0,0,0,0,0],[0,0,0]); #same with this
 
#DIRECT CONTROL

def directControl(): # use this to 
    c = XboxController([100,155,155,155,90,100])
    while True:
        controls = c.getUpdates()     
        if controls is None:
            print "Done"
            izzy.stop()
            break 
        controls[1] = 0
        controls[3] = 0
        izzy.control(controls)
        t.control([controls[5]])
        
        # store this however you please. (concatenate into array?)
        simpleControls = [controls[0], controls[2], controls[4], controls[5]]
        
        #print getSimpleState()
        time.sleep(.03)

def simpleControl(controls): # controls = rotation,extension,grip,turntable (run this in a loop)
    # this converts a list of 4 control inputs into the necessary 6 and sends them off
    izzy.control( [controls[0]]+[0]+[controls[1]]+[0]+[controls[2]]+[0])
    t.control([controls[4]])
    
def getSimpleState():
    # this returns the state of the 4 axis
    state = izzy.getState()  
    simpleState = [state[0],state[2],state[4]]+t.getState()
    return simpleState
        #IF YOU WANT TO AVOID UP AND DOWN, AND WRIST, SEND ZEROS FOR controls[1] and controls[3]
#print getSimpleState()
directControl()

# these are needed to close serial connections setup in the begining 
izzy.stop()
izzy.ser.close()
t.stop()
t.ser.close()