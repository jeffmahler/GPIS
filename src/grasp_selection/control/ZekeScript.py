from PyControl import * 
from xboxController import *
import numpy as np

zeke = PyControl()

zeke.stop()

maxStates = [6.58991792126, .3, 0.3, 6.61606370861, 0.0348490572685, 6.83086639225];
minStates = [0,.02,.01,0.183086039735,-.01,0.197775641646];


#POSITION CONTROL
def kinematicControl():
    c = XboxController([.06,.005,.01,.08,.002,.05])
    target = np.array(zeke.getState())
    j = 0;
    while True:
        j+=1
        updates = c.getUpdates()
        #print updates
        if updates is None:
            print "Done"
            zeke.stop()
            break
        target = target+updates
        for i in range(0,6):
            if target[i]>maxStates[i]:
                target[i] = maxStates[i]
            elif target[i] < minStates[i]:
                target[i] = minStates[i]

        zeke.sendStateRequest(target)
        print zeke.getState()[0]
        time.sleep(.03)

#DIRECT CONTROL
# Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
def directControl():
    c = XboxController([100,155,155,155,155,100])
    while True:
        controls = c.getUpdates()
        if controls is None:
            print "Done"
            zeke.stop()
            break 
       # else if controls is 'switch':
            # Switch control to the other robot
        state = zeke.getState();
        print state
        #print state[2],
        #print state[9]
        #print controls
        zeke.control(controls)
        time.sleep(.02)
kinematicControl()

zeke.stop()
zeke.ser.close()