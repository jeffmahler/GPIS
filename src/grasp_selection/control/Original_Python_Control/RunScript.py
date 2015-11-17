
from TurnTableControl import * 
from PyControl import * 
from xboxController import *
import numpy as np

zeke = PyControl()#"COM6",115200, .04, [.505, .2601, .234-.0043, 0.0164], [.12, 0, -.08])
t = TurnTableControl()
izzy = PyControl("COM3",115200, .04, [0,0,0,0,0],[0,0,0]);
 
maxStates = [6.58991792126, .3, 0.3, 6.61606370861, 0.0348490572685, 6.83086639225]
minStates = [0,.02,.01,0.183086039735,-.01,0.197775641646]


#POSITION CONTROL
def kinematicControl():
    c = XboxController([.06,.005,.01,.08,.002,.05])
    time.sleep(1)
    tableState = t.getState()
    target1 = np.array(zeke.getState())
    target2 = np.array(izzy.getState())
    
    
    select = True
    while True:
        updates = c.getUpdates()
        if updates is "switch":
            select = not select  
            #target1 = np.array(zeke.getState())
            #target2 = np.array(izzy.getState())
            #tableState = t.getState()
        elif updates is None:
            print "Done"
            zeke.stop()
            break
        elif select:
            target1 = target1+updates
            tableState = tableState+updates[5]
            #t.sendStateRequest(tableState)
        else:
            target2 = target2+updates
            tableState = tableState+updates[5]
            #t.sendStateRequest(tableState)
        
        
#        for i in range(0,6):
#            if target[i]>maxStates[i]:
#                target[i] = maxStates[i]
#            elif target[i] < minStates[i]:
#                target[i] = minStates[i]

        zeke.sendStateRequest(target1)
        izzy.sendStateRequest(target2)
        t.sendStateRequest(tableState)
        
        #print zeke.xyz()
        time.sleep(.01)

#DIRECT CONTROL

def directControl():
    c = XboxController([100,155,155,155,70,100])
    select = True
    while True:
        controls = c.getUpdates()
        if controls is "switch":
            select = not select
        elif controls is None:
            print "Done"
            zeke.stop()
            break 
        elif select:
            zeke.control(controls)
        else:
            izzy.control(controls)
        time.sleep(.03)
kinematicControl()

zeke.stop()
zeke.ser.close()
izzy.stop()
izzy.ser.close()
t.stop()
t.ser.close()
