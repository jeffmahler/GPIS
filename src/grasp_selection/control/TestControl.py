from PyControl import * 
from operator import add
import time
import numpy as np

NUM_STATES = 6

maxStates = [6.58991792126, 0.3, 0.3, 6.61606370861, 0.0348490572685, 6.83086639225];
minStates = [0,.02,.01,0.183086039735,-.01,0.197775641646];

#DIRECT CONTROL
#Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
def move(controls):
	zeke = PyControl()
	zeke.stop()
	print "Before: ", zeke.getState()
	print "Controlling: ", controls
	print "After: ", zeke.getState()
	zeke.control(controls)
	zeke.ser.close()
	
def stop():
	zeke = PyControl()
	zeke.stop()
	zeke.ser.close()

def setState(state):
	zeke = PyControl()
	print "Before: ", zeke.getState()
	print "Sent State: ", state
	zeke.sendStateRequest(state)
	zeke.ser.close()
	
def changeState(deltas):
	cur_state = getState()
	target_state = map(add, cur_state, deltas)
	setState(target_state)
	
def getState():
	zeke = PyControl()
	state = zeke.getState()
	zeke.ser.close()
	return state