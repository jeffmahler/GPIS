from PyControl.py import PyControl
zeke = PyControl()
# Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
def run(zeke):
    current = zeke.getState()
    states = [
		[3.4, 0.07, 0.0056, 6.4, 0.035, 0],
		[3.4, 0.07, 0.1265, 6.4, 0.035, 0],
		[3.4, 0.07, 0.1265, 6.4, 0, 0],
		[3.4, 0.07, 0.1265, 6.4, 0, 0],
		[3.4, 0.07, 0.1265, 6.4, 0, 0],
		[3.4, 0.07, 0.1265, 6.4, 0, 0]
    ]
    for state in states:
        print state
        zeke.sendStateRequest(state)