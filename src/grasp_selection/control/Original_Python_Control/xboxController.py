import pygame
import numpy as np
import time


class XboxController:
    def __init__(self,scales = [.3,.01,.01,.01,.01,.01]):
        pygame.init()
        pygame.joystick.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()
        
        self.lStick = LeftStick(self.controller.get_axis(0),
                                   self.controller.get_axis(1))
        self.rStick = RightStick(self.controller.get_axis(4),
                                     self.controller.get_axis(3))
        self.dPad = DPad(self.controller.get_hat(0))
        self.trigger = Trigger(self.controller.get_axis(2))
        self.inUse = [False,False,False,False]
        
        length = 6
        self.offsets = np.zeros(length)
        self.uScale = np.ones(length)
        self.lScale = np.ones(length)
        self.driftLimit = .05
        self.calibrate()
        self.scales = np.array(scales)
        time.sleep(1)
        self.calibrate()
        
    def getUpdates(self):
        for event in pygame.event.get(): # User did something
            if event.type == pygame.JOYBUTTONDOWN and self.controller.get_button(7) == 1.0: # If user clicked close
                return None
            elif self.controller.get_button(3) == 1.0:
                return 'switch'
                # Flag that we are done so we exit this loop
                
        state = self.getControllerState()
        updates = self.convert(state)     
        return updates*self.scales
        
    def calibrate(self):
        # calibrate sticks 
        # reset offsets and scaling factors 
        length = len(self.offsets)
        self.offsets = np.zeros(length)
        self.uScale = np.ones(length)
        self.lScale = np.ones(length)
        
        state = self.getControllerState()      
        self.offsets = self.convert(state)    
        self.uScale = abs(1/(np.ones(length)-self.offsets))
        self.lScale = abs(1/(-np.ones(length)-self.offsets))

        
    def convert(self,state):
        rot = -state['right_stick'][0]
        ext = -state['right_stick'][1]  
        wrist = state['left_stick'][0]
        elev = -state['left_stick'][1]
        grip = state['d_pad'][0]
        turntable = state['trigger']
        
        # offset
        updates = np.array([rot,elev,ext,wrist,grip,turntable])-self.offsets
        
        #scale upper and lower bounds 
        for i in range(0,len(updates)):
            if updates[i]>0:
                updates[i] = updates[i]*self.uScale[i]
            else:
                updates[i] = updates[i]*self.lScale[i]
            if abs(updates[i]) < self.driftLimit:
                updates[i] = 0
        
                
        return updates
        

    def getControllerState(self):
        pygame.event.clear()
        self.update()
        #if self.isInUse():
        state = {'left_stick':self.lStick.getPos(),
                     'right_stick':self.rStick.getPos(),
                     'd_pad':self.dPad.getPos(),
                     'trigger':self.trigger.getPos()}
        return state
            
    def update(self):
        self.lStick.setCurrent(self.controller.get_axis(0),
                                   self.controller.get_axis(1))
        self.rStick.setCurrent(self.controller.get_axis(4),
                                     self.controller.get_axis(3))
        self.dPad.setCurrent(self.controller.get_hat(0))
        self.trigger.setCurrent(self.controller.get_axis(2))

    def isInUse(self):
        self.inUse = [self.lStick.isInUse(), self.rStick.isInUse(),
                      self.dPad.isInUse(), self.trigger.isInUse()]
        for thing in self.inUse:
            if thing:
                return thing

        return False


class LeftStick:
    def __init__(self, axis0, axis1):
        self.initA0 = axis0
        self.initA1 = axis1

        self.a0 = self.initA0
        self.a1 = self.initA1

    def getPos(self):
        return np.array([self.a0, self.a1])

    def setCurrent(self, a0, a1):
        self.a0 = a0
        self.a1 = a1
        return self.getPos()

    def isInUse(self):
        return (self.a0!=self.initA0 or self.a1!=self.initA1)


class RightStick:
    def __init__(self, axis0, axis1):
        self.initA0 = axis0
        self.initA1 = axis1

        self.a0 = self.initA0
        self.a1 = self.initA1

    def getPos(self):
        return np.array([self.a0, self.a1])

    def setCurrent(self, a0, a1):
        self.a0 = a0
        self.a1 = a1
        return self.getPos()

    def isInUse(self):
        return (self.a0!=self.initA0 or self.a1!=self.initA1)

class DPad:
    def __init__(self, hat):
        self.initH = hat
        self.h = self.initH

    def getPos(self):
        return self.h

    def setCurrent(self, h):
        self.h = h
        return self.getPos()

    def isInUse(self):
        return self.h!=self.initH

class Trigger:
    def __init__(self, axis0):
        self.initA0 = axis0
        self.a0 = self.initA0

    def getPos(self):
        return self.a0

    def setCurrent(self, a0):
        self.a0 = a0
        return self.getPos()

    def isInUse(self):
        return self.a0!=self.initA0

