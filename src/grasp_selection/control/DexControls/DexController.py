from DexRobotZeke import DexRobotZeke
from DexRobotTurntable import DexRobotTurntable
from DexConstants import DexConstants
from DexNumericSolvers import DexNumericSolvers
from DexTurntableSolver import DexTurntableSolver
from tfx import pose, rotation, rotation_tb
from numpy import pi, array, cos, sin
from numpy.linalg import norm
from time import sleep, time
import matplotlib.pyplot as plt
class DexController:
    '''Transformation Controller class. Controls both Zeke and Turntable
    '''
    
    def __init__(self, zeke = None, table = None):
        if zeke is None:
            zeke = DexRobotZeke()
        if table is None:
            table = DexRobotTurntable()
            
        self._zeke = zeke
        self._table = table

    def do_grasp(self, target_pose, name = "", rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
        #target_pose is a tfx.pose object
        if target_pose.frame is not DexConstants.WORLD_FRAME:
            raise Exception("Given target_pose is not in WORLD frame")
        
        #change target pose to appropriate approach pose
        self._set_approach_pose(target_pose)
        
        #reset zeke to clear-table-rotation position
        self._zeke.reset_clear_table()
        
        #for debugging plot
        self._latest_pose = target_pose.copy()
        
        #transform target_pose to table
        self._table.transform(target_pose, rot_speed, tra_speed, name + "_table") 
        
        #wait til completed
        while not self._table.is_action_complete():
            sleep(0.01)
        
        #transform target_pose to zeke 
        self._zeke.transform(target_pose, rot_speed, tra_speed, name)

    def _set_approach_pose(self, target_pose):
        pos = [target_pose.position.x, target_pose.position.y]
        r = norm(pos)
        phi = target_pose.rotation.euler['sxyz'][2]
        d = DexConstants.ZEKE_ARM_ORIGIN_OFFSET
        theta = DexTurntableSolver.solve(r, d, phi)
        
        target_pose.position.x = r * cos(theta)
        target_pose.position.y = r * sin(theta)        
        
    def reset(self, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        self._table.reset()
        self._zeke.reset()
                
    def stop(self):
        self._table.stop()
        self._zeke.stop()
        
    def getState(self):
        return self._zeke.getState(), self._table.getState()
        
    def plot(self):
        plt.figure()
        
        axis = plt.gca()
        axis.set_xlim([-0.2,0.2])
        axis.set_ylim([-0.2,0.2])
        
        x = self._latest_pose.position.x
        y = self._latest_pose.position.y
        r = norm([x**2, y**2])
        phi = self._latest_pose.rotation.euler['sxyz'][2]
        theta = DexNumericSolvers.get_cartesian_angle(x, y)
        
        #vector to obj pos
        v_obj = array([0, 0, x, y])
        
        #vector of object in direction of grasp
        v_grasp = array([x, y, r * cos(phi + theta) * 10, r * sin(phi + theta) * 10])
        
        #vector of of arm to position of object
        v_arm = array([0.2, 0, x - 0.2, y])
               
        soa =array([v_arm, v_obj, v_grasp]) 
        X,Y,U,V = zip(*soa)
        axis.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
        
        plt.draw()
        plt.show()

origin = pose(DexConstants.ORIGIN, DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)
raised = pose((0, 0, 0.15), DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)

def test(phi):
    target = pose((0.05, 0.05, 0.1), rotation_tb(phi, 0, 0), frame = DexConstants.WORLD_FRAME)
    t = DexController()
    t.reset()
    t.do_grasp(target)
    t.plot()
    t.stop()

'''
def angle(ctrl, angle):
    ctrl.gotoState(ctrl.getState().set_gripper_rot(angle))
    
def dAngle(ctrl, delta):
    ctrl.gotoState(ctrl.getState().set_gripper_rot(ctrl.getState().gripper_rot + delta))
'''