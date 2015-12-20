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
                
        self._latest_pose_unprocessed = target_pose.copy()
        
        #change target pose to appropriate approach pose
        self._set_approach_pose(target_pose)
        
        #reset zeke to clear-table-rotation position
        self._zeke.reset_clear_table()
        
        #for debugging plot
        self._latest_pose = target_pose.copy()
        
        #transform target_pose to table
        self._table.transform(target_pose, name + "_table", rot_speed, tra_speed) 
        
        #wait til completed
        while not self._table.is_action_complete():
            sleep(0.01)
        
        #transform target_pose to zeke 
        self._zeke.transform_aim_extend_grip(target_pose, name, rot_speed, tra_speed)
        
        return target_pose.copy()

    def _set_approach_pose(self, target_pose):
        pos = [target_pose.position.x, target_pose.position.y]
        r = norm(pos)
        phi = target_pose.rotation.tb_angles.yaw_rad
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
        
    def pause(self, s):
        self._table.maintainState(s)
        self._zeke.maintainState(s)
        
    def plot_approach_angle(self):
        fig = plt.figure()
        
        axis = plt.gca()
        axis.set_xlim([-0.2,0.2])
        axis.set_ylim([-0.2,0.2])
        
        x = self._latest_pose.position.x
        y = self._latest_pose.position.y
        theta = DexNumericSolvers.get_cartesian_angle(x, y)
        
        r = norm([x**2, y**2])
        phi = self._latest_pose.rotation.tb_angles.yaw_rad
        
        x_o = self._latest_pose_unprocessed.position.x
        y_o = self._latest_pose_unprocessed.position.y
        theta_o =DexNumericSolvers.get_cartesian_angle(x_o, y_o)
        
        #vector to original obj pos
        v_obj_o = array([0, 0, x_o, y_o])
        
        #original vector of object in direction of grasp
        v_grasp_o = array([x_o, y_o, r * cos(phi + theta_o) * 10, r * sin(phi + theta_o) * 10])
        
        #vector to obj pos
        v_obj = array([0, 0, x, y])
        
        #vector of object in direction of grasp
        v_grasp = array([x, y, r * cos(phi + theta) * 10, r * sin(phi + theta) * 10])
        
        #vector of of arm to position of object
        v_arm = array([0.2, 0, x - 0.2, y])
               
        soa =array([v_arm, v_obj, v_grasp, v_obj_o, v_grasp_o]) 
        X,Y,U,V = zip(*soa)
        axis.quiver(X,Y,U,V,angles='xy',scale_units='xy',scale=1)
        
        plt.draw()
        plt.show()
        
origin = pose(DexConstants.ORIGIN, DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)
raised = pose((0, 0, 0.15), DexConstants.DEFAULT_GRIPPER_EULER, frame = DexConstants.WORLD_FRAME)

def test(phi):
    target = pose((0.05, 0.05, 0.05), rotation_tb(phi, 90, 0), frame = DexConstants.WORLD_FRAME)
    print target.rotation.euler
    t = DexController()
    t._table.reset()
    t.do_grasp(target)
    raised = target.copy()
    raised.position.z = 0.25
    sleep(0.1)
    t._zeke.transform(raised, "Raised")
    t.plot_approach_angle()
    t._zeke.plot()
    t.stop()