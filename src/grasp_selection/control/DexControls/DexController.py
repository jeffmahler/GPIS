from DexRobotIzzy import DexRobotIzzy
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState
from IzzyState import IzzyState
from DexConstants import DexConstants
from DexAngles import DexAngles
from DexNumericSolvers import DexNumericSolvers
from DexTurntableSolver import DexTurntableSolver
from tfx import pose, rotation, rotation_tb
from numpy import pi, array, cos, sin, arccos, dot, ravel
from copy import deepcopy
from numpy.linalg import norm
from time import sleep, time
import matplotlib.pyplot as plt

class DexController:
    '''Transformation Controller class. Controls both Izzy and Turntable
    '''
    
    def __init__(self, izzy = None, table = None):
        if izzy is None:
            print "Initialization Izzy..."
            izzy = DexRobotIzzy()
        if table is None:
            print "Initialization Table..."
            table = DexRobotTurntable()
            
        self._izzy = izzy
        self._table = table

    def do_grasp(self, stf, name = "", rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED):
     
        print "Computing Grasp Angles..."
        target_pose, angles = DexController._stf_to_graspable_pose_and_angles(stf)
        self._latest_pose_unprocessed = target_pose.copy()
       
        original_obj_angle = DexNumericSolvers.get_cartesian_angle(target_pose.position.x, target_pose.position.y)
 
        print "Modifying Grasp Pose for Table Rotation..."
        #change target pose to appropriate approach pose
        self._set_approach_pose(target_pose, angles)
        
        aligned_obj_angle = DexNumericSolvers.get_cartesian_angle(target_pose.position.x, target_pose.position.y)

        print 'Orig Theta', original_obj_angle
        print 'Aligned Theta', aligned_obj_angle

        print "Reseting Table Rotation..."
        #reset izzy to clear-table-rotation position
        self._table.reset()
        self._izzy.reset_clear_table()
        #wait til completed
        while not self._table.is_action_complete():
            sleep(0.01)
        
        #for debugging plot
        self._latest_pose = target_pose.copy()
        self._latest_angles = deepcopy(angles)
        
        print "Rotating Table..."
        #transform target_pose to table
        target_obj_angle = aligned_obj_angle - original_obj_angle
        if target_obj_angle < 0:
            target_obj_angle += 2*pi
        target_table_state = TurntableState().set_table_rot(target_obj_angle + DexRobotTurntable.THETA)
        self._table.gotoState(target_table_state, rot_speed, tra_speed, name+"_table")
        
        #wait til completed
        while not self._table.is_action_complete():
            sleep(0.01)
        
        print "Executing Grasp..."
        #transform target_pose to izzy 
        self._izzy.transform_aim_extend_grip(target_pose, name, angles, rot_speed, tra_speed)
        
        return target_pose.copy()

    @staticmethod
    def _stf_to_graspable_pose_and_angles(stf):
        original_pose = stf.pose
        translation = original_pose.position
        rotation = array(original_pose.rotation.matrix)
        
        def _angle_2d(u, v):
            u_norm = u / norm(u)
            R = array([[u_norm[0], u_norm[1]],
                       [-u_norm[1], u_norm[0]]])
            vp = R.dot(v)

            #returns angle between 2 vectors in degrees
            theta = DexNumericSolvers.get_cartesian_angle(vp[0], vp[1])
            return theta

        def _angle_3d(u, v):
            theta = arccos(dot(u,v) / norm(u) / norm(v) )
            if theta < 0:
                theta += 2*pi
            return theta
                
        #phi is angle between projection of grasp translation in world coords onto the table and the y axis of the grasp in world frame
        proj_g_t_w = array([translation[0], translation[1]])
        proj_y_axis_grasp = -ravel(rotation[:2,1])
        phi = _angle_2d(proj_g_t_w, proj_y_axis_grasp)

        print 'Phi', phi
        
        #psi is angle between x-axis of the grasp in world frame and the table's xy plane    
        x_axis_grasp = ravel(rotation[:,0])
        proj_x_axis_grasp = x_axis_grasp.copy()
        proj_x_axis_grasp[2] = 0
        proj_x_axis_grasp = dot(rotation.T, proj_x_axis_grasp).ravel()
        x_axis_grasp = dot(rotation.T, x_axis_grasp).ravel()

        u_x = array([x_axis_grasp[0], x_axis_grasp[2]])
        v_x = array([proj_x_axis_grasp[0], proj_x_axis_grasp[2]])
        psi = _angle_2d(u_x, v_x) + pi / 2

        print 'Psi', psi
        
        #gamma is angle between the y-axis of the grasp in world frame and the table's xy plane
        y_axis_grasp = ravel(rotation[:,1])
        proj_y_axis_grasp = y_axis_grasp.copy()
        proj_y_axis_grasp[2] = 0
        gamma = _angle_3d(proj_y_axis_grasp, y_axis_grasp)
        
        print 'Gamma', gamma

        return original_pose, DexAngles(phi, psi, gamma)
        
    def _set_approach_pose(self, target_pose, angles):
        pos = [target_pose.position.x, target_pose.position.y]
        r = norm(pos)
        #ANGLES using yaw
        phi = angles.yaw
        d = IzzyState.IZZY_ARM_ORIGIN_OFFSET
        theta = DexTurntableSolver.solve(r, d, phi)
        
        target_pose.position.x = r * cos(theta)
        target_pose.position.y = r * sin(theta)

    def reset(self, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        self._table.reset()
        self._izzy.reset()
                
    def stop(self):
        self._table.stop()
        self._izzy.stop()
        
    def getState(self):
        return self._izzy.getState(), self._table.getState()
        
    def pause(self, s):
        self._table.maintainState(s)
        self._izzy.maintainState(s)
        
    def plot_approach_angle(self):
        fig = plt.figure()
        
        axis = plt.gca()
        axis.set_xlim([-0.2,0.2])
        axis.set_ylim([-0.2,0.2])
        
        x = self._latest_pose.position.x
        y = self._latest_pose.position.y
        theta = DexNumericSolvers.get_cartesian_angle(x, y)
        
        r = norm([x**2, y**2])
        #ANGLES using yaw
        phi = self._latest_angles.yaw
        
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
       
class _STF:

    def __init__(self, pose):
        self.pose = pose
       
def test(phi):

    target = pose((0.05, 0.05, 0.05), rotation_tb(phi, 90, 0), frame = DexConstants.WORLD_FRAME)

    t = DexController()
    t._table.reset()
    t.do_grasp(target)
    raised = target.copy()
    raised.position.z = 0.25
    sleep(0.1)
    t._izzy.transform(raised, "Raised")
    t.plot_approach_angle()
    t._izzy.plot()
    t.stop()
