from DexRobotIzzy import DexRobotIzzy
from DexRobotZeke import DexRobotZeke
from DexRobotTurntable import DexRobotTurntable
from TurntableState import TurntableState
from IzzyState import IzzyState
from ZekeState import ZekeState
from DexConstants import DexConstants
from DexAngles import DexAngles
from DexNumericSolvers import DexNumericSolvers
from DexTurntableSolver import DexTurntableSolver
from tfx import pose, rotation, rotation_tb
from numpy import pi, array, cos, sin, arccos, dot, ravel, c_
from copy import deepcopy
from numpy.linalg import norm
from time import sleep, time
import matplotlib.pyplot as plt

import copy
import logging
import numpy as np
import sys
sys.path.append('/home/jmahler/jeff_working/GPIS/src/grasp_selection')
import IPython
import similarity_tf as stf
import tfx

# TODO: update with configurable sleep times / actually make the is action complete function work correctly

class DexController:
    '''Transformation Controller class. Controls both Izzy and Turntable
    '''
    
    def __init__(self, robot = None, table = None):
        if robot is None:
            logging.info('Initializing Robot')
            robot = DexRobotZeke()
        if table is None:
            logging.info('Initializing Turntable')
            table = DexRobotTurntable()
            
        self._robot = robot
        self._table = table

    def __del__(self):
        self.stop()

    def do_grasp(self, stf, name = "", rot_speed = DexConstants.DEFAULT_ROT_SPEED, tra_speed = DexConstants.DEFAULT_TRA_SPEED, sleep_val=1.0):
     
        logging.info('Computing grasp angles')
        target_pose, angles = DexController._stf_to_graspable_pose_and_angles(stf)
        self._latest_pose_unprocessed = target_pose.copy()

        print 'Before', target_pose.position
        original_obj_angle = DexNumericSolvers.get_cartesian_angle(target_pose.position.x, target_pose.position.y)

        #change target pose to appropriate approach pose
        logging.info('Modifying grasp pose for table rotation')
        self._set_approach_pose(target_pose, angles)
        
        print 'After', target_pose.position
        aligned_obj_angle = DexNumericSolvers.get_cartesian_angle(target_pose.position.x, target_pose.position.y)

        logging.info('Orig Theta: %f' %original_obj_angle)
        logging.info('Aligned Theta %f' %aligned_obj_angle)

        #reset izzy to clear-table-rotation position
        logging.info('Reseting table rotation')
        self._table.reset()
        self._robot.reset_clear_table()
        #wait til completed
        turntable_state = self._table.getState()
        
        #for debugging plot
        self._latest_pose = target_pose.copy()
        self._latest_angles = deepcopy(angles)

        #transform target_pose to table
        logging.info('Rotation table to grasp pose')
        target_obj_angle = aligned_obj_angle - original_obj_angle
        target_turntable_angle = target_obj_angle + turntable_state.table_rot

        # check valid angles
        if target_turntable_angle <= TurntableState.MIN_STATE().table_rot or \
                target_turntable_angle > TurntableState.MAX_STATE().table_rot:
            target_turntable_angle = target_turntable_angle - TurntableState.THETA
            target_turntable_angle = target_turntable_angle % (2*pi)
            target_turntable_angle = target_turntable_angle + TurntableState.THETA
            
        logging.info('Rotating table to %f' %(target_turntable_angle))
        target_table_state = TurntableState().set_table_rot(target_turntable_angle)
        self._table.gotoState(target_table_state, rot_speed, tra_speed, name+"_table")
        
        #wait til completed
        #transform target_pose to izzy 
        logging.info('Moving robot to grasp pose')
        target_state = self._robot.transform_aim_extend_grip(target_pose, name, angles, rot_speed, tra_speed)
        
        return copy.copy(target_state)

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

        logging.info('Phi: %f' %phi)
        
        #psi is angle between x-axis of the grasp in world frame and the table's xy plane    
        x_axis_grasp = ravel(rotation[:,0])
        proj_x_axis_grasp = x_axis_grasp.copy()
        proj_x_axis_grasp[2] = 0
        #psi = _angle_3d(x_axis_grasp, proj_x_axis_grasp) + pi / 2
        proj_x_axis_grasp = dot(rotation.T, proj_x_axis_grasp).ravel()
        x_axis_grasp = dot(rotation.T, x_axis_grasp).ravel()

        u_x = array([x_axis_grasp[0], x_axis_grasp[2]])
        v_x = array([proj_x_axis_grasp[0], proj_x_axis_grasp[2]])

        psi = _angle_2d(v_x, u_x)
        logging.info('Psi: %f' %psi)
        
        #gamma is angle between the y-axis of the grasp in world frame and the table's xy plane
        y_axis_grasp = ravel(rotation[:,1])
        proj_y_axis_grasp = y_axis_grasp.copy()
        proj_y_axis_grasp[2] = 0
        gamma = _angle_3d(proj_y_axis_grasp, y_axis_grasp)
        
        logging.info('Gamma: %f' %gamma)
        return original_pose, DexAngles(phi, psi, gamma)
        
    def _set_approach_pose(self, target_pose, angles):
        pos = [target_pose.position.x, target_pose.position.y]
        r = norm(pos)
        #ANGLES using yaw
        phi = angles.yaw
        d = ZekeState.ZEKE_ARM_ORIGIN_OFFSET
        theta = DexTurntableSolver.solve(r, d, phi)
        
        target_pose.position.x = r * cos(theta)
        target_pose.position.y = r * sin(theta)

        logging.debug('Target pose relative to table center: X = %f, Y = %f, Theta = %f' %(target_pose.position.x, target_pose.position.y, theta))

    def reset(self, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        self._table.reset()
        self._robot.reset()

    def reset_object(self, rot_speed=DexConstants.DEFAULT_ROT_SPEED, tra_speed=DexConstants.DEFAULT_TRA_SPEED):
        """ Reset function specific to object placements """
        self._table.reset()
        self._robot.reset_object()
                
    def stop(self):
        self._table.stop()
        self._robot.stop()
        
    def getState(self):
        return self._robot.getState(), self._table.getState()
        
    def pause(self, s):
        self._table.maintainState(s)
        self._robot.maintainState(s)
        
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
       
def test_state():
    target_state = ZekeState()
    target_state.set_arm_ext(ZekeState.ZEKE_ARM_ORIGIN_OFFSET - ZekeState.ZEKE_ARM_TO_GRIPPER_TIP_LENGTH)
    target_state.set_arm_rot(ZekeState.PHI + np.pi)
    target_state.set_gripper_grip(0.037)
    print 'Target'
    print target_state

    t = DexController()
    t._table.reset()
    t._robot.gotoState(target_state)

    current_state = t._robot.getState()
    print 'Reached'
    print current_state

def test_state_sequence():
    target_state = ZekeState()
    target_state.set_arm_ext(ZekeState.ZEKE_ARM_ORIGIN_OFFSET - ZekeState.ZEKE_ARM_TO_GRIPPER_TIP_LENGTH)
    target_state.set_arm_rot(ZekeState.PHI + np.pi)
    target_state.set_gripper_grip(0.037)
    target_state.set_gripper_rot(4.26)
    print 'Target'
    print target_state

    t = DexController()
    t._table.reset()
    t._robot.gotoState(target_state)

    target_state.set_arm_elev(0.1)
    t._robot.gotoState(target_state)

    target_state.set_gripper_rot(3.14)
    t._robot.gotoState(target_state)

    target_state.set_gripper_rot(4.26)
    t._robot.gotoState(target_state)

    target_state.set_arm_rot(ZekeState.PHI + 7 * np.pi / 8)
    t._robot.gotoState(target_state)

    current_state = t._robot.getState()
    print 'Reached'
    print current_state

def test_grasp():
    theta = np.pi / 2
    t = np.array([0.05, 0.05, 0.05])
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta), np.cos(theta), 0],
                  [0, 0, 1]])
    target = stf.SimilarityTransform3D(pose=tfx.pose(R, t), scale=1.0)

    t = DexController()
    t.do_grasp(target)

    current_state, _ = t.getState()
    print current_state
    
    t.plot_approach_angle()

    return t

def test_grip():
    t = DexController()

    #target_state = ZekeState([3.44, 0.108, 0.163, None, ZekeState.MAX_STATE().gripper_grip, 0])
    #t._robot.gotoState(target_state)
    #sleep(3)    

    
    print 'Ungripping'
    t._robot.unGrip()

    print 'Gripping'
    t._robot.grip()
    """

    current_state, _ = t.getState()
    high_state = current_state.copy().set_arm_elev(0.2)
    high_state.set_gripper_grip(ZekeState.MIN_STATE().gripper_grip)

    t._robot.gotoState(high_state)
    sleep(10)
    """

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    test_grip()
    #t = test_state()
    #t = test_state_sequence()
