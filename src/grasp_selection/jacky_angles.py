import sys
from tfx import pose, rotation_tb, rotation
from similarity_tf import SimilarityTransform3D
from numpy import array, c_, sqrt
from numpy.linalg import norm
from time import sleep
sys.path.append("src/grasp_selection/control/DexControls")
from DexController import DexController

def normalize(u):
    return u / norm(u)

x_axis = normalize(array([1, 1, 0]))
y_axis = normalize(array([-1, 1, 0]))
z_axis = normalize(array([0,0,1]))
R = c_[x_axis, c_[y_axis, z_axis]]

test_pose = pose((0.05, 0.05, 0.15), R)
test_stf = SimilarityTransform3D(test_pose)

def test_angles(stf):
    target_pose, angles = DexController._stf_to_graspable_pose_and_angles(test_stf)
    phi = angles.yaw_deg
    psi = angles.pitch_deg
    gamma = angles.roll_deg

    print "phi is {0}".format(phi)
    print "psi is {0}".format(psi)
    print "gamma is {0}".format(gamma)
    
def test_grasp(stf):
    ctrl = DexController()
    ctrl.do_grasp(stf)
    while not ctrl._zeke.is_action_complete():
        sleep(0.01)
    grasp_state, _ = ctrl.getState()
    high_state = grasp_state.copy().set_arm_elev(0.3)
    print grasp_state
    print "Sending high low states..."
    #ctrl._zeke.gotoState(high_state)
    #ctrl._zeke.maintainState(3)
    #ctrl._zeke.gotoState(grasp_state)
    #ctrl.plot_approach_angle()
    #ctrl._zeke.plot()
    return ctrl
        
c = test_grasp(test_stf)
c._zeke.plot()
