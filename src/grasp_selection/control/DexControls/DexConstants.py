from tfx import point, rotation#, rotation_euler
from numpy import pi
from ZekeState import ZekeState
from TurntableState import TurntableState

class DexConstants:

    DEBUG = False
    PRINT_STATES = False
    LOGGING = True

    ZEKE_COMM = "/dev/ttyACM0"
    TABLE_COMM = "/dev/ttyACM1"
    IZZY_COMM = "/dev/ttyACM2"
    BAUDRATE = 115200
    SER_TIMEOUT = 0.01

    INIT_DELAY = 4 #magic 3 seconds initial wait for serial connection to stablize
    PAUSE_DELAY = 0.01

    RESET_FISHING_DELAY = 13
    
    ROBOT_OP_TIMEOUT = 2

    ZEKE_LOCAL_FRAME = "ZEKE_LOCAL_FRAME"
    IZZY_LOCAL_FRAME = "IZZY_LOCAL_FRAME"
    WORLD_FRAME = "WORLD_FRAME"
    
    MAX_ROT_SPEED = pi/180*150 #150 degrees per second maximum rotation
    MAX_TRA_SPEED = 0.3 #30cm per second maximum translation
    
    INTERP_TIME_STEP = 0.03 #30ms interpolation time step
    INTERP_MAX_RAD = MAX_ROT_SPEED * INTERP_TIME_STEP 
    INTERP_MAX_M = MAX_TRA_SPEED * INTERP_TIME_STEP 

    DEFAULT_ROT_SPEED = pi/6 #30 degrees per second
    DEFAULT_TRA_SPEED = 0.04 #4cm per second (roughly 5 secs for 1 ft)
    
    MAX_ELEVATION = 0.3
    ROLL_THRESH = 0.4
    GRIPPER_CLOSE_FORCE_THRESH = 400.0
    GRIPPER_CLOSE_EPS = 0.001
    GRIPPER_CLOSE_TIMEOUT = 3.0

    #DEFAULT_GRIPPER_EULER = rotation_euler(0, pi/2, 0, 'sxyz')
    
