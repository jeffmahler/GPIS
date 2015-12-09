from tfx import point, rotation, rotation_euler
from numpy import pi
from ZekeState import ZekeState
from TurntableState import TurntableState

class DexConstants:

    DEBUG = True
    PRINT_STATES = False
    LOGGING = True

    COMM = "COM3"
    BAUDRATE = 115200
    SER_TIMEOUT = 0.01

    INIT_DELAY = 3 #magic 3 seconds initial wait for serial connection to stablize
    PAUSE_DELAY = 0.1
    
    ROBOT_OP_TIMEOUT = 2

    ZEKE_LOCAL_FRAME = "ZEKE_LOCAL_FRAME"
    WORLD_FRAME = "WORLD_FRAME"
    
    ORIGIN = point(0,0,0)

    ZEKE_ARM_ORIGIN_OFFSET = 0.22
    
    MAX_ROT_SPEED = pi/180*150 #150 degrees per second maximum rotation
    MAX_TRA_SPEED = 0.3 #30cm per second maximum translation
    
    INTERP_TIME_STEP = 0.03 #30ms interpolation time step
    INTERP_MAX_RAD = MAX_ROT_SPEED * INTERP_TIME_STEP 
    INTERP_MAX_M = MAX_TRA_SPEED * INTERP_TIME_STEP 

    DEFAULT_ROT_SPEED = pi/6 #30 degrees per second
    DEFAULT_TRA_SPEED = 0.04 #4cm per second (roughly 5 secs for 1 ft)
    
    MAX_ELEVATION = 0.3
    
    DEFAULT_GRIPPER_EULER = rotation_euler(0, pi/2, 0, 'sxyz')
    
    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    MIN_STATE = ZekeState([0 , 0.008, 0.008, 0.1665, 0.001, 0])
    MAX_STATE = ZekeState([2*pi, 0.3, 0.3, pi, 0.032, 2*pi])
    
    TURNTABLE_MIN_STATE = TurntableState([0])
    TURNTABLE_MAX_STATE = TurntableState([2 * pi])
    