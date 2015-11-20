from tfx import point, rotation
from numpy import pi
from ZekeState import ZekeState

class DexConstants:

    DEBUG = False
    PRINT_STATES = False

    COMM = "COM3"
    BAUDRATE = 115200
    SER_TIMEOUT = 0.01

    INIT_DELAY = 3 #magic 3 seconds initial wait for serial connection to stablize
    
    ROBOT_OP_TIMEOUT = 2

    ZEKE_LOCAL_FRAME = "ZEKE_LOCAL_FRAME"
    WORLD_FRAME = "WORLD_FRAME"
    
    ORIGIN = point(0,0,0)
    
    MAX_ROT_SPEED = pi/180*70 #70 degrees per second maximum rotation
    MAX_TRA_SPEED = 0.15 #15cm per second maximum translation
    
    INTERP_TIME_STEP = 0.03 #30ms interpolation time step
    INTERP_MAX_RAD = MAX_ROT_SPEED * INTERP_TIME_STEP 
    INTERP_MAX_M = MAX_TRA_SPEED * INTERP_TIME_STEP 

    DEFAULT_ROT_SPEED = pi/6 #30 degrees per second
    DEFAULT_TRA_SPEED = 0.04 #4cm per second (roughly 5 secs for 1 ft)
    
    MAX_ELEVATION = 0.3
    
    # Rotation, Elevation, Extension, Wrist rotation, Grippers, Turntable
    MIN_STATE = ZekeState([0 , 0.008, 0.008, 0.1831, 0.001, 0])
    MAX_STATE = ZekeState([2*pi, 0.3, 0.3, 2*pi, 0.036, 2*pi])
    
    DEBUG_INIT_STATE = ZekeState([3.49, 0.01, 0.01, 0.53, 0, 0])