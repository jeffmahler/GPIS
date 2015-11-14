from tfx import point, rotation
from numpy import pi
class DexConstants:

    DEBUG = False

    COMM = "COM3"
    BAUDRATE = 115200
    SER_TIMEOUT = 0.01

    INIT_DELAY = 3 #magic 3 seconds initial wait for serial connection to stablize
    
    ROBOT_OP_TIMEOUT = 2

    ZEKE_LOCAL_FRAME = "ZEKE_LOCAL_FRAME"
    WORLD_FRAME = "WORLD_FRAME"
    
    ORIGIN = point(0,0,0)
    
    MAX_ROT_SPEED = pi/4 #45 degrees per second maximum rotation
    MAX_TRA_SPEED = 0.08 #8cm per second maximum translation
    
    INTERP_TIME_STEP = 0.03 #30ms interpolation time step
    INTERP_MAX_RAD = MAX_ROT_SPEED * INTERP_TIME_STEP 
    INTERP_MAX_M = MAX_TRA_SPEED * INTERP_TIME_STEP 

    DEFAULT_ROT_SPEED = pi/6 #30 degrees per second
    DEFAULT_TRA_SPEED = 0.04 #4cm per second (roughly 5 secs for 1 ft)
    
    MAX_ELEVATION = 0.3