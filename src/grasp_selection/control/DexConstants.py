from tfx import point, rotation
from numpy import pi
class DexConstants:

    ZEKE_LOCAL_FRAME = "ZEKE_LOCAL_FRAME"
    WORLD_FRAME = "WORLD_FRAME"
    
    ORIGIN = point(0,0,0)
    
    INTERP_TIME_STEP = 0.05 #50ms interpolation time step
    INTERP_MAX_RAD = pi/6 #30 degrees maximum rotation per interp time step
    INTERP_MAX_M = 0.1 #10cm maximum translation per interp time step
