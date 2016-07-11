import numpy as np
import os
import sys
import csv
from  scipy.stats import multivariate_normal

sys.path.append("/home/jmahler/sherdil_working/GPIS/src/grasp_selection/control/DexControls")
from DexAngles import DexAngles
from DexConstants import DexConstants
from DexController import DexController
from DexRobotZeke import DexRobotZeke
from ZekeState import ZekeState
from TurntableState import TurntableState

sys.path.append('/home/jmahler/sherdil_working/GPIS/src/grasp_selection')
import similarity_tf as stf
import tfx

import pdb


if __name__ == '__main__':
    
    num_runs  = int(sys.argv[1])
    ctrl = DexController()
    
    for i in range(num_runs):
        ctrl.reset()
	print("")
	yesno = raw_input('Place object for RANDOM push. Hit [ENTER] when done')
	print("Running Random Push " + str(i + 1) + " of " + str(num_runs))
        ctrl.random_push()
