from os import walk, path
from sdf_class import SDF
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

from engine_creation import train_and_test_lsh

NUM_TRAIN = 20
NUM_TEST = 5
ROOT_DIR = "datasets/Cat50_ModelDatabase"

train_and_test_lsh(NUM_TRAIN, NUM_TEST, ROOT_DIR)

