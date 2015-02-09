from engine_creation import train_and_test_lsh
import os
import IPython

NUM_TRAIN = 1100
NUM_TEST = 270
ROOT_DIR = "/mnt/terastation/shape_data/Cat50_ModelDatabase"
K = 5

for k in range(K,K+1):
    accuracy, engine, results = train_and_test_lsh(NUM_TRAIN, NUM_TEST, ROOT_DIR, k)
    print "Accuracy for K = %d: %f" %(k, accuracy)

