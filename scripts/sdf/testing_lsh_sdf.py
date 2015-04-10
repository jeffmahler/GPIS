import pickle
from engine_creation import train_and_test_lsh
import os
import sys
import IPython

#NUM_TRAIN = 1100
#NUM_TEST = 270
#ROOT_DIR = "/mnt/terastation/shape_data/Cat50_ModelDatabase"
NUM_TRAIN = 200
NUM_TEST = 30
ROOT_DIR = "datasets/Cat50_ModelDatabase"

K = 5
NUM_CLUSTERS = 20
SAVE_ENGINE = False
SAVE_LOCATION = "testing_lsh_storage.pk1"

model_count = 0
for root, dirs, files in os.walk(ROOT_DIR):
    for f in files:
        if f.endswith('.obj'):
            model_count += 1
print 'Num Models', model_count
exit()
for k in range(K,K+1):
    accuracy, engine, results = train_and_test_lsh(NUM_TRAIN, NUM_TEST, ROOT_DIR, k, NUM_CLUSTERS)
    print "Accuracy for K = %d: %f" %(k, accuracy)

if SAVE_ENGINE:
    with open(SAVE_LOCATION, 'wb') as output:
        pickle.dump(engine, output, pickle.HIGHEST_PROTOCOL)

print "Accuracy: %s"%(accuracy)

