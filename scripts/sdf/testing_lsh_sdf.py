import pickle
from engine_creation import train_and_test_lsh

NUM_TRAIN = 50
NUM_TEST = 3
ROOT_DIR = "datasets/Cat50_ModelDatabase"

SAVE_ENGINE = True
SAVE_LOCATION = "testing_lsh_storage.pk1"

accuracy, engine, results = train_and_test_lsh(NUM_TRAIN, NUM_TEST, ROOT_DIR)
if SAVE_ENGINE:
    with open(SAVE_LOCATION, 'wb') as output:
        pickle.dump(engine, output, pickle.HIGHEST_PROTOCOL)

print "Accuracy: %s"%(accuracy)

