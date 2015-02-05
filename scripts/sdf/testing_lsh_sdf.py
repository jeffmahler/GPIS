from engine_creation import train_and_test_lsh

NUM_TRAIN = 50
NUM_TEST = 3
ROOT_DIR = "datasets/Cat50_ModelDatabase"

accuracy, engine, results = train_and_test_lsh(NUM_TRAIN, NUM_TEST, ROOT_DIR)

print "Accuracy: %s"%(accuracy)

