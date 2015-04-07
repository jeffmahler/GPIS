import numpy as np
from operator import itemgetter

from random_functions import find_sdf
from SDF_bag_of_words import SDFBagOfWords

ROOT_DIR = "datasets/Cat50_ModelDatabase"
ENDS_WITH = "clean.sdf"

NUM_TRAIN = 15
NUM_TEST = 5
K = 10


sdf_files = find_sdf(ROOT_DIR, ENDS_WITH)
permuted_indices = np.random.permutation(len(sdf_files))
get_training = itemgetter(*permuted_indices[:NUM_TRAIN])
get_testing = itemgetter(*permuted_indices[NUM_TRAIN:NUM_TRAIN+NUM_TEST])

training = get_training(sdf_files)
testing = get_testing(sdf_files)

model = SDFBagOfWords()
predictions = model.fit(training, K)
answers = model.transform(testing)
print answers
print answers.shape

print predictions
print predictions.shape






