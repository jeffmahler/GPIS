import sys
sys.path.insert(0, 'src/grasp_selection/feature_vectors/')
import nearest_features
import feature_database
from feature_object import FeatureObject
from feature_object import FeatureObjectDatabase
import pickle
import kernels
import database

class PriorComputationEngine:
	def __init__(db, grasp_kernel, config, feature_db=None):
		if feature_db = None:
			self.feature_db = FeatureDatabase()
		else:
			self.feature_db = feature_db
		self.db = db
		self.grasp_kernel = grasp_kernel
		self.neighbor_kernel = kernels.SquaredExponentialKernel(
			sigma=1.0, l=config['neighbor_kernel_l'], phi=lambda x: x.feature_vector)

	def compute_priors(self, key, candidates):
		feature_vector = self.feature_db.feature_vectors[key]
		neighbor_vector_dict = self.nearest_features().within_distance(feature_vector, dist=DISTANCE)

		alpha_priors = []
		beta_priors = []
		for candidate in candidates:
			alpha = 0
			beta = 0
			for neighbor_key in neighbor_vector_dict:
				object_distance = self.neighbor_kernel.evaluate(feature_vector, neighbor_vector_dict[neighbor_key])
				grasps = self.db.load_grasps(neighbor_key)
				for neighbor_grasp in neighbor.grasps():
					grasp_distance = self.grasp_kernel(candidate, neighbor_grasp)
					alpha += object_distance * grasp_distance * neighbor_grasp.num_successes
					beta += object_distance * grasp_distance * neighbor_grasp.num_failures
			alpha_priors.append(alpha)
			beta_priors.append(beta)

		return alpha_priors, beta_priors


def test_prior_computation_engine():
	pce = PriorComputationEngine(nearest_features_path, feature_object_db_path, db, config)
	obj = ...
	candidates = ...
	alpha_priors, beta_priors = pce.compute_priors(obj, candidates)
	assert alpha_priors[0] == ... && beta_priors[0] == ...
	assert alpha_priors[1] == ... && beta_priors[1] == ...

if __name__ == '__main__':
	test_prior_computation_engine()
