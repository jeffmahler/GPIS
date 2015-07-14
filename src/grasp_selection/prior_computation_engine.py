import sys
sys.path.insert(0, gpis_root+'feature_vectors')
import nearest_features
from feature_object import FeatureObject
from feature_object import FeatureObjectDatabase
import pickle
import kernels
import database

class PriorComputationEngine:
	def __init__(nearest_features_path, feature_object_db_path, db, config):
		self.nearest_features = pickle.load(open(nearest_features_path, "rb"))
		self.feature_object_db = pickle.load(open(feature_object_db_path, "rb"))
		self.db = db
		self.neighbor_kernel = kernels.SquaredExponentialKernel(
        	sigma=config['neighbor_kernel_sigma'], l=config['neighbor_kernel_l'], phi=lambda x: x.feature_vector)

	def compute_priors(self, obj, candidates):
		self.feature_object = feature_object_db.get_feature_object_dict[key]
		self.neighbors = nearest_features.within_distance(feature_object, dist=DISTANCE)

		alpha_priors = []
		beta_priors = []
		for candidate in candidates:
			alpha = 0
			beta = 0
			for neighbor in neighbors:
				object_distance = self.neighbor_kernel.evaluate(candidate.obj, neighbor)
				grasps = self.db.grasps_for_key(neighbor.key)
				for neighbor_grasp in neighbor.grasps():
					grasp_distance = distance(candidate, neighbor_grasp)
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
