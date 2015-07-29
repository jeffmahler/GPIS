import sys
sys.path.insert(0, 'src/grasp_selection/feature_vectors/')

from feature_database import FeatureDatabase
import kernels
import database
import feature_functions as ff
import time

class PriorComputationEngine:
	def __init__(self, db, config, feature_db=None):
		if feature_db == None:
			self.feature_db = FeatureDatabase()
		else:
			self.feature_db = feature_db
		self.db = db
		self.grasp_kernel = kernels.SquaredExponentialKernel(
			sigma=config['kernel_sigma'], l=config['kernel_l'])
		self.neighbor_kernel = kernels.SquaredExponentialKernel(
			sigma=1.0, l=config['prior_neighbor_kernel_l'])
		self.neighbor_distance = config['prior_neighbor_distance']
		self.num_neighbors = config['prior_num_neighbors']
		self.config = config

	def compute_priors(self, key, candidates):
		nf = self.feature_db.nearest_features()
		feature_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[key])
		neighbor_vector_dict = nf.k_nearest(feature_vector, k=self.num_neighbors+1)#nf.within_distance(feature_vector, dist=self.neighbor_distance)
		# return neighbor_vector_dict
		print 'Found %d neighbors!' % (len(neighbor_vector_dict))
		print 'Loading features...'
		neighbor_features_for_key = {}
		neighbor_grasps_for_key = {}

		for neighbor_key in neighbor_vector_dict:
			if neighbor_key == key:
				continue
			print 'Loading features for %s' % (neighbor_key)
			grasps, all_features = self._load_grasps_and_features_for_key(neighbor_key)
			neighbor_features_for_key[neighbor_key] = all_features
			neighbor_grasps_for_key[neighbor_key] = grasps
		print 'Finished loading features'

		print 'Creating priors...'
		prior_compute_start = time.clock()
		alpha_priors = []
		beta_priors = []
		for candidate in candidates:
			alpha = 1
			beta = 1
			for neighbor_key in neighbor_grasps_for_key:
				object_distance = self.neighbor_kernel.evaluate(feature_vector, neighbor_vector_dict[neighbor_key])
				all_features = neighbor_features_for_key[neighbor_key]
				grasps = neighbor_grasps_for_key[neighbor_key]
				for neighbor_grasp, features in zip(grasps, all_features):
					grasp_distance = self.grasp_kernel(candidate.features, features.phi)
					alpha += object_distance * grasp_distance * neighbor_grasp.successes
					beta += object_distance * grasp_distance * neighbor_grasp.failures
			alpha_priors.append(alpha)
			beta_priors.append(beta)
		prior_compute_end = time.clock()
		print 'Finished creating priors. TIME: %.2f' % (prior_compute_end - prior_compute_start)

		return alpha_priors, beta_priors

	def compute_grasp_kernels(self, key, candidates):
		nf = self.feature_db.nearest_features()
		feature_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[key])
		neighbor_vector_dict = nf.k_nearest(feature_vector, k=self.num_neighbors+1)
		neighbor_keys = neighbor_vector_dict.keys()
		all_neighbor_pfc_diffs = []
		all_neighbor_kernels = []
		all_distances = []
		object_indices = range(0, 28)
		for neighbor_key in neighbor_keys:
			if neighbor_key == key:
				continue
			neighbor_pfc_diffs, neighbor_kernels, object_distance = self.kernel_info_from_neighbor(key, candidates, neighbor_key)
			all_neighbor_pfc_diffs.append(neighbor_pfc_diffs)
			all_neighbor_kernels.append(neighbor_kernels)
			all_distances.append(object_distance)
		return neighbor_keys, all_neighbor_kernels, all_neighbor_pfc_diffs, all_distances

	def kernel_info_from_neighbor(self, key, candidates, neighbor_key):
		nf = self.feature_db.nearest_features()
		feature_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[key])
		neighbor_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[neighbor_key])
		neighbor_vector_dict = {neighbor_key: neighbor_vector}
		print 'Loading features for %s' % (neighbor_key)
		grasps, all_features = self._load_grasps_and_features_for_key(neighbor_key)
		neighbor_features_for_key = {neighbor_key: all_features}
		neighbor_grasps_for_key = {neighbor_key: grasps}
		print 'Finished loading features'

		print 'Creating priors...'
		prior_compute_start = time.clock()
		alpha_priors = []
		beta_priors = []
		neighbor_pfc_diffs = []
		neighbor_kernels = []
		for candidate in candidates:
			alpha = 1
			beta = 1
			for neighbor_key in neighbor_grasps_for_key:
				object_distance = self.neighbor_kernel.evaluate(feature_vector, neighbor_vector_dict[neighbor_key])
				all_features = neighbor_features_for_key[neighbor_key]
				grasps = neighbor_grasps_for_key[neighbor_key]
				for neighbor_grasp, features in zip(grasps, all_features):
					grasp_distance = self.grasp_kernel(candidate.features, features.phi)
					neighbor_pfc_diffs.append(abs(candidate.grasp.quality - neighbor_grasp.quality))
					neighbor_kernels.append(grasp_distance)
			alpha_priors.append(alpha)
			beta_priors.append(beta)
		prior_compute_end = time.clock()
		print 'Finished creating priors. TIME: %.2f' % (prior_compute_end - prior_compute_start)

		return neighbor_pfc_diffs, neighbor_kernels, object_distance

	def _load_grasps_and_features_for_key(self, key):
		obj = self.db[key]
		grasps = self.db.load_grasps(key)
		feature_loader = ff.GraspableFeatureLoader(obj, self.db.name, self.config)
		all_features = feature_loader.load_all_features(grasps)
		return grasps, all_features


def test_prior_computation_engine():
	# TODO: fill in test
	# pce = PriorComputationEngine(nearest_features_path, feature_object_db_path, db, config)
	# obj = ...
	# candidates = ...
	# alpha_priors, beta_priors = pce.compute_priors(obj, candidates)
	# assert alpha_priors[0] == ... && beta_priors[0] == ...
	# assert alpha_priors[1] == ... && beta_priors[1] == ...
	pass

if __name__ == '__main__':
	test_prior_computation_engine()
