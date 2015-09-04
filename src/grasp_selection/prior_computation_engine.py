import sys
sys.path.insert(0, 'src/grasp_selection/feature_vectors/')

from feature_database import FeatureDatabase
import kernels
import database
import feature_functions as ff
import feature_matcher as fm
import IPython
import registration as reg
import obj_file as of
import grasp_transfer as gt
import numpy as np
import time

class PriorComputationEngine:
	GRASP_TRANSFER_METHOD_NONE = 0
	GRASP_TRANSFER_METHOD_SHOT = 1
	GRASP_TRANSFER_METHOD_SCALE_XYZ = 2
	GRASP_TRANSFER_METHOD_SCALE_SINGLE = 3

	def __init__(self, db, config):
		self.feature_db = FeatureDatabase(config)
		self.db = db
		self.grasp_kernel = kernels.SquaredExponentialKernel(
			sigma=config['kernel_sigma'], l=config['kernel_l'])
		self.neighbor_kernel = kernels.SquaredExponentialKernel(
			sigma=1.0, l=(1/config['prior_neighbor_weight']))
		self.neighbor_distance = config['prior_neighbor_distance']
		self.num_neighbors = config['prior_num_neighbors']
		self.config = config
		self.kernel_tolerance = config['prior_kernel_tolerance']

	def compute_priors(self, obj, candidates, nearest_features_name=None, grasp_transfer_method=0):
		if nearest_features_name == None:
			nf = self.feature_db.nearest_features()
		else:
			nf = self.feature_db.nearest_features(name=nearest_features_name)
		feature_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[obj.key])
		neighbor_vector_dict = nf.k_nearest(feature_vector, k=self.num_neighbors) # nf.within_distance(feature_vector, dist=self.neighbor_distance)
		print 'Found %d neighbors!' % (len(neighbor_vector_dict))
		return self._compute_priors_with_neighbor_vectors(obj, feature_vector, candidates, neighbor_vector_dict, grasp_transfer_method=grasp_transfer_method)

	def _compute_priors_with_neighbor_vectors(self, obj, feature_vector, candidates, neighbor_vector_dict, grasp_transfer_method=0):
		print 'Loading features...'
		neighbor_features_dict = {}
		neighbor_grasps_dict = {}
		for neighbor_key in neighbor_vector_dict:
			if neighbor_key == obj.key:
				continue
			print 'Loading features for %s' % (neighbor_key)
			neighbor_obj = self.db[neighbor_key]
			grasps, neighbor_features = self._load_grasps_and_features(neighbor_obj)
			neighbor_features_dict[neighbor_key] = neighbor_features
			neighbor_grasps_dict[neighbor_key] = grasps
			
		print 'Finished loading features'

		if grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SHOT:
			reg_solver_dict = {}
			for neighbor_key in neighbor_vector_dict:
				reg_solver_dict[neighbor_key] = self._registration_solver(obj, neighbor_obj)
			self.reg_solver_dict = reg_solver_dict
		elif grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SCALE_XYZ:
			self._create_feature_scales_xyz(obj, neighbor_vector_dict.keys())
		elif grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SCALE_SINGLE:
			self._create_feature_scales_single(obj, neighbor_vector_dict.keys())

		print 'Creating priors...'
		prior_compute_start = time.clock()
		alpha_priors = []
		beta_priors = []
		for candidate in candidates:
			alpha = 1.0
			beta = 1.0
			for neighbor_key in neighbor_grasps_dict:
                                print 'Priors for neighbor', neighbor_key
				object_distance = self.neighbor_kernel.evaluate(feature_vector, neighbor_vector_dict[neighbor_key])
				neighbor_features = neighbor_features_dict[neighbor_key]
				grasps = neighbor_grasps_dict[neighbor_key]
				for neighbor_grasp, features in zip(grasps, neighbor_features):
					successes = neighbor_grasp.successes
					failures = neighbor_grasp.failures
					features = self._transfer_features(features, neighbor_grasp, neighbor_key, grasp_transfer_method)
					grasp_distance = object_distance * self.grasp_kernel(candidate.features, features.phi)
					if grasp_distance >= self.kernel_tolerance:
						alpha += grasp_distance * successes
						beta += grasp_distance * failures
			alpha_priors.append(alpha)
			beta_priors.append(beta)
		prior_compute_end = time.clock()
		print 'Finished creating priors. TIME: %.2f' % (prior_compute_end - prior_compute_start)

		return alpha_priors, beta_priors

	def compute_grasp_kernels(self, obj, candidates, nearest_features_name=None, grasp_transfer_method=0):
		if nearest_features_name == None:
			nf = self.feature_db.nearest_features()
		else:
			nf = self.feature_db.nearest_features(name=nearest_features_name)
		feature_vector = nf.project_feature_vector(self.feature_db.feature_vectors()[obj.key])
		neighbor_vector_dict = nf.k_nearest(feature_vector, k=self.num_neighbors)
		neighbor_keys = []
		all_neighbor_pfc_diffs = []
		all_neighbor_kernels = []
		all_distances = []
		object_indices = range(0, 28)
		for neighbor_key in neighbor_vector_dict.keys():
			if neighbor_key == obj.key:
				continue
			neighbor_obj = self.db[neighbor_key]
			neighbor_pfc_diffs, neighbor_kernels, object_distance = self.kernel_info_from_neighbor(obj, candidates, neighbor_obj, grasp_transfer_method=grasp_transfer_method)
			all_neighbor_pfc_diffs.append(neighbor_pfc_diffs)
			all_neighbor_kernels.append(neighbor_kernels)
			all_distances.append(object_distance)
			neighbor_keys.append(neighbor_key)
		return neighbor_keys, all_neighbor_kernels, all_neighbor_pfc_diffs, all_distances

	def kernel_info_from_neighbor(self, obj, candidates, neighbor, grasp_transfer_method=0):
		feature_vectors = self.feature_db.feature_vectors()
		# nf = self.feature_db.nearest_features()
		feature_vector = np.array(feature_vectors[obj.key]) # nf.project_feature_vector(feature_vectors[obj.key])
		neighbor_vector = np.array(feature_vectors[neighbor.key]) # nf.project_feature_vector(feature_vectors[neighbor.key])
		print 'Loading features for %s' % (neighbor.key)
		grasps, all_features = self._load_grasps_and_features(neighbor)
		print 'Finished loading features'

		reg_solver = self._registration_solver(obj, neighbor)
		print 'Creating priors...'
		prior_compute_start = time.clock()
		neighbor_pfc_diffs = []
		neighbor_kernels = []
		object_distance = self.neighbor_kernel.evaluate(feature_vector, neighbor_vector)
		for candidate in candidates:
			alpha = 1.0
			beta = 1.0
			for neighbor_grasp, features in zip(grasps, all_features):
				features = self._transfer_features(features, neighbor_grasp, neighbor.key, grasp_transfer_method)
				grasp_distance = self.grasp_kernel(candidate.features, features.phi)
				neighbor_pfc_diffs.append(abs(candidate.grasp.quality - neighbor_grasp.quality))
				neighbor_kernels.append(grasp_distance*object_distance)
		prior_compute_end = time.clock()
		print 'Finished creating priors. TIME: %.2f' % (prior_compute_end - prior_compute_start)

		return neighbor_pfc_diffs, neighbor_kernels, object_distance

	def _load_grasps_and_features(self, obj):
		grasps = self.db.load_grasps(obj.key)
		feature_loader = ff.GraspableFeatureLoader(obj, self.db.name, self.config)
		all_features = feature_loader.load_all_features(grasps)
		return grasps, all_features

	def _registration_solver(self, source_obj, neighbor_obj):
		feature_matcher = fm.RawDistanceFeatureMatcher()
		correspondences = feature_matcher.match(source_obj.features, neighbor_obj.features)
		reg_solver = reg.SimilaritytfSolver()
		reg_solver.register(correspondences)
		reg_solver.add_source_mesh(source_obj.mesh)
		reg_solver.scale(neighbor_obj.mesh)
		return reg_solver

	def _create_feature_scales_xyz(self, obj, neighbor_keys):
		scales = {}
		principal_dims = obj.mesh.principal_dims()
		for neighbor_key in neighbor_keys:
			scale_vector = principal_dims / self.db[neighbor_key].mesh.principal_dims()
			scales[neighbor_key] = scale_vector
		self.scales = scales

	def _create_feature_scales_single(self, obj, neighbor_keys):
		scales = {}
		normalized_scale = np.mean(np.linalg.norm(np.array(obj.mesh.vertices_), axis=0))
		for neighbor_key in neighbor_keys:
			neighbor_normalized_scale = np.mean(np.linalg.norm(np.array(self.db[neighbor_key].mesh.vertices_), axis=0))
			scales[neighbor_key] = normalized_scale / neighbor_normalized_scale
		self.scales = scales

	def _transfer_features(self, features, neighbor_grasp, neighbor_key, grasp_transfer_method):
		if grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SHOT:
			return self._transfer_features_shot(features, neighbor_grasp, self.reg_solver_dict[neighbor_key])
		elif grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SCALE_XYZ:
			return self._transfer_features_by_scale(features, neighbor_key)
		elif grasp_transfer_method == self.GRASP_TRANSFER_METHOD_SCALE_SINGLE:
			return self._transfer_features_by_scale(features, neighbor_key)
		return features

	def _transfer_features_shot(self, features, grasp, reg_solver):
		grasp = gt.transformgrasp(grasp, reg_solver)
		features.extractors_[2].center_ = grasp.center
		features.extractors_[3].axis_ = grasp.axis
		return features

	def _transfer_features_by_scale(self, features, neighbor_key):
		old_center = features.extractors_[2].center_
		features.extractors_[2].center_ = old_center * self.scales[neighbor_key]
		return features


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
