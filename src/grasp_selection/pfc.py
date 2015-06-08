import copy
import logging
import matplotlib.pyplot as plt
import mayavi.mlab as mv
import numpy as np
import time

import scipy.linalg
import scipy.stats
import sklearn.cluster

import antipodal_grasp_sampler as ags
import grasp as gr
import graspable_object as go
import obj_file
import quality as pgq
import sdf_file
import similarity_tf as stf
import tfx

import discrete_adaptive_samplers as das
import models
import objectives
import termination_conditions as tc

import IPython

# TODO: move this function somewhere interesting
def skew(xi):
    S = np.array([[0, -xi[2], xi[1]],
                  [xi[2], 0, -xi[0]],
                  [-xi[1], xi[0], 0]])
    return S
    
class GraspableObjectGaussianPose:
    def __init__(self, obj, config):
        self.obj_ = obj
        self._parse_config(config)

        self.s_rv_ = scipy.stats.norm(obj.tf.scale, self.sigma_scale_**2)
        self.t_rv_ = scipy.stats.multivariate_normal(obj.tf.translation, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        if self.num_prealloc_samples_ > 0:
            self._preallocate_samples()

    def _parse_config(self, config):
        # NOTE: scale sigma only for now
        self.sigma_rot_ = config['sigma_rot_obj']
        self.sigma_trans_ = config['sigma_trans_obj']
        self.sigma_scale_ = config['sigma_scale_obj']
        self.num_prealloc_samples_ = config['num_prealloc_obj_samples']

    def _preallocate_samples(self):
        """ Preallocate samples for faster adative sampling """
        self.prealloc_samples_ = []
        for i in range(self.num_prealloc_samples_):
            self.prealloc_samples_.append(self.sample())

    def sample(self, size=1):
        """ Sample |size| random variables from the model """
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)
            R = scipy.linalg.expm(S_xi).dot(self.obj_.tf.rotation)
            s = self.s_rv_.rvs(size=1)[0]
            t = self.t_rv_.rvs(size=1)
            sample_tf = stf.SimilarityTransform3D(tfx.transform(R.T, t), s)

            # transform object by pose
            obj_sample = self.obj_.transform(sample_tf)
            samples.append(obj_sample)

        # not a list if only 1 sample
        if size == 1:
            return samples[0]
        return samples        

    def rvs(self, size=1, iteration=1):
        """ Samples random variables """
        if self.num_prealloc_samples_ > 0:
            samples = []
            for i in range(size):
                samples.append(self.prealloc_samples_[(iteration + i) % self.num_prealloc_samples_])
            if size == 1:
                return samples[0]
            return samples        
        # generate a new sample
        return self.sample(size=size)

class ParallelJawGraspGaussian:
    def __init__(self, grasp, config):
        self.grasp_ = grasp
        self._parse_config(config)

        self.t_rv_ = scipy.stats.multivariate_normal(grasp.center, self.sigma_trans_**2)
        self.r_xi_rv_ = scipy.stats.multivariate_normal(np.zeros(3), self.sigma_rot_**2)

        if self.num_prealloc_samples_ > 0:
            self._preallocate_samples()

    def _parse_config(self, config):
        # NOTE: scale sigma only for now
        self.sigma_rot_ = config['sigma_rot_grasp']
        self.sigma_trans_ = config['sigma_trans_grasp']
        self.num_prealloc_samples_ = config['num_prealloc_grasp_samples']

    def _preallocate_samples(self):
        """ Preallocate samples for faster adative sampling """
        self.prealloc_samples_ = []
        for i in range(self.num_prealloc_samples_):
            self.prealloc_samples_.append(self.sample())
    
    @property
    def grasp(self):
        return self.grasp_

    def sample(self, size=1):
        samples = []
        for i in range(size):
            # sample random pose
            xi = self.r_xi_rv_.rvs(size=1)
            S_xi = skew(xi)
            v = scipy.linalg.expm(S_xi).dot(self.grasp_.axis)
            t = self.t_rv_.rvs(size=1)

            # transform object by pose
            #grasp_sample = copy.copy(self.grasp_)
            grasp_sample = gr.ParallelJawPtGrasp3D(t, v, self.grasp_.grasp_width)

            samples.append(grasp_sample)

        if size == 1:
            return samples[0]
        return samples

    def rvs(self, size=1, iteration=1):
        """ Samples |size| random variables """
        if self.num_prealloc_samples_ > 0:
            samples = []
            for i in range(size):
                samples.append(self.prealloc_samples_[(iteration + i) % self.num_prealloc_samples_])
            if size == 1:
                return samples[0]
            return samples        
        # generate a new sample
        return self.sample(size=size)

class ForceClosureRV:
    """ RV class for grasps in force closure on an object """
    def __init__(self, grasp_rv, obj_rv, friction_coef_rv, config):
        self.grasp_rv_ = grasp_rv
        self.obj_rv_ = obj_rv
        self.friction_coef_rv_ = friction_coef_rv # scipy stat rv

        self._parse_config(config)
        self.sample_count_ = 0

    def _parse_config(self, config):
        self.num_cone_faces_ = config['num_cone_faces']

    @property
    def grasp(self):
        return self.grasp_rv_.grasp

    def sample_success(self):
        # sample grasp
        cur_time = time.clock()
        grasp_sample = self.grasp_rv_.rvs(size=1, iteration=self.sample_count_)
        grasp_time = time.clock()

        # sample object
        obj_sample = self.obj_rv_.rvs(size=1, iteration=self.sample_count_)
        obj_time = time.clock()

        # sample friction cone
        friction_coef_sample = self.friction_coef_rv_.rvs(size=1)
        friction_time = time.clock()

        #logging.info('Grasp sample time %f'%(grasp_time - cur_time))
        #logging.info('Obj sample time %f'%(obj_time - grasp_time))
        #logging.info('Friction sample time %f'%(friction_time - obj_time))

        # compute force closure
        fc = pgq.PointGraspMetrics3D.grasp_quality(grasp_sample, obj_sample, "force_closure", friction_coef = friction_coef_sample,
                                                   num_cone_faces = self.num_cone_faces_, soft_fingers = True)
        self.sample_count_ = self.sample_count_ + 1
        return fc

def cartesian_to_spherical(x):
    v = x[0]**2 + x[1]**2
    r = np.sqrt(v + x[2]**2)               # r
    elev = np.arctan2(x[2], np.sqrt(v))     # theta
    az = np.arctan2(x[1], x[0])                           # phi
    return r, elev, az

def space_partition_grasps(grasps, config):
    """ Spatially partition grasps """
    # setup variables
    num_grasps = len(grasps)
    num_clusters = config['num_grasp_clusters']

    # no clustering if too few grasps
    if num_grasps < num_clusters:
        return [[grasp] for grasp in grasps]

    # create grasp features
    grasp_features = np.zeros([num_grasps, 5])
    i = 0
    for grasp in grasps:
        grasp_features[i, :3] = grasp.center

        r, elev, az = cartesian_to_spherical(grasp.axis)
        grasp_features[i, 3] = az
        grasp_features[i, 4] = elev
        i = i+1

    # whiten
    Sigma = np.cov(grasp_features.T)
    Sigma_sqrt = scipy.linalg.sqrtm(Sigma)
    grasp_features_whitened = np.linalg.inv(Sigma_sqrt).dot(grasp_features.T)
    
    # run kmeans
    km = sklearn.cluster.KMeans(n_clusters=num_clusters)
    labels = km.fit_predict(grasp_features_whitened.T)

    # partition grasps into bins
    labels_grasps = zip(labels, grasps)
    sorted_labels_grasps = sorted(labels_grasps, key=lambda g: g[0])
    grasp_bins = [[] for i in range(num_clusters)]
    for i in range(len(sorted_labels_grasps)):
        grasp_bins[sorted_labels_grasps[i][0]].append(sorted_labels_grasps[i][1])

    return grasp_bins

def plot_value_vs_time_beta_bernoulli(result, candidate_true_p, true_max=None, color='blue'):
    """ Plots the number of samples for each value in for a discrete adaptive sampler"""
    best_values = [candidate_true_p[m.best_pred_ind] for m in result.models]
    plt.plot(result.iters, best_values, color=color, linewidth=2)
    if true_max is not None: # also plot best possible
        plt.plot(result.iters, true_max*np.ones(len(result.iters)), color='green', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Success')

def test_antipodal_grasp_thompson():
    np.random.seed(100)

#    h = plt.figure()
#    ax = h.add_subplot(111, projection = '3d')

    # load object
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'
    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()

    mesh_name = 'data/test/meshes/Co_clean.obj'
    of = obj_file.ObjFile(mesh_name)
    m = of.read()

    graspable = go.GraspableObject3D(sdf_3d, mesh=m, model_name=mesh_name)

    config = {
        'grasp_width': 0.1,
        'friction_coef': 0.5,
        'num_cone_faces': 8,
        'grasp_samples_per_surface_point': 4,
        'dir_prior': 1.0,
        'alpha_thresh_div': 32,
        'rho_thresh': 0.75, # as pct of object max moment
        'vis_antipodal': False,
        'min_num_grasps': 20,
        'alpha_inc': 1.1,
        'rho_inc': 1.1,
        'sigma_mu': 0.1,
        'sigma_trans_grasp': 0.001,
        'sigma_rot_grasp': 0.1,
        'sigma_trans_obj': 0.001,
        'sigma_rot_obj': 0.1,
        'sigma_scale_obj': 0.1,
        'num_prealloc_obj_samples': 100,
        'num_prealloc_grasp_samples': 0,
        'min_num_collision_free_grasps': 10,
        'grasp_theta_res': 1
    }
    sampler = ags.AntipodalGraspSampler(config)

    start_time = time.clock()
    grasps, alpha_thresh, rho_thresh = sampler.generate_grasps(graspable, vis=False)
    end_time = time.clock()
    duration = end_time - start_time
    logging.info('Antipodal grasp candidate generation took %f sec' %(duration))

    # convert grasps to RVs for optimization
    graspable_rv = GraspableObjectGaussianPose(graspable, config)
    f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu'])
    candidates = []
    for grasp in grasps:
        grasp_rv = ParallelJawGraspGaussian(grasp, config)
        candidates.append(ForceClosureRV(grasp_rv, graspable_rv, f_rv, config))

    objective = objectives.RandomBinaryObjective()

    # run bandits
    eps = 5e-4
    ua_tc_list = [tc.MaxIterTerminationCondition(1000)]#, tc.ConfidenceTerminationCondition(eps)]
    ua = das.UniformAllocationMean(objective, candidates)
    ua_result = ua.solve(termination_condition = tc.OrTerminationCondition(ua_tc_list), snapshot_rate = 100)
    logging.info('Uniform allocation took %f sec' %(ua_result.total_time))

    ts_tc_list = [tc.MaxIterTerminationCondition(1000), tc.ConfidenceTerminationCondition(eps)]
    ts = das.ThompsonSampling(objective, candidates)
    ts_result = ts.solve(termination_condition = tc.OrTerminationCondition(ts_tc_list), snapshot_rate = 100)
    logging.info('Thompson sampling took %f sec' %(ts_result.total_time))

    true_means = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)

    # plot results
    plt.figure()
    plot_value_vs_time_beta_bernoulli(ua_result, true_means, color='red')
    plot_value_vs_time_beta_bernoulli(ts_result, true_means, color='blue')
    plt.show()

    das.plot_num_pulls_beta_bernoulli(ua_result)
    plt.title('Observations Per Variable for Uniform allocation')    

    das.plot_num_pulls_beta_bernoulli(ts_result)
    plt.title('Observations Per Variable for Thompson sampling')    

    plt.show()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    test_antipodal_grasp_thompson()
