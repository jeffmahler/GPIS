import shot_feature_matching
import obj_file as of
import similarity_tf as stf
import feature_file as ffile
import feature_matcher as fm
import registration as reg
import os
import sdf_file as sf
import antipodal_grasp_sampler as ags
import termination_conditions as tc
import discrete_adaptive_samplers as das
import experiment_config as ec
import feature_functions as ff
import kernels
import models
import objectives
import pfc
import pr2_grasp_checker as pgc
import graspable_object as go
import sdf
import numpy as np
import experiment_config as ec
import grasp_sampler as gs
import scipy
import IPython
from grasp import *
from math import *
import time

class SHOTpipeline():
	def __init__(self):
		self.objectlist=None
	def loadobjectlist(self,objectlist):
		self.objectlist=objectlist
	def startpipeline(self):
		found_objects=getobjects(self.objectlist)
		tps_x=[]
		tps_y=[]
		rigid_x=[]
		rigid_y=[]
		line_x=[0,1]
		line_y=[0,1]
		for object in found_objects:
			a_features=get_feature(object)
			grasplist=getgrasps(object)
			try:
				grasp_probability=getprobability(object,grasplist)
				#IPython.embed()
				for other_obj in found_objects:
					if other_obj==object:
						continue
					print other_obj[0]," vs. ",object[0]
					b_features=get_feature(other_obj)
					feat_matcher = fm.RawDistanceFeatureMatcher()
					ab_corrs = feat_matcher.match(a_features, b_features)
					reg_solver2 = reg.SimilaritytfSolver()
					reg_solver2.register(ab_corrs)
					reg_solver2.add_source_mesh(of.ObjFile(object[1]).read())
					reg_solver2.scale(of.ObjFile(other_obj[1]).read())
					reg_solver = reg.tpsRegistrationSolver(1,1e-4)
					reg_solver.register(ab_corrs)


					grasplist_tf=[]
					grasplist_tps=[]
					for grasp in grasplist:
						targetgrasp=transformgrasp(grasp,reg_solver2)
						grasplist_tf.append(targetgrasp)
						grasplist_tps.append(transformgrasp(grasp,reg_solver))
					try:
						tps_probability=getprobability(other_obj,grasplist_tps)
						if len(tps_probability)<=len(grasp_probability):
							temp_tps=np.zeros(len(grasp_probability))
							for i,probability in enumerate(tps_probability):
								temp_tps[i]=probability
							tps_x.extend(grasp_probability.tolist())
							tps_y.extend(temp_tps.tolist())
						else:
							temp_og=np.zeros(len(tps_probability))
							for i,probability in enumerate(grasp_probability):
								temp_og[i]=probability
							tps_x.extend(temp_og.tolist())
							tps_y.extend(tps_probability.tolist())
					except ValueError:
						pass
					try:
						rigid_probability=getprobability(other_obj,grasplist_tf)
						if len(rigid_probability)<=len(grasp_probability):
							temp_tf=np.zeros(len(grasp_probability))
							for i,probability in enumerate(rigid_probability):
								temp_tf[i]=probability
							rigid_x.extend(grasp_probability.tolist())
							rigid_y.extend(temp_tf.tolist())
						else:
							temp_og=np.zeros(len(rigid_probability))
							for i,probability in enumerate(grasp_probability):
								temp_og[i]=probability
							rigid_x.extend(temp_og.tolist())
							rigid_y.extend(rigid_probability.tolist())
					except ValueError:
						pass
					#mv.show()
					#tempobj=of.ObjFile(other_obj[1]).read()
					#tempobj.add_sdf(sf.SdfFile(other_obj[2]).read())
					#tempobj.visualize()
					#grasplist_tf[0].visualize(tempobj)
					#time.sleep(200)
					#return None
			except ValueError:
				continue
		plt.title('PFC of Untransformed Grasp vs. PFC of Transformed Grasp')
		plt.subplot(121)
		plt.ylabel("Transformed Grasp PFC")
		plt.xlabel("Untransformed Grasp PFC")
		plt.plot(line_x,line_y)
		plt.scatter(tps_x,tps_y)
		plt.subplot(122)
		plt.xlabel("Untransformed Grasp PFC")
		plt.plot(line_x,line_y)
		plt.scatter(rigid_x,rigid_y)
		plt.savefig("grasp_transfer_results.png",dpi=100)
		plt.show()

def transformgrasp(grasp,reg_solver):
	if isinstance(reg_solver,reg.SimilaritytfSolver):
		graspmatrix=makeblockmatrix(grasp)
		transformmatrix=maketransformmatrix(reg_solver)
		newmatrix=transformmatrix.dot(graspmatrix)
		return makegrasp(grasp,bmat=newmatrix)
	else:
		graspmatrix=makeblockmatrix(grasp)
		position=getposition(graspmatrix)
		orientation=getdirection(graspmatrix)
		jacob_eval=make_eval_jacobian(position,reg_solver)
		new_direction=jacob_eval.dot(orientation)
		orthonormal_direction=make_orthonormal(new_direction)
		return makegrasp(grasp,center=position,direction=orthonormal_direction)

def make_orthonormal(matrix):
	U,s,v=np.linalg.svd(matrix)
	return U

def getposition(matrix):
	return matrix[:-1,3]
def getdirection(matrix):
	return matrix[:3,:3]

def getobjects(objectlist):
	toreturn=[]
	listedfiles=os.listdir("test_data/")
	for filename in listedfiles:
		if filename.find(".txt")!=-1:
			if "test_data-"+filename.replace("_features.txt",".obj") in listedfiles:
				toreturn.append(("test_data/"+filename,"test_data/test_data-"+filename.replace("_features.txt",".obj"),"test_data/test_data-"+filename.replace("_features.txt",".sdf")))
	return toreturn

def get_feature(object):
	feat_file = ffile.LocalFeatureFile(object[0])
	return feat_file.read()

def getgrasps(object):
	obj_name=object[1]
	sdf_name=object[2]
	obj_mesh=of.ObjFile(obj_name).read()
	sdf_=sf.SdfFile(sdf_name).read()
	obj=go.GraspableObject3D(sdf_,mesh=obj_mesh,key=object[0].replace("_features.txt",""),model_name=obj_name)
	config_name="cfg/correlated.yaml"
	config=ec.ExperimentConfig(config_name)
	np.random.seed(100)
	if config['grasp_sampler'] == 'antipodal':
		sampler = ags.AntipodalGraspSampler(config)
		grasps = sampler.generate_grasps(obj, check_collisions=config['check_collisions'], vis=False)
		num_grasps = len(grasps)
		min_num_grasps = config['min_num_grasps']
		if num_grasps < min_num_grasps:
			target_num_grasps = min_num_grasps - num_grasps
			gaussian_sampler = gs.GaussianGraspSampler(config)        
			gaussian_grasps = gaussian_sampler.generate_grasps(obj, target_num_grasps=target_num_grasps,
									check_collisions=config['check_collisions'], vis=False)
			grasps.extend(gaussian_grasps)
		
	else:
		sampler = gs.GaussianGraspSampler(config)        
		grasps = sampler.generate_grasps(obj, check_collisions=config['check_collisions'], vis=False,
			grasp_gen_mult = 6)	
	max_num_grasps = config['max_num_grasps']
	if len(grasps) > max_num_grasps:
		np.random.shuffle(grasps)
		grasps = grasps[:max_num_grasps]
	return grasps
def makeblockmatrix(grasp):
	return grasp.gripper_pose().array
def maketransformmatrix(solver):
	translation=solver.t_
	if translation.shape[0]!=3:
		translation=translation.T
	scale=solver.scale_
	rotation=solver.R_
	translation=np.array(translation)

	return np.concatenate((np.concatenate((rotation*scale,translation),axis=1),[[0,0,0,1]]),axis=0)

def makegrasp(originalgrasp,bmat=None,center=None,direction=None):
	if bmat is not None:
		axis=bmat[:,1][:-1]
		center=bmat[:,3][:-1]
		return ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(center,axis,originalgrasp.grasp_width,originalgrasp.approach_angle,originalgrasp.jaw_width))
	else:
		axis=direction[:,1]
		return ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(center,axis,originalgrasp.grasp_width,originalgrasp.approach_angle,originalgrasp.jaw_width))

def getprobability(object,grasps):
	obj_name=object[1]
	sdf_name=object[2]
	obj_mesh=of.ObjFile(obj_name).read()
	sdf_=sf.SdfFile(sdf_name).read()
	obj=go.GraspableObject3D(sdf_,mesh=obj_mesh,key=object[0].replace("_features.txt",""),model_name=obj_name)
	config_name="cfg/correlated.yaml"
	config=ec.ExperimentConfig(config_name)
	np.random.seed(100)
	
	brute_force_iter = config['bandit_brute_force_iter']
	max_iter = config['bandit_max_iter']
	confidence = config['bandit_confidence']
	snapshot_rate = config['bandit_snapshot_rate']
	tc_list = [
		tc.MaxIterTerminationCondition(max_iter),
#		tc.ConfidenceTerminationCondition(confidence)
	]
	
	
	# run bandits!
	graspable_rv = pfc.GraspableObjectGaussianPose(obj, config)
	f_rv = scipy.stats.norm(config['friction_coef'], config['sigma_mu']) # friction Gaussian RV

	# compute feature vectors for all grasps
	feature_extractor = ff.GraspableFeatureExtractor(obj, config)
	all_features = feature_extractor.compute_all_features(grasps)

	candidates = []
	for grasp, features in zip(grasps, all_features):
		grasp_rv = pfc.ParallelJawGraspGaussian(grasp, config)
		pfc_rv = pfc.ForceClosureRV(grasp_rv, graspable_rv, f_rv, config)
		if features is None:
			pass
		else:
			pfc_rv.set_features(features)
			candidates.append(pfc_rv)

	def phi(rv):
		return rv.features

	nn = kernels.KDTree(phi=phi)
	kernel = kernels.SquaredExponentialKernel(
		sigma=config['kernel_sigma'], l=config['kernel_l'], phi=phi)
	objective = objectives.RandomBinaryObjective()

	# uniform allocation for true values
	ua = das.UniformAllocationMean(objective, candidates)
	ua_result = ua.solve(termination_condition=tc.MaxIterTerminationCondition(brute_force_iter),
		snapshot_rate=snapshot_rate)
	estimated_pfc = models.BetaBernoulliModel.beta_mean(ua_result.models[-1].alphas, ua_result.models[-1].betas)
	return estimated_pfc

def make_eval_jacobian(position,reg_solver):
	bigarray=None
	for point in reg_solver.source_points:
		bottom_coef=(sqrt(np.square(position[0]-point[0])+np.square(position[1]-point[1])+np.square(position[2]-point[2])))
		temp_x=2*(position[0]-point[0])/bottom_coef
		temp_y=2*(position[1]-point[1])/bottom_coef
		temp_z=2*(position[2]-point[2])/bottom_coef
		temp_array=np.array([[temp_x],[temp_y],[temp_z]])
		if bigarray==None:
			bigarray=temp_array
		else:
			bigarray=np.hstack((bigarray,temp_array))
	return reg_solver.w_ng_.T.dot(bigarray.T)+reg_solver.lin_ag_.T

if __name__ == '__main__':
	SHOTpipeline().startpipeline()
		
