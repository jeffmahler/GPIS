import numpy as np
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from os import walk,path
from sdf_class import SDF
from operator import itemgetter
import sklearn.decomposition as skdec

UNKNOWN_TAG = 'No Results'

class testing_suite:
"""
Class to test SDF files in a nearest neighbor lookup format, under different models of representation 
such as PCA, FactorAnalysis, KernelPCA with the rbf kernel, FastICA, and DictionaryLearning

Sample Usage:

	test=testing_suite()
	test.adddir("/mnt/terastation/shape_data/Cat50_ModelDatabase/screwdriver")
	num_train=12
	num_test=4

	test.make_train_test(num_train,num_test)
	accuracy,results=test.perform_PCA_tests()

"""

	def __init__(self):
		self.PCA_changed_=True
		self.FA_changed_=True
		self.KPCA_changed_=True
		self.FICA_changed_=True
		self.DL_changed_=True
		self.all_files_=[]
		self.PCA_=None
		self.FA_ = None
		self.KPCA_ = None
		self.FICA_ = None
		self.DL_ = None
		self.testing_=[]
		self.training_=[]
		self.engine_=[]
		self.training_vectors_=None
		self.confusion_={}
		self.biggest=0

	def adddir(self,dir_to_add):
		"""
			add all sdf filepaths from a root directory tree (dir_to_add) to the all_files_
			instance variable
		"""
		sdf_files = []
	    for root,dirs,files in walk(dir_to_add):
	        for file_ in files:
	            if file_.endswith(".sdf"):
	                sdf_files.append(path.join(root,file_))
	    self.all_files_+=sdf_files

	def addfile(self,file_to_add):
		"""add only one file to all_files"""
		self.all_files_.append(file_to_add)

	def make_train_test(self,num_train, num_test):
		"""
		populates the list of training files and testing files with filepaths based on a random
		number generator seeded with np.random.seed(100)

		Sample Usage:
			test=testing_suite()
			test.adddir("/mnt/terastation/shape_data/Cat50_ModelDatabase/screwdriver")
			num_train=12
			num_test=4

			test.make_train_test(num_train,num_test)
		"""
		assert num_train+num_test<=len(self.all_files_)
		np.random.seed(100)
	    permuted_indices = np.random.permutation(len(self.all_files_))
	    get_training = itemgetter(*permuted_indices[:num_train])
	    get_testing = itemgetter(*permuted_indices[num_train:num_train+num_test])
	    
	    if num_test > 1:
	        self.training_ = get_training(self.all_files_)
	    else:
	        self.training_= [get_training(self.all_files_)]


	    if num_test > 1:
	        self.testing_ = get_testing(self.all_files_)
	    else:
	        self.testing_ = [get_testing(self.all_files_)]

	def normalize_vector(self,vector,largest_dimension):
		"""normalizes smaller sdf vectors to a larger size by vertical stacking a column of zeros underneath"""
		return np.vstack((vector,np.zeros((largest_dimension-vector.shape[0],1))))

	def get_PCA_training_vectors(self):
		"""
		gets all training_vectors from the set of training files, normalizes them using normalize 
		vector and adds them all to a numpy array that gets returned
		"""
		training_sdf=[SDF(i) for i in list(self.training_)]
		biggest=0
		for item in training_sdf:
			biggest=max(biggest,item.dimensions[0])
		return_train_vectors=None
		for tempsdf in training_sdf:
			vectorized=numpy.reshape(tempsdf.data(),(tempsdf.dimensions()[0],1))
			normal_vector=self.normalize_vector(vectorized,biggest)
			if return_train_vectors==None:
				return_train_vectors=normal_vector
			else:
				return_train_vectors=np.concatenate((return_train_vectors,normal_vector),axis=1)
		return return_train_vectors

	"""
	-any function begining with make creates the sklearn.decomposition framework for the specified 
	decomposition type 
	-any function begining with fit fits the training vectors to the decomposition framework
	-any function begining with transform transforms the training vectors based on the fitted 
	decomposition framework
	"""

	def make_PCA(self,num_components=len(list(self.training_))):
		self.PCA_=skdec.PCA(n_components=num_components)

	def fit_PCA(self,training_vectors):
		self.PCA_.fit(training_vectors)

	def transform_PCA(self,training_vectors):
		return_vectors=self.PCA_.transform(training_vectors)
		return return_vectors

	def make_FA(self,num_components=len(list(self.training_))):
		self.FA_=skdec.FactorAnalysis(n_components=num_components)

	def fit_FA(self,training_vectors):
		self.FA_.fit(training_vectors)

	def transform_FA(self,training_vectors):
		return self.FA_.transform(training_vectors)

	def make_KPCA(self,num_components=len(list(self.training_))):
		self.KPCA_=skdec.KernelPCA(n_components=num_components,kernel="rbf")

	def fit_KPCA(self,training_vectors):
		self.KPCA_.fit(training_vectors)

	def transform_KPCA(self,training_vectors):
		return self.KPCA_.transform(training_vectors)

	def make_KPCA(self,num_components=len(list(self.training_))):
		self.FICA_=skdec.FastICA(n_components=num_components)

	def fit_KPCA(self,training_vectors):
		self.FICA_.fit(training_vectors)

	def transform_KPCA(self,training_vectors):
		return self.FICA_.transform(training_vectors)

	def make_DL(self,num_components=len(list(self.training_))):
		self.DL_=skdec.DictionaryLearning(n_components=num_components,alpha= [0.01, 0.1, 1, 10, 100, 1000])

	def fit_DL(self,training_vectors):
		self.DL_.fit(training_vectors)

	def transform_DL(self,training_vectors):
		return self.DL_.transform(training_vectors)

	def load_to_engine(self,vector_set):
	"""
	reinitializes our engine and loads a numpy set of vectors of dimension (self.biggest,1) 
	into self.engine_
	"""
		rbp = RandomBinaryProjections('rbp',10)
    	self.engine_ = Engine(self.biggest, lshashes=[rbp])
		for i in range(len(list(self.training_))):
			vector=vector_set[:,i]
			self.engine_.store_vector(vector,self.training_[i])

	def engine_query(self,test_vector):
	"""
	queries the engine with a (self.biggest,1) dimension vector and returns the file_names of nearest
	neighbors and the results
	"""
		results = self.engine.neighbours(test_vector)
        file_names = [i[1] for i in results]
        return file_names, results

    def setup_confusion(self):
    """
    reinitializes the self.confusion_ confusion matrix variable
    """
    	self.confusion_={}
    	self.confusion_[UNKNOWN_TAG] = {}
	    for file_ in self.all_files_:
	        category = cat50_file_category(file_)
	        self.confusion[category] = {}
	    for query_cat in self.confusion.keys():
	        for pred_cat in self.confusion.keys():
	            self.confusion[query_cat][pred_cat] = 0

	"""
	Makes a test vector by taking in an SDF, reshaping it, normalizing it, then returns a transformed
	version of that vector based on the corresponding decomposition model that was already trained
	"""

	def make_test_vector(self,sdf_array,vector_type):
		if vector_type=="PCA":
			return make_PCA_test_vector(sdf_array)
		elif vector_type=="FA":
			return make_FA_test_vector(sef_array)
		elif vector_type=="KPCA":
			return make_KPCA_test_vector(sdf_array)
		elif vector_type=="FICA":
			return make_FICA_test_vector(sdf_array)
		elif vector_type=="DL":
			return make_DL_test_vector(sdf_array)

	def make_DL_test_vector(self,sdf_array):
		reshaped=np.reshape(sdf_array.data(),(sdf_array.dimensions()[0],1))
		normalized=self.normalize_vector(reshaped,self.biggest)
		return self.DL_.transform(normalized)

	def make_FICA_test_vector(self,sdf_array):
		reshaped=np.reshape(sdf_array.data(),(sdf_array.dimensions()[0],1))
		normalized=self.normalize_vector(reshaped,self.biggest)
		return self.FICA_.transform(normalized)

	def make_KPCA_test_vector(self,sdf_array):
		reshaped=np.reshape(sdf_array.data(),(sdf_array.dimensions()[0],1))
		normalized=self.normalize_vector(reshaped,self.biggest)
		return self.KPCA_.transform(normalized)

	def make_FA_test_vector(self,sdf_array):
		reshaped=np.reshape(sdf_array.data(),(sdf_array.dimensions()[0],1))
		normalized=self.normalize_vector(reshaped,self.biggest)
		return self.FA_.transform(normalized)

	def make_PCA_test_vector(self,sdf_array):
		reshaped=np.reshape(sdf_array.data(),(sdf_array.dimensions()[0],1))
		normalized=self.normalize_vector(reshaped,self.biggest)
		return self.PCA_.transform(normalized)

	"""
	querys the loaded and trained engine with each of your test vectors from make_train_test
		Returns
	        accuracy: float representing the accuracy of querying the nearpy engine with the test results
	        test_results: dictionary of the results from the "testing" for each of the sdf_files 
	"""
	def perform_tests(self,K,test_type):
		for file_ in list(test_files):
			query_cat=cat50_file_category(file_)
			print "Querying: %s with category %s "%(file_, query_category)
			converted = SDF(file_)
			test_vector=make_test_vector(converted,test_type)
			closest_names, closest_vals=engine_query(test_vector)

			pred_category=UNKNOWN_TAG

			if len(closest_names)>0:
				closest_category=closest_names[0]
				pred_category=cat50_file_category(closest_category)

				for i in range(1,min(K,len(closest_names))):
					closest_category = closest_names[i]
	                potential_category = cat50_file_category(closest_category)

	                if potential_category == query_category:
	                    pred_category = potential_category
	        print "Result Category: %s"%(pred_category)

	        self.confusion[query_category][pred_category] += 1
        	test_results[file_]= [(closest_names, closest_vals)]

        row_names=self.confusion_.keys()
        confusion_mat=np.zeros([len(row_names),len(row_names)])
        i=0
        for query_cat in self.confusion.keys():
	        j = 0
	        for pred_cat in self.confusion.keys():
	            confusion_mat[i,j] = confusion[query_cat][pred_cat]
	            j += 1
	        i += 1

	    # get true positives, etc for each category
	    num_preds = len(test_files)
	    tp = np.diag(confusion_mat)
	    fp = np.sum(confusion_mat, axis=0) - np.diag(confusion_mat)
	    fn = np.sum(confusion_mat, axis=1) - np.diag(confusion_mat)
	    tn = num_preds * np.ones(tp.shape) - tp - fp - fn

	    # compute useful statistics
	    recall = tp / (tp + fn)
	    tnr = tn / (fp + tn)
	    precision = tp / (tp + fp)
	    npv = tn / (tn + fn)
	    fpr = fp / (fp + tn)
	    accuracy = np.sum(tp) / num_preds # correct predictions over entire dataset

	    # remove nans
	    recall[np.isnan(recall)] = 0
	    tnr[np.isnan(tnr)] = 0
	    precision[np.isnan(precision)] = 0
	    npv[np.isnan(npv)] = 0
	    fpr[np.isnan(fpr)] = 0

	    return accuracy, test_results


	"""
	runs perform_tests on a specific type of decomposition after creating that decomposition type 
	framework with the training vectors and loading those training vectors into the engine

	K is the number of neighbors to check
	"""
	def perform_PCA_tests(self,K):
		train_vectors=self.get_PCA_training_vectors()
		self.make_PCA()
		self.fit_PCA(train_vectors)
		train_vectors=self.transform_PCA(train_vectors)
		self.load_to_engine(train_vectors)
		self.setup_confusion()
		accuracy,test_results=self.perform_tests(K,"PCA")
		return accuracy,test_results

	def perform_FA_tests(self,K):
		train_vectors=self.get_PCA_training_vectors()
		self.make_FA()
		self.fit_FA(train_vectors)
		train_vectors=self.transform_FA(train_vectors)
		self.load_to_engine(train_vectors)
		self.setup_confusion()
		accuracy,test_results=self.perform_tests(K,"FA")
		return accuracy,test_results

	def perform_KPCA_tests(self,K):
		train_vectors=self.get_PCA_training_vectors()
		self.make_KPCA()
		self.fit_KPCA(train_vectors)
		train_vectors=self.transform_KPCA(train_vectors)
		self.load_to_engine(train_vectors)
		self.setup_confusion()
		accuracy,test_results=self.perform_tests(K,"KPCA")
		return accuracy,test_results


	def perform_FICA_tests(self,K):
		train_vectors=self.get_PCA_training_vectors()
		self.make_FICA()
		self.fit_FICA(train_vectors)
		train_vectors=self.transform_FICA(train_vectors)
		self.load_to_engine(train_vectors)
		self.setup_confusion()
		accuracy,test_results=self.perform_tests(K,"FICA")
		return accuracy,test_results

	def perform_DL_tests(self,K):
		train_vectors=self.get_PCA_training_vectors()
		self.make_DL()
		self.fit_DL()
		train_vectors=self.transform_DL(train_vectors)
		self.load_to_engine(training_vectors)
		self.setup_confusion()
		accuracy,test_results=self.perform_tests(K,"DL")
		return accuracy,test_results

	def get_engine(self):
		return self.engine_

	def get_PCA(self):
		return self.PCA_

	def get_FA(self):
		return self.FA_

	def get_KPCA(self):
		return self.KPCA_

	def get_FICA(self):
		return self.FICA_

	def get_DL(self):
		return self.DL_

def cat50_file_category(filename):
    """
    Returns the category associated with the full path of file |filename|
    """
    full_filename = path.abspath(filename)
    dirs, file_root = path.split(full_filename)
    head, category = path.split(dirs)
    return category