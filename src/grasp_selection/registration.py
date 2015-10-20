from abc import ABCMeta, abstractmethod

import IPython
import numpy as np
import scipy.spatial.distance as ssd
import scipy.optimize as opt
import obj_file as of
import similarity_tf as stf
import tfx
import mesh


class RegistrationFunc:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def register(self, correspondences):
        """ Register objects to one another """
        pass

class tpsRegistrationSolver(RegistrationFunc):
    def __init__(self):
        pass

    def __init__(self,bend_coef,rot_coef):
        self.lin_ag_=None
        self.bend_coef=bend_coef
        self.rot_coef=rot_coef

    def register(self,correspondences, weights=None, K_nn = None):
        self.source_points=correspondences.source_points
        self.target_points_=correspondences.target_points
        N,D = self.source_points.shape     

        K_nn = tps_kernel_matrix(self.source_points) if K_nn is None else K_nn
        coef_ratio = self.bend_coef / self.rot_coef if self.rot_coef > 0 else 0
        #if weights is None: reg_nn = self.bend_coef * np.eye(N)    
        #else: reg_nn = np.diag(self.bend_coef/(weights + 1e-6))
        #print weights
    
        A = np.zeros((N+D+1, N+D+1))
	    
        A[:N, :N] = K_nn

        A.flat[np.arange(0,N)*(N+D+2)] += self.bend_coef/(weights if weights is not None else 1)

        A[:N, N:N+D] = self.source_points
        A[:N, N+D] = 1

        A[N:N+D,:N] = self.source_points.T
        A[N+D,:N] = 1

        A[N:N+D, N:N+D] = coef_ratio*np.eye(D)

        B = np.empty((N+D+1, D))
        B[:N] = self.target_points_
        B[N:N+D] = coef_ratio*np.eye(D)
        B[N+D] = 0

        X = np.linalg.solve(A, B)
        self.w_ng_ = X[:N,:]
        self.lin_ag_ = X[N:N+D,:]
        self.trans_g_ = X[N+D,:]

    def transform(self,x):
        K=tps_kernel_matrix2(x,self.source_points)
        return np.transpose(np.dot(np.transpose(self.w_ng_),K)+np.dot(np.transpose(self.lin_ag_),np.transpose(x)))+self.trans_g_

class RigidRegistrationSolver(RegistrationFunc):
    def __init__(self):
        pass

    def register(self, correspondences, weights=None):
        """ Register objects to one another """
        # setup the problem
        self.source_points = correspondences.source_points
        self.target_points = correspondences.target_points
        N = correspondences.num_matches

        if weights is None:
            weights = np.ones([correspondences.num_matches, 1])
        if weights.shape[1] == 1:
            weights = np.tile(weights, (1, 3)) # tile to get to 3d space

        # calculate centroids (using weights)
        source_centroid = np.sum(weights * self.source_points, axis=0) / np.sum(weights, axis=0)
        target_centroid = np.sum(weights * self.target_points, axis=0) / np.sum(weights, axis=0)
        
        # center the datasets
        source_centered_points = self.source_points - np.tile(source_centroid, (N,1))
        target_centered_points = self.target_points - np.tile(target_centroid, (N,1))

        # find the covariance matrix and finding the SVD
        H = np.dot((weights * source_centered_points).T, weights * target_centered_points)
        U, S, V = np.linalg.svd(H) # this decomposes H = USV, so V is "V.T"

        # calculate the rotation
        R = np.dot(V.T, U.T)
        
        # special case (reflection)
        if np.linalg.det(R) < 0:
                V[2,:] *= -1
                R = np.dot(V.T, U.T)
        
        # calculate the translation + concatenate the rotation and translation
        t = np.matrix(np.dot(-R, source_centroid) + target_centroid)
        tf_source_target = np.hstack([R, t.T])
        self.R_=R
	self.t_=t
        self.source_centroid=source_centroid
	self.target_centroid=target_centroid

    def transform(self,x):
        return self.R_.dot(x.T)+self.t_.T


class SimilaritytfSolver(RigidRegistrationSolver):
    def __init__(self):
        super(self.__class__,self).__init__()

    def scale(self,target_mesh):
        a=self.get_a()
        self.target_mesh=target_mesh
        b=self.get_b()
        self.scale_=b/a
        self.source_mesh.vertices_=np.dot(self.scale_,self.source_mesh.vertices_)

    def add_source_mesh(self,meshname):
        self.source_mesh=meshname

    def perform_stf(self,meshtotransform):
        self.source_mesh=meshtotransform
        tftransform=tfx.canonical.pose(self.transform(self.source_points))
        similaritytf=stf.SimilarityTransform3D(tftransform)
        self.source_mesh=self.source_mesh.transform(similaritytf)

    def write(self, filepath):
        of.ObjFile(filepath).write(self.source_mesh)

    def get_a(self):
        toreturn=None
        for vertex in self.source_points:
            vi=self.R_.dot(vertex-self.source_centroid)
            if toreturn is None:
                toreturn=np.dot(vi.T,vi)
            else:
                toreturn=toreturn+np.dot(vi.T,vi)
        return toreturn
    def get_b(self):
        toreturn=None
        for i in range(len(self.source_points)):
            vi=self.R_.dot(self.source_points[i]-self.source_centroid)
            wi=self.target_points[i]-self.target_centroid
            if toreturn is None:
                toreturn=np.dot(vi.T,wi)
            else:
                toreturn=toreturn+np.dot(vi.T,wi)
        return toreturn

def tps_apply_kernel(distmat, dim):
    """
    if d=2: 
        k(r) = 4 * r^2 log(r)
       d=3:
        k(r) = -r
            
    import numpy as np, scipy.spatial.distance as ssd
    x = np.random.rand(100,2)
    d=ssd.squareform(ssd.pdist(x))
    print np.clip(np.linalg.eigvalsh( 4 * d**2 * log(d+1e-9) ),0,inf).mean()
    print np.clip(np.linalg.eigvalsh(-d),0,inf).mean()
    
    Note the actual coefficients (from http://www.geometrictools.com/Documentation/ThinPlateSplines.pdf)
    d=2: 1/(8*sqrt(pi)) = 0.070523697943469535
    d=3: gamma(-.5)/(16*pi**1.5) = -0.039284682964880184
    """

    if dim==2:       
        return 4 * distmat**2 * np.log(distmat+1e-20)
        
    elif dim ==3:
        return -distmat
    else:
        print 'dim =', dim
        raise NotImplementedError
    
def tps_kernel_matrix(x_na):
    dim = x_na.shape[1]
    distmat = ssd.squareform(ssd.pdist(x_na))
    return tps_apply_kernel(distmat,dim)
    
def tps_kernel_matrix2(XA,XB):
    dim = XA.shape[1]
    distmat = ssd.cdist(XA,XB)
    return tps_apply_kernel(distmat,2)
