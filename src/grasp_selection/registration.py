from abc import ABCMeta, abstractmethod

import IPython
import logging
import numpy as np
import scipy.spatial.distance as ssd
import scipy.optimize as opt
import tfx

import feature_matcher as fm
import mesh
import obj_file as of
import similarity_tf as stf

try:
    import mayavi.mlab as mlab
except:
    logging.warning('Failed to import mayavi')
import scipy.spatial.distance as ssd

class RegistrationResult(object):
    def __init__(self, R, t, cost):
        self.R = R
        self.t = t
        self.cost = cost

def skew(xi):
    S = np.array([[0, -xi[2,0], xi[1,0]],
                  [xi[2,0], 0, -xi[0,0]],
                  [-xi[1,0], xi[0,0], 0]])
    return S

def get_closest(source_points, target_points):
    dists = ssd.cdist(source_points, target_points, 'euclidean')
    match_indices = np.argmin(dists, axis=1)
    return match_indices

def get_closest_plane(source_points, target_points, source_normals, target_normals, dist_thresh=0.05, norm_thresh=0.75):
    dists = ssd.cdist(source_points, target_points, 'euclidean')
    ip = source_normals.dot(target_normals.T) # abs because we don't have correct orientations
    source_ip = source_points.dot(target_normals.T)
    target_ip = target_points.dot(target_normals.T)
    target_ip = np.diag(target_ip)
    target_ip = np.tile(target_ip, [source_points.shape[0], 1])

    abs_diff = np.abs(source_ip - target_ip)

    invalid_dists = np.where(dists > dist_thresh)
    abs_diff[invalid_dists[0], invalid_dists[1]] = np.inf

    invalid_norms = np.where(ip < norm_thresh)
    abs_diff[invalid_norms[0], invalid_norms[1]] = np.inf

    match_indices = np.argmin(abs_diff, axis=1)
    match_vals = np.min(abs_diff, axis=1)
    invalid_matches = np.where(match_vals == np.inf)
    match_indices[invalid_matches[0]] = -1

    return match_indices

def vis_corrs(source_points, target_points, matches, source_normals=None, target_normals=None, plot_lines=False):
    mlab.figure()
    mlab.points3d(source_points[:,0], source_points[:,1], source_points[:,2], scale_factor=0.005, color=(1,0,0))
    mlab.points3d(target_points[:,0], target_points[:,1], target_points[:,2], scale_factor=0.005, color=(0,1,0))

    if plot_lines:
        num_source = source_points.shape[0]
        pair = np.zeros([2,3])
        for i in range(num_source):
            if matches[i] != -1:
                pair[0,:] = source_points[i,:]
                pair[1,:] = target_points[matches[i], :]
                mlab.plot3d(pair[:,0], pair[:,1], pair[:,2], color=(0,0,1), line_width=0.1, tube_radius=None)

    if source_normals is not None:
        t = 1e-2
        num_source = 200#source_points.shape[0]
        pair = np.zeros([2,3])
        for i in range(num_source):
            pair[0,:] = source_points[i,:]
            pair[1,:] = source_points[i,:] + t * source_normals[i,:]
            mlab.plot3d(pair[:,0], pair[:,1], pair[:,2], color=(0,0,1), line_width=0.1, tube_radius=None)        

    if target_normals is not None:
        t = 1e-2
        subsample_inds = np.random.choice(target_points.shape[0], size=200).tolist()
        pair = np.zeros([2,3])
        for i in subsample_inds:
            pair[0,:] = target_points[i,:]
            pair[1,:] = target_points[i,:] + t * target_normals[i,:]
            mlab.plot3d(pair[:,0], pair[:,1], pair[:,2], color=(0,0,1), line_width=0.1, tube_radius=None)        

    mlab.axes()
    mlab.show()

def icp_mesh_point_cloud(mesh, point_cloud, num_iterations):
    orig_source_points = np.array(mesh.vertices())
    target_points = point_cloud.T

    R_sol = np.eye(3)
    t_sol = np.zeros([3, 1])

    for i in range(num_iterations):
        # transform source points
        source_points = (R_sol.dot(orig_source_points.T) + np.tile(t_sol, [1, orig_source_points.shape[0]])).T

        # closest points
        match_indices = get_closest(source_points, target_points)

        vis_corrs(source_points, target_points, match_indices, plot_lines = False)
        
        # solve optimal rotation + translation
        target_corr_points = target_points[match_indices, :]
        source_centroid = np.mean(source_points, axis=0).reshape(1,3)
        target_centroid = np.mean(target_corr_points, axis=0).reshape(1,3)

        source_centroid_arr = np.tile(source_centroid, [source_points.shape[0], 1])
        target_centroid_arr = np.tile(target_centroid, [target_corr_points.shape[0], 1])
        H = (source_points - source_centroid_arr).T.dot(target_corr_points - target_centroid_arr)

        U, S, V = np.linalg.svd(H)
        R = V.T.dot(U.T)
        if np.linalg.det(R) < 0:
            V[2,:] = -1 * V[2,:]
            R = V.T.dot(U.T)
        t = -R.dot(source_centroid.T) + target_centroid.T

        R_sol = R.dot(R_sol)
        t_sol = R.dot(t_sol) + t

        
    vis_corrs(source_points, target_points, match_indices)

def point_plane_icp_mesh_point_cloud(mesh, point_cloud, mesh_normals, point_cloud_normals, num_iterations, alpha=0.98, mu=1e-2, sample_size=100, gamma=100.0):
    # setup the problem
    orig_source_points = np.array(mesh.vertices())
    target_points = point_cloud.T
    orig_source_normals = mesh_normals
    target_normals = point_cloud_normals.T

    normal_norms = np.linalg.norm(target_normals, axis=1)
    valid_inds = np.nonzero(normal_norms)
    target_points = target_points[valid_inds[0],:]
    target_normals = target_normals[valid_inds[0],:]

    normal_norms = np.linalg.norm(orig_source_normals, axis=1)
    valid_inds = np.nonzero(normal_norms)
    orig_source_points = orig_source_points[valid_inds[0],:]
    orig_source_normals = orig_source_normals[valid_inds[0],:]

    #vis_corrs(source_points, target_points, None, source_normals=source_normals, target_normals=target_normals, plot_lines=False)

    # alloc buffers
    source_mean_point = np.mean(orig_source_points, axis=0)
    target_mean_point = np.mean(target_points, axis=0)
    R_sol = np.eye(3)
    t_sol = np.zeros([3, 1]) #init with diff between means
    t_sol[:,0] = target_mean_point - source_mean_point

    # iterate
    for i in range(num_iterations):
        logging.info('Point to plane ICP iteration %d' %(i))

        # subsample points
        subsample_inds = np.random.choice(orig_source_points.shape[0], size=sample_size)
        source_points = orig_source_points[subsample_inds,:]
        source_normals = orig_source_normals[subsample_inds,:]

        # transform source points
        source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
        source_normals = (R_sol.dot(source_normals.T)).T
        
        # closest points
        match_indices = get_closest_plane(source_points, target_points, source_normals, target_normals)

        #vis_corrs(source_points, target_points, match_indices, source_normals=source_normals, target_normals=target_normals, plot_lines=True)
        #vis_corrs(source_points, target_points, match_indices, plot_lines=True)
        
        # solve optimal rotation + translation
        valid_corrs = np.where(match_indices != -1)[0]
        num_corrs = valid_corrs.shape[0]
        source_corr_points = source_points[valid_corrs,:]
        target_corr_points = target_points[match_indices[valid_corrs], :]
        target_corr_normals = target_normals[match_indices[valid_corrs], :]

        if num_corrs == 0:
            break

        # point to plane
        A = np.zeros([6,6])
        b = np.zeros([6,1])
        G = np.zeros([3,6])
        G[:,3:] = np.eye(3)

        Ap = np.zeros([6,6])
        bp = np.zeros([6,1])

        for i in range(num_corrs):
            s = source_corr_points[i:i+1,:].T
            t = target_corr_points[i:i+1,:].T
            n = target_corr_normals[i:i+1,:].T
            G[:,:3] = skew(s).T
            A += G.T.dot(n).dot(n.T).dot(G)
            b += G.T.dot(n).dot(n.T).dot(t - s)

            Ap += G.T.dot(G)
            bp += G.T.dot(t - s)

        v = np.linalg.solve(A+gamma*Ap+mu*np.eye(6), b+gamma*bp)

        R = np.eye(3)
        R = R + skew(v[:3])
        U, S, V = np.linalg.svd(R)
        R = U.dot(V)
        t = v[3:]

        R_sol = R.dot(R_sol)
        t_sol = R.dot(t_sol) + t

        # point to plane
        """
        A = np.zeros([3,3])
        b = np.zeros([3,1])
        G = np.zeros([3,3])
        G[:2,1:] = np.eye(2)

        Ap = np.zeros([3,3])
        bp = np.zeros([3,1])

        for i in range(num_corrs):
            s = source_corr_points[i:i+1,:].T
            t = target_corr_points[i:i+1,:].T
            n = target_corr_normals[i:i+1,:].T
            G[:,0] = skew(s).T[:,0]
            A += G.T.dot(n).dot(n.T).dot(G)
            b += G.T.dot(n).dot(n.T).dot(t - s)

            Ap += G.T.dot(G)
            bp += G.T.dot(t - s)

        v = np.linalg.solve(A+gamma*Ap+mu*np.eye(3), b+gamma*bp)

        R = np.eye(3)
        R[0,1] = -v[0]
        R[1,0] = v[0]
        U, S, V = np.linalg.svd(R)
        R_sol = U.dot(V)
        t_sol[0,0] = v[1]
        t_sol[1,0] = v[2]
        """
        
        # point dist
        """
        source_centroid = np.mean(source_corr_points, axis=0).reshape(1,3)
        target_centroid = np.mean(target_corr_points, axis=0).reshape(1,3)

        source_centroid_arr = np.tile(source_centroid, [source_corr_points.shape[0], 1])
        target_centroid_arr = np.tile(target_centroid, [target_corr_points.shape[0], 1])
        H = (source_corr_points - source_centroid_arr).T.dot(target_corr_points - target_centroid_arr)

        U, S, V = np.linalg.svd(H)
        R_p = V.T.dot(U.T)
        if np.linalg.det(R_p) < 0:
            V[2,:] = -1 * V[2,:]
            R_p = V.T.dot(U.T)
        t_p = -R_p.dot(source_centroid.T) + target_centroid.T

        # interpolate solutions
        R = alpha * R + (1-alpha) * R_p 
        U, S, V = np.linalg.svd(R)
        R = U.dot(V)
        t = alpha * t + (1-alpha) * t_p 

        R_sol = R.dot(R_sol)
        t_sol = R.dot(t_sol) + t
        """

    # get final cost
    source_points = (R_sol.dot(orig_source_points.T) + np.tile(t_sol, [1, orig_source_points.shape[0]])).T
    source_normals = (R_sol.dot(orig_source_normals.T)).T
    match_indices = get_closest_plane(source_points, target_points, source_normals, target_normals)
    valid_corrs = np.where(match_indices != -1)[0]
    num_corrs = valid_corrs.shape[0]
    if num_corrs == 0:
        return RegistrationResult(R_sol, t_sol, np.inf)

    source_corr_points = source_points[valid_corrs,:]
    target_corr_points = target_points[match_indices[valid_corrs], :]
    target_corr_normals = target_normals[match_indices[valid_corrs], :]
    
    source_target_alignment = np.diag((source_corr_points - target_corr_points).dot(target_corr_normals.T))
    point_plane_cost = (1.0 / num_corrs) * np.sum(source_target_alignment * source_target_alignment)
    point_dist_cost = (1.0 / num_corrs) * np.sum(np.linalg.norm(source_corr_points - target_corr_points, axis=1)**2)
    total_cost = point_plane_cost + gamma * point_dist_cost
    
    vis_corrs(source_points, target_points, match_indices, plot_lines=False)
    #print R_sol, t_sol
    return RegistrationResult(R_sol, t_sol, total_cost)

class IterativeRegistrationSolver:
    __metaclass__ = ABCMeta

    @abstractmethod
    def register(self, source, target, matcher, num_iterations=1):
        """ Iteratively register objects to one another """
        pass

class PointToPlaneICPSolver(IterativeRegistrationSolver):
    def __init__(self, sample_size=100, gamma=100.0, mu=1e-2):
        self.sample_size_ = sample_size
        self.gamma_ = gamma
        self.mu_ = mu
        IterativeRegistrationSolver.__init__(self)
    
    def register(self, orig_source_points, target_points, orig_source_normals, target_normals, matcher,
                 num_iterations=1, compute_total_cost=True, vis=False):
        """
        Iteratively register objects to one another using a modified version of point to plane ICP.
        The cost func is actually PointToPlane_COST + gamma * PointToPoint_COST

        Params:
           source_points: (Nx3 array) source object points
           target_points: (Nx3 array) target object points
           source_normals: (Nx3 array) source object outward-pointing normals
           target_normals: (Nx3 array) target object outward-pointing normals
           matcher: (PointToPlaneFeatureMatcher) object to match the point sets
           num_iterations: (int) the number of iterations to run
        Returns:
           RegistrationResult object containing the source to target transformation
        """        
        # setup the problem
        normal_norms = np.linalg.norm(target_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        target_points = target_points[valid_inds[0],:]
        target_normals = target_normals[valid_inds[0],:]

        normal_norms = np.linalg.norm(orig_source_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_source_points = orig_source_points[valid_inds[0],:]
        orig_source_normals = orig_source_normals[valid_inds[0],:]

        # alloc buffers for solutions
        source_mean_point = np.mean(orig_source_points, axis=0)
        target_mean_point = np.mean(target_points, axis=0)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1]) #init with diff between means
        t_sol[:,0] = target_mean_point - source_mean_point

        # iterate through
        for i in range(num_iterations):
            logging.info('Point to plane ICP iteration %d' %(i))

            # subsample points
            subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.sample_size_)
            source_points = orig_source_points[subsample_inds,:]
            source_normals = orig_source_normals[subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T
        
            # closest points
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)

            # solve optimal rotation + translation
            valid_corrs = np.where(corrs.index_map != -1)[0]
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                break

            # create A and b matrices for Gauss-Newton step on joint cost function
            A = np.zeros([6,6])
            b = np.zeros([6,1])
            Ap = np.zeros([6,6])
            bp = np.zeros([6,1])
            G = np.zeros([3,6])
            G[:,3:] = np.eye(3)

            for i in range(num_corrs):
                s = source_corr_points[i:i+1,:].T
                t = target_corr_points[i:i+1,:].T
                n = target_corr_normals[i:i+1,:].T
                G[:,:3] = skew(s).T
                A += G.T.dot(n).dot(n.T).dot(G)
                b += G.T.dot(n).dot(n.T).dot(t - s)

                Ap += G.T.dot(G)
                bp += G.T.dot(t - s)
            v = np.linalg.solve(A + self.gamma_*Ap + self.mu_*np.eye(6),
                                b + self.gamma_*bp)

            # create pose values from the solution
            R = np.eye(3)
            R = R + skew(v[:3])
            U, S, V = np.linalg.svd(R)
            R = U.dot(V)
            t = v[3:]

            # incrementally update the final transform
            R_sol = R.dot(R_sol)
            t_sol = R.dot(t_sol) + t

        total_cost = 0
        source_points = (R_sol.dot(orig_source_points.T) + np.tile(t_sol, [1, orig_source_points.shape[0]])).T
        source_normals = (R_sol.dot(orig_source_normals.T)).T

        if compute_total_cost:
            # rematch all points to get the final cost
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)
            valid_corrs = np.where(corrs.index_map != -1)[0]
            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                return RegistrationResult(R_sol, t_sol, np.inf)

            # get the corresponding points
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            # determine total cost
            source_target_alignment = np.diag((source_corr_points - target_corr_points).dot(target_corr_normals.T))
            point_plane_cost = (1.0 / num_corrs) * np.sum(source_target_alignment * source_target_alignment)
            point_dist_cost = (1.0 / num_corrs) * np.sum(np.linalg.norm(source_corr_points - target_corr_points, axis=1)**2)
            total_cost = point_plane_cost + self.gamma_ * point_dist_cost

        if vis:
            vis_corrs(source_points, target_points, corrs.index_map, plot_lines=False)        
        return RegistrationResult(R_sol, t_sol, total_cost)

    def register_2d(self, orig_source_points, target_points, orig_source_normals, target_normals, matcher,
                    num_iterations=1, compute_total_cost=True, vis=False):
        """
        Iteratively register objects to one another using a modified version of point to plane ICP
        which only solves for tx and ty (translation in the plane) and theta (rotation about the z axis).
        The cost func is actually PointToPlane_COST + gamma * PointToPoint_COST
        Points should be specified in the basis of the planar worksurface

        Params:
           source_points: (Nx3 array) source object points
           target_points: (Nx3 array) target object points
           source_normals: (Nx3 array) source object outward-pointing normals
           target_normals: (Nx3 array) target object outward-pointing normals
           matcher: (PointToPlaneFeatureMatcher) object to match the point sets
           num_iterations: (int) the number of iterations to run
        Returns:
           RegistrationResult object containing the source to target transformation
        """        
        # setup the problem
        logging.info('Setting up problem')
        normal_norms = np.linalg.norm(target_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        target_points = target_points[valid_inds[0],:]
        target_normals = target_normals[valid_inds[0],:]

        normal_norms = np.linalg.norm(orig_source_normals, axis=1)
        valid_inds = np.nonzero(normal_norms)
        orig_source_points = orig_source_points[valid_inds[0],:]
        orig_source_normals = orig_source_normals[valid_inds[0],:]

        # alloc buffers for solutions
        source_mean_point = np.mean(orig_source_points, axis=0)
        target_mean_point = np.mean(target_points, axis=0)
        R_sol = np.eye(3)
        t_sol = np.zeros([3, 1]) #init with diff between means
        t_sol[:,0] = target_mean_point - source_mean_point
        t_sol[2,0] = 0

        if vis:
            vis_corrs(orig_source_points, target_points, None, plot_lines=False)                    

        # iterate through
        for i in range(num_iterations):
            logging.info('Point to plane ICP iteration %d' %(i))

            # subsample points
            subsample_inds = np.random.choice(orig_source_points.shape[0], size=self.sample_size_)
            source_points = orig_source_points[subsample_inds,:]
            source_normals = orig_source_normals[subsample_inds,:]

            # transform source points
            source_points = (R_sol.dot(source_points.T) + np.tile(t_sol, [1, source_points.shape[0]])).T
            source_normals = (R_sol.dot(source_normals.T)).T
        
            # closest points
            corrs = matcher.match(source_points, target_points, source_normals, target_normals)

            # solve optimal rotation + translation
            valid_corrs = np.where(corrs.index_map != -1)[0]
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                break

            # create A and b matrices for Gauss-Newton step on joint cost function
            A = np.zeros([3,3])
            b = np.zeros([3,1])
            Ap = np.zeros([3,3])
            bp = np.zeros([3,1])
            G = np.zeros([3,3])
            G[:2,1:] = np.eye(2)

            for i in range(num_corrs):
                s = source_corr_points[i:i+1,:].T
                t = target_corr_points[i:i+1,:].T
                n = target_corr_normals[i:i+1,:].T
                G[0,0] = -s[1] 
                G[1,0] = s[0]
                A += G.T.dot(n).dot(n.T).dot(G)
                b += G.T.dot(n).dot(n.T).dot(t - s)

                Ap += G.T.dot(G)
                bp += G.T.dot(t - s)
            v = np.linalg.solve(A + self.gamma_*Ap + self.mu_*np.eye(3),
                                b + self.gamma_*bp)

            # create pose values from the solution
            R = np.eye(3)
            R = R + skew(np.array([[0],[0],[v[0,0]]]))
            U, S, V = np.linalg.svd(R)
            R = U.dot(V)
            t = np.array([[v[1,0]], [v[2,0]], [0]])

            # incrementally update the final transform
            R_sol = R.dot(R_sol)
            t_sol = R.dot(t_sol) + t

        total_cost = 0
        if compute_total_cost:
            # rematch all points to get the final cost
            source_points = (R_sol.dot(orig_source_points.T) + np.tile(t_sol, [1, orig_source_points.shape[0]])).T
            source_normals = (R_sol.dot(orig_source_normals.T)).T

            corrs = matcher.match(source_points, target_points, source_normals, target_normals)
            valid_corrs = np.where(corrs.index_map != -1)[0]
            num_corrs = valid_corrs.shape[0]
            if num_corrs == 0:
                return RegistrationResult(R_sol, t_sol, np.inf)

            # get the corresponding points
            source_corr_points = corrs.source_points[valid_corrs,:]
            target_corr_points = corrs.target_points[corrs.index_map[valid_corrs], :]
            target_corr_normals = corrs.target_normals[corrs.index_map[valid_corrs], :]

            # determine total cost
            source_target_alignment = np.diag((source_corr_points - target_corr_points).dot(target_corr_normals.T))
            point_plane_cost = (1.0 / num_corrs) * np.sum(source_target_alignment * source_target_alignment)
            point_dist_cost = (1.0 / num_corrs) * np.sum(np.linalg.norm(source_corr_points - target_corr_points, axis=1)**2)
            total_cost = point_plane_cost + self.gamma_ * point_dist_cost

        if vis:
            vis_corrs(source_points, target_points, corrs.index_map, plot_lines=False)        
        return RegistrationResult(R_sol, t_sol, total_cost)

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
