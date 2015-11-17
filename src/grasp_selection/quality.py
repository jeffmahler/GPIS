import cvxopt as cvx
import numpy as np
import pyhull.convex_hull as cvh
import sys
import time

import grasp as g
import graspable_object as go
import obj_file
import sdf_file

import logging
import matplotlib.pyplot as plt

import IPython

class PointGraspMetrics3D:

    @staticmethod
    def grasp_quality(grasp, obj, method = 'force_closure', soft_fingers = False, friction_coef = 0.5, num_cone_faces = 8, params = None, vis=False):
        if not isinstance(grasp, g.PointGrasp):
            raise ValueError('Must provide a point grasp object')
        if not isinstance(obj, go.GraspableObject3D):
            raise ValueError('Must provide a 3D graspable object')
        if not hasattr(PointGraspMetrics3D, method):
            raise ValueError('Illegal point grasp metric specified')

        # get point grasp contacts
        contacts_found, contacts = grasp.close_fingers(obj, vis=vis)
        if not contacts_found:
            logging.debug('Contacts not found')
            return 0#-np.inf

        # add the forces, torques, etc at each contact point
        num_contacts = len(contacts)
        forces = np.zeros([3,0])
        torques = np.zeros([3,0])
        normals = np.zeros([3,0])
        for i in range(num_contacts):
            contact = contacts[i]
            if vis:
                contact.plot_friction_cone()

            # get contact forces
            force_success, contact_forces, contact_outward_normal = contact.friction_cone(num_cone_faces, friction_coef)

            if not force_success:
                logging.debug('Force computation failed')
                continue

            # get contact torques
            torque_success, contact_torques = contact.torques(contact_forces)
            if not torque_success:
                logging.debug('Torque computation failed')
                continue

            # get the magnitude of the normal force that the contacts could apply
            n = contact.normal_force_magnitude()

            forces = np.c_[forces, n * contact_forces]
            torques = np.c_[torques, n * contact_torques]
            normals = np.c_[normals, n * -contact_outward_normal] # store inward pointing normals

        if normals.shape[1] == 0:
            logging.debug('No normals')
            return 0#-np.inf

        if vis:
            ax = plt.gca()
            ax.set_xlim3d(0, obj.sdf.dims_[0])
            ax.set_ylim3d(0, obj.sdf.dims_[1])
            ax.set_zlim3d(0, obj.sdf.dims_[2])
            plt.show()

        # evaluate the desired quality metric
        Q_func = getattr(PointGraspMetrics3D, method)
        quality = Q_func(forces, torques, normals, soft_fingers, params)
        return quality

    @staticmethod
    def grasp_matrix(forces, torques, normals, soft_fingers=False, params = None):
        num_forces = forces.shape[1]
        num_torques = torques.shape[1]
        if num_forces != num_torques:
            raise ValueError('Need same number of forces and torques')

        num_cols = num_forces
        if soft_fingers:
            num_normals = 1
            if normals.ndim > 1:
                num_normals = normals.shape[1]
            num_cols = num_cols + num_normals

        G = np.zeros([6, num_cols])
        for i in range(num_forces):
            G[:3,i] = forces[:,i]
            G[3:,i] = torques[:,i]

        if soft_fingers:
            G[3:,-num_normals:] = normals
        return G

    @staticmethod
    def force_closure(forces, torques, normals, soft_fingers=False, params=None):
        """ Force closure """
        eps = 1e-2
        if params is not None:
            eps = params['eps']

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        min_norm = PointGraspMetrics3D.min_norm_vector_in_facet(G)
        return 1 * (min_norm < eps) # if greater than eps, 0 is outside of hull

    @staticmethod
    def partial_closure(forces, torques, normals, soft_fingers=False, params=None):
        """ Partial closure: whether or not the forces and torques can resist a specific wrench givien in the params"""
        force_limit = None
        if params is None:
            return 0
        force_limit = params['force_limits']
        target_wrench = params['target_wrench']        

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        return PointGraspMetrics.wrench_in_span(G, target_wrench, force_limit)

    @staticmethod
    def min_singular(forces, torques, normals, soft_fingers=False, params=None):
        """ Min singular value of grasp matrix - measure of wrench that grasp is "weakest" at resisting """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        min_sig = S[5]
        return min_sig

    @staticmethod
    def wrench_volume(forces, torques, normals, soft_fingers=False, params=None):
        """ Volume of grasp matrix singular values - score of all wrenches that the grasp can resist """
        k = 1
        if params is not None:
            k = params['k']

        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        sig = S
        return k * np.sqrt(np.prod(sig))

    @staticmethod
    def grasp_isotropy(forces, torques, normals, soft_fingers=False, params=None):
        """ Condition number of grasp matrix - ratio of "weakest" wrench that the grasp can exert to the "strongest" one """
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        _, S, _ = np.linalg.svd(G)
        max_sig = S[0]
        min_sig = S[5]
        isotropy = min_sig / max_sig
        if np.isnan(isotropy) or np.isinf(isotropy):
            return 0
        return isotropy

    @staticmethod
    def ferrari_canny_L1(forces, torques, normals, soft_fingers=False, params=None):
        """ The Ferrari-Canny L-1 metric """
        eps = 1e-2
        if params is not None:
            eps = params['eps']

        # create grasp matrix
        G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers)
        s = time.clock()
        hull = cvh.ConvexHull(G.T, joggle=True)
        e = time.clock()
        logging.debug('Convex hull took %f sec' %(e-s))

        if len(hull.vertices) == 0:
            logging.warning('Convex hull could not be computed')
            return -sys.float_info.max

        # determine whether or not zero is in the convex hull
        min_norm_in_hull = PointGraspMetrics3D.min_norm_vector_in_facet(G)

        # if norm is greater than 0 then forces are outside of hull
        if min_norm_in_hull > eps:
            return -min_norm_in_hull

        # find minimum norm vector across all facets of convex hull
        min_dist = sys.float_info.max
        for v in hull.vertices:
            if np.max(np.array(v)) < G.shape[1]: # because of some occasional odd behavior from pyhull
                facet = G[:, v]
                dist = PointGraspMetrics3D.min_norm_vector_in_facet(facet)
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    @staticmethod
    def wrench_in_span(W, target_wrench, f):
        """ Check whether wrench W can be resisted by forces and torques in G with limit force f """
        eps = 1e-4
        num_wrenches = W.shape[1]

        # quadratic and linear costs
        P = W.T.dot(W)
        q = -2 * target_wrench
        
        # inequalities 
        lam_geq_zero = -1 * np.eye(num_wrenches)
        force_constraint = np.ones(num_wrenches)
        G = np.c_[lam_geq_zero, force_constraint]
        h = zeros(num_wrenches+1)
        h[num_wrenches] = f

        sol = cvx.solvers.qp(P, q, G, h)        
        min_dist = sol['primal objective']
        return min_dist < eps

    @staticmethod
    def min_norm_vector_in_facet(facet):
        eps = 1e-2
        dim = facet.shape[1] # num vertices in facet

        # create alpha weights for vertices of facet
        G = facet.T.dot(facet)
        grasp_matrix = G + eps * np.eye(G.shape[0])

        # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
        P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
        q = cvx.matrix(np.zeros((dim, 1)))
        G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
        h = cvx.matrix(np.zeros((dim, 1)))
        A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
        b = cvx.matrix(np.ones(1))         # combinations of vertices

        sol = cvx.solvers.qp(P, q, G, h, A, b)

        min_norm = np.sqrt(sol['primal objective'])
        return abs(min_norm)

def test_gurobi_qp():
    import gurobipy as gb
    np.random.seed(100)
    dim = 20
    forces = 2 * (np.random.rand(3, dim) - 0.5)
    torques = 2 * (np.random.rand(3, dim) - 0.5)
    normal = 2 * (np.random.rand(3,1) - 0.5)
    G = PointGraspMetrics3D.grasp_matrix(forces, torques, normal)

    G = forces.T.dot(forces)
    m = gb.Model("qp")
    m.modelSense = gb.GRB.MINIMIZE
    alpha = [m.addVar(name="m"+str(v)) for v in range(dim)]
    alpha = np.array(alpha)
    m.update()

    obj = alpha.T.dot(G).dot(alpha)
    m.setObjective(obj)

    ones_v = np.ones(dim)
    cvx_const = ones_v.T.dot(alpha)
    m.addConstr(cvx_const, gb.GRB.EQUAL, 1.0, "c0")

    for i in range(dim):
        m.addConstr(alpha[i], gb.GRB.GREATER_EQUAL, 0.0)

    m.optimize()
    for v in m.getVars():
        print('Var {}: {}'.format(v.varName, v.x))
    print('Objective: {}'.format(obj.getValue()))

def test_cvxopt_qp():
    np.random.seed(100)
    dim = 20
    forces = 2 * (np.random.rand(3, dim) - 0.5)
    torques = 2 * (np.random.rand(3, dim) - 0.5)
    normal = 2 * (np.random.rand(3,1) - 0.5)
    # G = PointGraspMetrics3D.grasp_matrix(forces, torques, normal)
    grasp_matrix = forces.T.dot(forces) # not sure if this is a correct name...

    # Minimizes .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))
    b = cvx.matrix(np.ones(1))

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    for i, v in enumerate(sol['x']):
        print('Var m{}: {}'.format(i, v))
    print('Objective: {}'.format(sol['primal objective']))

def test_ferrari_canny_L1_synthetic():
    np.random.seed(100)
    dim = 20
    forces = 2 * (np.random.rand(3, dim) - 0.5)
    torques = 2 * (np.random.rand(3, dim) - 0.5)
    normal = 2 * (np.random.rand(3,1) - 0.5)

    start_time = time.clock()
    fc = PointGraspMetrics3D.ferrari_canny_L1(forces, torques, normal, soft_fingers=True)
    end_time = time.clock()
    fc_comp_time = end_time - start_time
    print 'FC Quality: %f' %(fc)
    print 'Computing FC took %f sec' %(fc_comp_time)

def test_quality_metrics(vis=True):
    np.random.seed(100)

    mesh_file_name = 'data/test/meshes/Co_clean.obj'
    sdf_3d_file_name = 'data/test/sdf/Co_clean.sdf'

    sf = sdf_file.SdfFile(sdf_3d_file_name)
    sdf_3d = sf.read()
    of = obj_file.ObjFile(mesh_file_name)
    mesh_3d = of.read()
    graspable = go.GraspableObject3D(sdf_3d, mesh = mesh_3d)

    z_vals = np.linspace(-0.025, 0.025, 3)
    for i in range(z_vals.shape[0]):
        print 'Evaluating grasp with z val %f' %(z_vals[i])
        grasp_center = np.array([0, 0, z_vals[i]])
        grasp_axis = np.array([0, 1, 0])
        grasp_width = 0.1
        grasp = g.ParallelJawPtGrasp3D(grasp_center, grasp_axis, grasp_width)

        qualities = []
        metrics = ['force_closure', 'min_singular', 'wrench_volume', 'grasp_isotropy', 'ferrari_canny_L1']
        for metric in metrics:
            q = PointGraspMetrics3D.grasp_quality(grasp, graspable, metric, soft_fingers=True)
            qualities.append(q)
            print 'Grasp quality according to %s: %f' %(metric, q)

        if vis:
            grasp.visualize(graspable)
            graspable.visualize()
            mv.show()


# TODO: find a way to log output?
cvx.solvers.options['show_progress'] = False

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    # test_gurobi_qp()
    test_cvxopt_qp()
    # test_ferrari_canny_L1_synthetic()
    # test_quality_metrics(vis=False)
