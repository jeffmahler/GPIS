from __future__ import print_function

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import time

import contacts
import database as db
from quality import PointGraspMetrics3D

import grasp as g, quality as q

FRICTION_COEF = 0.5

DEBUG = False
def debug(*args):
    if DEBUG:
        print(*args)

def force_closure_nguyen(contacts):
    """Compute force closure for two contacts according to Nguyen 1988."""
    c1, c2 = contacts
    p1, p2 = c1.point, c2.point
    n1, n2 = -c1.normal, -c2.normal # inward facing normals

    if (p1 == p2).all(): # same point
        return False

    for p in [p1, p2]:
        as_grid = graspable.sdf.transform_pt_obj_to_grid(p)
        on_surface, _ = graspable.sdf.on_surface(as_grid)
        if not on_surface:
            return False

    for normal, contact, other_contact in [(n1, p1, p2), (n2, p2, p1)]:
        normal = normal / np.linalg.norm(normal)
        diff = other_contact - contact

        if normal.dot(diff) < 0: # wrong side!
            return False

        # dist_normal = np.sqrt(np.linalg.norm(diff)**2 - dist_tangent_plane**2)
        # if dist_normal / dist_tangent_plane > FRICTION_COEF:
        alpha = np.arccos(normal.dot(diff) / np.linalg.norm(diff))
        if alpha > np.arctan(FRICTION_COEF):
            return False

    return True

MIN_NORMS = []
def force_closure_qp(contacts):
    """Compute force closure for two contacts."""
    forces = np.zeros([3, 0])
    torques = np.zeros([3, 0])
    normals = np.zeros([3, 0])

    for contact in contacts:
        c_in_normal = contact.in_direction_
        force_success, c_forces, _ = contact.friction_cone(8, FRICTION_COEF) # 8 cone faces
        if not force_success:
            continue
        torque_success, c_torques = contact.torques(c_forces)
        if not torque_success:
            continue

        forces = np.c_[forces, c_forces]
        torques = np.c_[torques, c_torques]
        normals = np.c_[normals, c_in_normal]

    G = PointGraspMetrics3D.grasp_matrix(forces, torques, normals, soft_fingers=True)
    try:
        min_norm = PointGraspMetrics3D.min_norm_vector_in_facet(G)
        MIN_NORMS.append(min_norm)
        return min_norm < 1e-2
    except:
        MIN_NORMS.append(10)
        return False

def benchmark(contacts, force_closure, n=10):
    contacts = contacts[:n]
    results = np.zeros((len(contacts), len(contacts)), dtype=np.dtype(bool))
    start = time.time()
    for i in range(len(contacts)):
        logging.error('%d / %d - %f (%d found)', i+1, len(contacts), time.time() - start, np.sum(results))
        for j in range(len(contacts)):
            # grasp = grasp_from_contacts(contacts[i], contacts[j])
            # in_force_closure = compute_quality(grasp, graspable)
            in_force_closure = force_closure([contacts[i], contacts[j]])
            results[i, j] = in_force_closure
    print('Time:', force_closure.__name__, '-', time.time() - start)
    return results

def load_all_contacts(graspable, max_contacts=5000):
    mesh = graspable.mesh
    vertex_contacts = []
    for i, (v, n) in enumerate(zip(mesh.vertices(), mesh.normals())):
        if i >= max_contacts:
            break
        c = contacts.Contact3D(graspable, np.array(v), -np.array(n))
        vertex_contacts.append(c)
    return vertex_contacts

# debugging stuff

def vis_axis_and_cone(c1, c2):
    import matplotlib.pyplot as plt
    graspable.sdf.scatter()

    to_grid = graspable.sdf.transform_pt_obj_to_grid
    c1_grid = to_grid(c1.point)
    c2_grid = to_grid(c2.point)
    diff_grid = c2_grid - c1_grid
    axis = g.ParallelJawPtGrasp3D.create_line_of_action(c1_grid, diff_grid / np.linalg.norm(diff_grid),
                                                        np.linalg.norm(diff_grid), graspable, 20, convert_grid=False)
    axis = np.array(axis)

    ax = plt.gca(projection='3d')
    ax.scatter(axis[:, 0], axis[:, 1], axis[:, 2], c='r')
    c1.plot_friction_cone(color='y')
    c2.plot_friction_cone(color='y')
    # plt.legend([
    #     c1.plot_friction_cone(color='b'),
    #     c2.plot_friction_cone(color='y'),
    # ], ['Cone 1', 'Cone 2'])
    plt.xlim(0, 50)
    plt.ylim(0, 50)
    plt.show()

def grasp_from_contacts(c1, c2):
    """Constructs a ParallelJawPtGrasp3D object from two contact points.
    Default width is 0.1, approx the PR2 gripper width.
    """
    grasp_center = 0.5 * (c1.point + c2.point)
    grasp_axis = c2.point - c1.point
    grasp_configuration = np.r_[grasp_center, grasp_axis, 0.1, 0.0, 0.0]
    return g.ParallelJawPtGrasp3D(grasp_configuration)

def compute_quality(grasp, obj):
    """Wrapper function to compute the ferrari canny metric."""
    return q.PointGraspMetrics3D.grasp_quality(grasp, obj, 'force_closure',
                                               soft_fingers=True, friction_coef=FRICTION_COEF)

def compare_hist(yes, no):
    bins = np.linspace(0, 1, 201)
    plt.hist(yes, bins, alpha=0.5, color='g', label='Nguyen FC')
    plt.hist(no, bins, alpha=0.5, color='r', label='Nguyen not FC')
    plt.ylim(0, 100)
    plt.show()

if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.ERROR)
    config = {
        'database_dir': '/home/jmahler/brian/uniform_meshes/test_db',
        'database_name': 'google_test_db.hdf5',
        'database_cache_dir': '.',
        'datasets': {
            'google': {'start_index': 0, 'end_index': 1}
        }
    }
    database_filename = os.path.join(config['database_dir'], config['database_name'])
    database = db.Hdf5Database(database_filename, config)
    dataset = database['google']
    graspable = dataset['actual-google-drill']

    all_contacts = load_all_contacts(graspable)
    print('Loaded.')

    def benchmark_with(n):
        nguyen_results = benchmark(all_contacts, force_closure_nguyen, n)
        qp_results = benchmark(all_contacts, force_closure_qp, n)
        print('n:', n**2)
        print('Nguyen:', np.sum(nguyen_results))
        print('QP:', np.sum(qp_results))
        print('Diff:', np.sum(nguyen_results ^ qp_results))
        return nguyen_results, qp_results

    MIN_NORMS = []
    n = 300
    nguyen, qp = benchmark_with(n)
    norms = np.array(MIN_NORMS).reshape((n, n))
    nguyen_yes = norms[nguyen]
    nguyen_no = norms[np.logical_not(nguyen)]
    nguyen_no = nguyen_no[nguyen_no < 10]
    compare_hist(nguyen_yes, nguyen_no)

    # print(force_closure_nguyen([all_contacts[19], all_contacts[39]]))
    # print(force_closure_qp([all_contacts[19], all_contacts[39]]))
    # vis_axis_and_cone(all_contacts[19], all_contacts[39])

    import IPython; IPython.embed()
