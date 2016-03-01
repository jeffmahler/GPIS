"""
Computes the statistical distribution of stable poses for a polyhedron
Author: Nikhil Sharma
"""
import logging
try:
    import cvxopt as cvx
    cvx.solvers.options['show_progress'] = False
except:
    logging.warning('Failed to import cvx')
import math
import sys
import IPython
import numpy as np
# import similarity_tf as stf
# import tfx
try:
    import mayavi.mlab as mv
except:
    logging.warning('Failed to import mayavi')

import mesh
import obj_file
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

import sklearn.decomposition

# TODO: find a way to log output?


def compute_basis(vertices, m):
    """
    Computes axes for a transformed basis relative to the plane in which input vertices lies

    vertices -- a list of numpy arrays representing points in 3D space
    """
    vertex_array = np.array(m.vertices())
    centroid = compute_centroid(vertices)
    z_o = np.cross(np.subtract(vertices[1], vertices[0]), np.subtract(vertices[2], vertices[0]))
    z_o = z_o / np.linalg.norm(z_o)
    
    # ensure that all vertices are on the positive halfspace (aka above the table)
    dot_product = (vertex_array - centroid).dot(z_o)
    if dot_product[0] < 0:
        z_o = -z_o
    
    x_o = np.array([-z_o[1], z_o[0], 0])
    if np.linalg.norm(x_o) == 0.0:
        x_o = np.array([1, 0, 0])
    else:
        x_o = x_o / np.linalg.norm(x_o)
    y_o = np.cross(z_o, x_o)
    y_o = y_o / np.linalg.norm(y_o)

    R = np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])

    # rotate the vertices and then align along the principal axes
    rotated_vertices = R.dot(vertex_array.T)
    xy_components = rotated_vertices[:2,:].T

    pca = sklearn.decomposition.PCA(n_components = 2)
    pca.fit(xy_components)
    comp_array = pca.components_
    x_o = np.array([comp_array[0,0], comp_array[0,1], 0])

    x_o = R.T.dot(x_o)
    y_o = np.cross(z_o, x_o)
    R = np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])    

    """
    rotated_vertices = R.dot(vertex_array.T)
    plt.figure()
    ax = plt.gca(projection='3d')
    ax.scatter(rotated_vertices[0,:], rotated_vertices[1,:], rotated_vertices[2,:])
    plt.show()
    """
    return R

def compute_centroid(vertices):
    """
    Computes the centroid of input points

    vertices -- a list of numpy arrays representing points in 3D space    
    """
    centroid = []
    for i in range(len(vertices)):
        centroid.append(sum([vertex[i] for vertex in vertices]) / len(vertices))
    return np.array(centroid)

# A function for computing the statistical distribution of stable poses of a polyhedron.
def compute_stable_poses(mesh):
    """
    Computes convex hull of the input mesh and returns stable final faces along with 
    corresponding probabilities.

    mesh -- a 3D Mesh object
    """
    convex_hull = mesh.convex_hull()
    cm = mesh.center_of_mass

    # mapping each edge in the convex hull to the two faces it borders
    triangles, vertices, edge_to_faces, triangle_to_vertex = convex_hull.triangles(), convex_hull.vertices(), {}, {}

    max_tri = None
    max_area = 0
    i = 0

    for triangle in triangles:
        triangle_vertices = [vertices[i] for i in triangle]
        e1, e2, e3 = Segment(triangle_vertices[0], triangle_vertices[1]), Segment(triangle_vertices[0], triangle_vertices[2]), Segment(triangle_vertices[1], triangle_vertices[2])
        for edge in [e1, e2, e3]:
            # order vertices for consistent hashing
            p_1, p_2 = tuple(edge.endpoint_1), tuple(edge.endpoint_2)
            k = (p_1, p_2)
            if p_1[0] < p_2[0]:
                k = (p_2, p_1)
            elif p_1[0] == p_2[0] and p_1[1] < p_2[1]:
                k = (p_2, p_1)
            elif p_1[0] == p_2[0] and p_1[1] == p_2[1] and p_1[2] < p_2[2]:
                k = (p_2, p_1)                
 
            if k in edge_to_faces:
                edge_to_faces[k] += [triangle]
            else:
                edge_to_faces[k] = [triangle]

        v = np.array(triangle_vertices)
        n = np.cross(v[1,:] - v[0,:],  v[2,:] - v[0,:])
        n = n / np.linalg.norm(n)
        t1 = np.array([n[1], -n[0], 0])
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(t1, n)
        A = np.c_[t1, t2]
        x = v.dot(A)
        x = np.c_[x, np.ones(3)]
        area = abs(np.linalg.norm(x))
        if area > max_area:
            max_area = area
            max_tri = [triangle[0], triangle[1], triangle[2]]

        p_i = compute_projected_area(triangle_vertices, cm) / (4 * math.pi)
        vt = Vertex(p_i, triangle)
        triangle_to_vertex[tuple(triangle)] = vt

        """
        print p_i
        print triangle[0], triangle[1], triangle[2]
        if abs(n.dot(np.array([0,0,1]))) > 0.95:
            mv.figure()
            convex_hull.visualize()
            mv.points3d(cm[0], cm[1], cm[2], scale_factor=0.005, color=(1,0,0))
            mv.points3d(v[:,0], v[:,1], v[:,2], scale_factor=0.005)
            mv.axes()
            mv.show()
        """

    # determining if landing on a given face implies toppling, and initializes a directed acyclic graph
    # a directed edge between two graph nodes implies that landing on one face will lead to toppling onto its successor
    # an outdegree of 0 for any graph node implies it is a sink (the object will come to rest if it topples to this face)
    for triangle in triangles:
        triangle_vertices = [vertices[i] for i in triangle]

        # computation of projection of cm onto plane of face
        v_0 = np.subtract(triangle_vertices[2], triangle_vertices[0])
        v_1 = np.subtract(triangle_vertices[1], triangle_vertices[0])
        w = np.array(triangle_vertices).T

        v_0 = v_0 / np.linalg.norm(v_0)
        v_1 = v_1 / np.linalg.norm(v_1)
        normal_vector = np.cross(v_0, v_1)
        
        origin_point = np.array(triangle_vertices[0])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        proj_vector = np.subtract(cm, origin_point)
        dist = np.dot(normal_vector, proj_vector)
        proj_cm = np.subtract(cm, dist*normal_vector)

        # barycentric coordinates/minimum distance analysis (adapted from implementation provided at http://www.blackpawn.com/texts/pointinpoly/)
        dim = 3
        P = cvx.matrix(2 * w.T.dot(w))
        q = cvx.matrix(-2 * proj_cm.T.dot(w))
        G = cvx.matrix(-np.eye(dim))
        h = cvx.matrix(np.zeros(dim))
        A = cvx.matrix(np.ones((1, dim)))
        b = cvx.matrix(np.ones(1))
        sol = cvx.solvers.qp(P, q, G, h, A, b)

        u = np.array(sol['x'])
        norm_diff = np.sqrt(sol['primal objective'] + proj_cm.T.dot(proj_cm))
        p = w.dot(u)

        proj_cm_in_triangle = norm_diff < 1e-4
        
        """
        if triangle[0] == 807 and triangle[1] == 742 and triangle[2] == 739:
            mv.figure()
            convex_hull.visualize()
            mv.points3d(cm[0], cm[1], cm[2], scale_factor=0.005, color=(1,0,0))
            mv.points3d(proj_cm[0], proj_cm[1], proj_cm[2], scale_factor=0.005, color=(0,1,0))
            mv.points3d(w[0,:], w[1,:], w[2,:], scale_factor=0.005)
            mv.points3d(p[0], p[1], p[2], scale_factor=0.005, color=(0,0,1))
            mv.axes()
            mv.show()
            IPython.embed()
        """

        # update list of top vertices, add edges between vertices as needed
        if not proj_cm_in_triangle:
            s1, s2, s3 = Segment(triangle_vertices[0], triangle_vertices[1]), Segment(triangle_vertices[0], triangle_vertices[2]), Segment(triangle_vertices[1], triangle_vertices[2])
            closest_edges, common_endpoint = closest_segment(proj_cm, [s1, s2, s3])
            if len(closest_edges) > 1:
                index = triangle_vertices.index(common_endpoint)
                if index == len(triangle_vertices) - 1:
                    index = 0
                else:
                    index = index + 1
                closest_edge = Segment(common_endpoint, triangle_vertices[index])

            else:
                closest_edge = closest_edges[0]

            # order vertices for consistent hashing
            p_1, p_2 = tuple(closest_edge.endpoint_1), tuple(closest_edge.endpoint_2)
            k = (p_1, p_2)
            if p_1[0] < p_2[0]:
                k = (p_2, p_1)
            elif p_1[0] == p_2[0] and p_1[1] < p_2[1]:
                k = (p_2, p_1)
            elif p_1[0] == p_2[0] and p_1[1] == p_2[1] and p_1[2] < p_2[2]:
                k = (p_2, p_1)

            for face in edge_to_faces[k]:
                if list(face) != list(triangle):
                    topple_face = face
            #if triangle[0] == 807 and triangle[1] == 742 and triangle[2] == 739:
            #    IPython.embed()
            predecessor, successor = triangle_to_vertex[tuple(triangle)], triangle_to_vertex[tuple(topple_face)]
            predecessor.add_edge(successor)

    probabilities = propagate_probabilities(triangle_to_vertex.values())    

    return probabilities


def compute_projected_area(vertices, cm):
    """
    Projects input vertices onto unit sphere and computes the area of the projection
    
    triangle_vertices -- list of three 3-element arrays; each array is a vertex coordinate of the triangle
    cm -- 3-element array representing the center of mass of the mesh being handled
    """
    angles, projected_vertices = [], [np.subtract(vertex, cm) / np.linalg.norm(np.subtract(vertex, cm)) for vertex in vertices]

    a = math.acos(min(1, max(-1, np.dot(projected_vertices[0], projected_vertices[1]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[1])))))
    b = math.acos(min(1, max(-1, np.dot(projected_vertices[0], projected_vertices[2]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[2])))))
    c = math.acos(min(1, max(-1, np.dot(projected_vertices[1], projected_vertices[2]) / (np.linalg.norm(projected_vertices[1]) * np.linalg.norm(projected_vertices[2])))))
    s = (a + b + c) / 2

    try:
        return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*math.tan((s-b)/2)*math.tan((s-c)/2)))
    except:
        s = s + 0.001
        return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*math.tan((s-b)/2)*math.tan((s-c)/2)))
    
    

def propagate_probabilities(vertices):
    """
    Pushes the probability of each vertex onto it sink by traversing the graph for each vertex. Slow, but works

    vertices -- list of all vertices in grasp
    prob_mapping -- mapping of sink vertices to their propogated probabilities
    """
    prob_mapping = {}
    for vertex in vertices:
        c = vertex
        visited = []
        while not c.is_sink:
            if c in visited:
                break
            visited.append(c)
            c = c.children[0]

        if tuple(c.face) not in prob_mapping.keys():
            prob_mapping[tuple(c.face)] = 0.0
        prob_mapping[tuple(c.face)] += vertex.probability

    for vertex in vertices:
        if not vertex.is_sink:
            prob_mapping[tuple(vertex.face)] = 0
    return prob_mapping


def closest_segment(point, line_segments):
    """
    Returns the finite line segment the least distance from the input point

    point -- A 3-element array-like data structure containing x,y,z coordinates
    line_segments -- An array of Segments
    """
    min_dist, min_segs, distances, segments, common_endpoint = sys.maxint, [], [], [], None
    for segment in line_segments:
        proj_point = project(point, segment)
        if on_line_segment(proj_point, segment):
            dist = distance_between(point, proj_point)
        else:
            min_endpoint_distance = min(distance_between(proj_point, segment.endpoint_1), distance_between(proj_point, segment.endpoint_2))
            dist = math.sqrt((distance_between(point, proj_point))**2 + (min_endpoint_distance)**2)
        distances.append(dist)
        segments.append(segment)
        if dist < min_dist:
            min_dist = dist

    for i in range(len(distances)):
        if min_dist + 0.000001 >= distances[i]:
            min_segs.append(segments[i])
    if len(min_segs) > 1:
        for i in range(len(min_segs)):
            seg_1 = min_segs[i]
            for endpoint in [seg_1.endpoint_1, seg_1.endpoint_2]:
                for seg_2 in min_segs[i+1:]:
                    if endpoint in [seg_2.endpoint_1, seg_2.endpoint_2]:
                        common_endpoint = endpoint
                        break
    return min_segs, common_endpoint


def distance_between(point_1, point_2):
    """
    Returns the distance between the two input points.

    point_1 -- A 3-element array-like data structure containing x,y,z coordinates
    point_2 -- Another 3-element array-like data structure containing x,y,z coordinates
    """
    return math.sqrt((point_1[0] - point_2[0])**2 + (point_1[1] - point_2[1])**2 + (point_1[2] - point_2[2])**2)


def on_line_segment(point, segment):
    """
    Returns whether a point is on a line segment

    In this case, we assume that point is on the line containing the input segment
    
    point -- a 3-element array containing x,y,z coordinates
    segment -- a Segment object
    """
    endpoint_1, endpoint_2 = segment.endpoint_1, segment.endpoint_2
    if (point[0] >= endpoint_1[0] and point[0] <= endpoint_2[0]) or (point[0] <= endpoint_1[0] and point[0] >= endpoint_2[0]):
        if (point[1] >= endpoint_1[1] and point[1] <= endpoint_2[1]) or (point[1] <= endpoint_1[1] and point[1] >= endpoint_2[1]):
            if (point[2] >= endpoint_1[2] and point[2] <= endpoint_2[2]) or (point[2] <= endpoint_1[2] and point[2] >= endpoint_2[2]):
                return True
    return False


def project(point, segment):
    """
    Returns a 3-element array that is the projection of point onto the line containing segment
    
    point -- a 3-element array containing x,y,z coordinates
    segment -- a Segment object
    """
    x = np.linalg.norm(np.subtract(point, segment.endpoint_1))
    y = np.linalg.norm(np.subtract(point, segment.endpoint_2))
    z = np.linalg.norm(np.subtract(segment.endpoint_2, segment.endpoint_1))
    scale_factor = 0.5*(z - ((x**2 - y**2) / z))
    
    return np.add(segment.endpoint_2, scale_factor * (np.subtract(segment.endpoint_1, segment.endpoint_2) / np.linalg.norm(np.subtract(segment.endpoint_1, segment.endpoint_2))))


class Segment:
    """
    Object representation of a finite line segment in 3D space
    """

    def __init__(self, endpoint_1, endpoint_2):
        """
        Creates a Segment with given endpoints

        endpoint_1 -- a 3-element array containing x,y,z coordinates
        endpoint_2 -- a 3-element array containing x,y,z coordinates
        """
        self.endpoint_1 = endpoint_1
        self.endpoint_2 = endpoint_2


class Vertex:
    """
    Each face of the convex hull of a polyhedron is represented as a Vertex
    in a directed acyclic graph
    """
    marked = False

    def __init__(self, probability, face, children=[]):
        """
        Create a Vertex with given probability and children
        
        probability -- float probability of toppling
        children -- array of vertices pointed at by this vertex
        """
        self.probability = probability
        self.children = []
        self.parents = []
        self.face = face
        self.is_sink = True if not self.children else False
        self.has_parent = False
        self.num_parents = 0

    def add_edge(self, child):
        """
        Connects this vertex to the input child vertex

        child -- Vertex object
        """
        self.is_sink = False
        self.children.append(child)
        child.parents.append(self)
        child.has_parent = True
        child.num_parents += 1



def transform_meshes(path):
    """
    Transforms meshes in the directory given by path based on their highest probability stable poses.

    path -- path leading to directory containing meshes
    num_poses -- variable denoting number of desired poses per object
    """
    f = open()

    # write header
    f.write('###########################################################\n')
    f.write('# POSE TRANSFORM DATA file generated by UC Berkeley Automation Sciences Lab\n')
    f.write('#\n')
    
    min_prob = sys.argv[1]
    for filename in os.listdir(path):
        if filename[-4:] == ".obj":
            ob = obj_file.ObjFile(path + "/" + filename)
            mesh = ob.read()
            mesh.remove_unreferenced_vertices()
            prob_mapping, cv_hull = compute_stable_poses(mesh), mesh.convex_hull()

            R_list = []
            for face, p in prob_mapping.items():
                if p >= min_prob:
                    R_list.append([p, compute_basis([cv_hull.vertices()[i] for i in face])])

            # write necessary data to compute each pose (R and pose probability)
            for i in range(len(R_list)):
                f.write('# Pose ' + str(i + 1) + ':\n')
                f.write('# Probability: ' + str(R_list[i][0]) + '\n')
                for p in range(3):
                    f.write(str(R_list[i][1][p][0]) + ' ' + str(R_list[i][1][p][1]) + ' ' + str(R_list[i][1][p][2]) + '\n')
                f.write('#\n')

    f.write('###########################################################\n')
    f.write('\n')



