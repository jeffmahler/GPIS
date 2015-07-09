"""
Computes the statistical distribution of stable poses for a polyhedron
Author: Nikhil Sharma
"""
import math
import sys
import numpy as np
import Queue

import mesh
import obj_file
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

<<<<<<< HEAD
def compute_basis(vertices):
    """
    Computes axes for a transformed basis relative to the plane in which input vertices lies

    vertices -- a list of numpy arrays representing points in 3D space
    """
    centroid = compute_centroid(vertices)
    z_o = np.cross(np.subtract(centroid, vertices[0]), np.subtract(centroid, vertices[1]))
    z_o = z_o / np.linalg.norm(z_o)
    x_o = np.array([-z_o[1], z_o[0], 0])
    x_o = x_o / np.linalg.norm(x_o)
    y_o = np.cross(z_o, x_o)
    y_o = y_o / np.linalg.norm(y_o)

    R = np.array([np.transpose(x_o), np.transpose(y_o), np.transpose(z_o)])
    x_p, y_p, z_p = np.dot(R, x_o), np.dot(R, y_o), np.dot(R, z_o)
    return (x_p, y_p, z_p)

def compute_centroid(vertices):
    """
    Computes the centroid of input points

    vertices -- a list of numpy arrays representing points in 3D space    
    """
    centroid = []
    for i in range(len(vertices)):
        centroid.append(sum([vertex[i] for vertex in vertices]) / len(vertices))
    return np.array(centroid)
=======
import IPython
>>>>>>> 081c08b7ed867ca5dcfa4f5cba88e97d9c6754e8

# A function for computing the statistical distribution of stable poses of a polyhedron.
def compute_stable_poses(mesh):
    """
    Computes convex hull of the input mesh and returns stable final faces along with 
    corresponding probabilities.

    mesh -- a 3D Mesh object
    """
    convex_hull, cm = mesh.convex_hull(), mesh.vertex_mean_

    # mapping each edge in the convex hull to the two faces it borders
    triangles, vertices, edge_to_faces, triangle_to_vertex = convex_hull.triangles(), convex_hull.vertices(), {}, {}

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
        p_i = compute_projected_area(triangle_vertices, cm) / (4 * math.pi)
        v = Vertex(p_i, triangle)
        triangle_to_vertex[tuple(triangle)] = v

    # determining if landing on a given face implies toppling, and initializes a directed acyclic graph
    # a directed edge between two graph nodes implies that landing on one face will lead to toppling onto its successor
    # an outdegree of 0 for any graph node implies it is a sink (the object will come to rest if it topples to this face)
    for triangle in triangles:
        triangle_vertices = [vertices[i] for i in triangle]

        # computation of projection of cm onto plane of face
        v_0 = np.subtract(triangle_vertices[2], triangle_vertices[0])
        v_1 = np.subtract(triangle_vertices[1], triangle_vertices[0])

        v_0 = v_0 / np.linalg.norm(v_0)
        v_1 = v_1 / np.linalg.norm(v_1)
        normal_vector = np.cross(v_0, v_1)
        
        origin_point = np.array(triangle_vertices[0])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)

        proj_vector = np.subtract(cm, origin_point)
        dist = np.dot(normal_vector, proj_vector)
        proj_cm = np.subtract(cm, dist*normal_vector)

        other_dist = np.linalg.norm(cm - np.array(triangle_vertices[1]))

        # barycentric coordinates/minimum distance analysis (adapted from implementation provided at http://www.blackpawn.com/texts/pointinpoly/)
        v_2 = np.subtract(proj_cm, triangle_vertices[0])

        dot_00 = np.dot(v_0, v_0)
        dot_01 = np.dot(v_0, v_1)
        dot_02 = np.dot(v_0, v_2)
        dot_11 = np.dot(v_1, v_1)
        dot_12 = np.dot(v_1, v_2)

        inv_denom = 1.0 / (dot_00 * dot_11 - dot_01 * dot_01)
        u = (dot_11 * dot_02 - dot_01 * dot_12) * inv_denom
        v = (dot_00 * dot_12 - dot_01 * dot_02) * inv_denom

        proj_cm_in_triangle = (u >= 0) and (v >= 0) and (u + v < 1)

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
            predecessor, successor = triangle_to_vertex[tuple(triangle)], triangle_to_vertex[tuple(topple_face)]
            predecessor.add_edge(successor)


    # computes dictionary mapping faces to probabilities
    """
    top_vertices = []
    prob_sum = 0
    for vertex in triangle_to_vertex.values():
        prob_sum += vertex.probability
        if not vertex.has_parent:
            top_vertices.append(vertex)
    """
    probabilities = prop_probs_slow(triangle_to_vertex.values())

    return probabilities


def compute_projected_area(vertices, cm):
    """
    Projects input vertices onto unit sphere and computes the area of the projection
    
    triangle_vertices -- list of three 3-element arrays; each array is a vertex coordinate of the triangle
    cm -- 3-element array representing the center of mass of the mesh being handled
    """
    angles, projected_vertices = [], [np.subtract(vertex, cm) / np.linalg.norm(np.subtract(vertex, cm)) for vertex in vertices]

    a = math.acos(np.dot(projected_vertices[0], projected_vertices[1]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[1])))
    b = math.acos(np.dot(projected_vertices[0], projected_vertices[2]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[2])))
    c = math.acos(np.dot(projected_vertices[1], projected_vertices[2]) / (np.linalg.norm(projected_vertices[1]) * np.linalg.norm(projected_vertices[2])))
    s = (a + b + c) / 2

    return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*math.tan((s-b)/2)*math.tan((s-c)/2)))
    

def prop_probs_slow(vertices):
    """
    Pushes the probability of each vertex onto it sink by traversing the graph for each vertex. Slow, but works

    vertices -- list of all vertices in grasp
    prob_mapping -- mapping of sink vertices to their propogated probabilities
    """
    prob_mapping = {}
    for vertex in vertices:
        c = vertex
        while not c.is_sink:
            c = c.children[0]

        if not vertex.is_sink:
            c.probability += vertex.probability
        prob_mapping[tuple(c.face)] = c.probability

    return prob_mapping
        

def propagate_probabilities(top_vertices):
    """
    Traverses graph of faces with depth first search, setting all non-sink
    probabilities to 0 and all sink probabilities to the sum of all parent
    vertices from vertex to sink vertex

    vertex -- starting vertex from which graph is to be traversed
    prob_sum -- current probability sum
    prob_mapping -- mapping of sink vertices to their propogated probabilities

    NOTE: DOES NOT WORK. DO NOT USE
    """
    q, prob_mapping = Queue.Queue(), {}
    for vertex in top_vertices:
        q.put(vertex)

    while not q.empty():
        curr_vertex = q.get()

        if curr_vertex.is_sink:
            prob_mapping[tuple(curr_vertex.face)] = curr_vertex.probability
            curr_vertex.marked = True;

        else:
            num_children = len(curr_vertex.children)
            curr_vertex.probability /= num_children
            for child in curr_vertex.children:
                if curr_vertex in child.children:
                    print 'SMALL LOOP', curr_vertex.face

                if child.marked:
                    c = child
                    while not c.is_sink:
                        c = c.children[0]
                    c.probability += curr_vertex.probability
                else:
                    child.probability += curr_vertex.probability
                    q.put(child)
            curr_vertex.marked = True;

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

    def __init__(self, probability, face, chilren=[]):
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


##############
#TESTING AREA#
##############
filename = "data/test/meshes/Co_clean.obj"
#filename = "data/test/features/pepper_orig.obj"
ob = obj_file.ObjFile(filename)
mesh = ob.read()
mesh.remove_unreferenced_vertices()
prob_mapping = compute_stable_poses(mesh)

print(prob_mapping)
print 'Total sink sum', sum([val for val in prob_mapping.values()])


new_basis_axes = []
for face in prob_mapping.keys():
    vertices = [mesh.vertices()[i] for i in face]
    new_basis_axes.append(compute_basis(vertices))
print(new_basis_axes)
