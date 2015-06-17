"""
Computes the statistical distribution of stable poses for a polyhedron
Author: Nikhil Sharma
"""

import math
import sys
import numpy as np

import mesh
import obj_file

# A function for computing the statistical distribution of stable poses of a polyhedron.
def compute_stable_poses(mesh):
    """
    Computes convex hull of the input mesh and returns stable final faces along with 
    corresponding probabilities.

    mesh -- a 3D Mesh object
    """
    convex_hull, cm = mesh.convex_hull(), mesh.vertex_mean_
    print(mesh.vertices()
        )
    print(cm)

    # mapping each edge in the convex hull to the two faces it borders
    triangles, vertices, edge_to_faces, triangle_to_vertex = convex_hull.triangles(), convex_hull.vertices(), {}, {}
    for triangle in triangles:
        triangle_vertices = [vertices[i] for i in triangle]
        e1, e2, e3 = Segment(triangle_vertices[0], triangle_vertices[1]), Segment(triangle_vertices[0], triangle_vertices[2]), Segment(triangle_vertices[1], triangle_vertices[2])
        for edge in [e1, e2, e3]:
            p_1, p_2 = tuple(sorted(edge.endpoint_1)), tuple(sorted(edge.endpoint_2))
            k = (p_1, p_2) if p_1[0] > p_2[0] else (p_2, p_1)
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
        normal_vector, origin_point = np.cross(np.subtract(triangle_vertices[0], triangle_vertices[1]), np.subtract(triangle_vertices[0], triangle_vertices[2])), triangle_vertices[0]
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        proj_vector = np.subtract(cm, origin_point)
        dist = np.dot(normal_vector, proj_vector)
        proj_cm = np.subtract(cm, dist*normal_vector)

        # barycentric coordinates/minimum distance analysis (adapted from implementation provided at http://www.blackpawn.com/texts/pointinpoly/)
        v_0 = np.subtract(triangle_vertices[2], triangle_vertices[0])
        v_1 = np.subtract(triangle_vertices[1], triangle_vertices[0])
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
            closest_edge = closest_segment(proj_cm, [s1, s2, s3])

            p_1, p_2 = tuple(sorted(closest_edge.endpoint_1)), tuple(sorted(closest_edge.endpoint_2))
            k = (p_1, p_2) if p_1[0] > p_2[0] else (p_2, p_1)
            for face in edge_to_faces[k]:
                if list(face) != list(triangle):
                    topple_face = face
            predecessor, successor = triangle_to_vertex[tuple(triangle)], triangle_to_vertex[tuple(topple_face)]
            predecessor.add_edge(successor)

    # computes dictionary mapping faces to probabilities
    probabilities = {}
    for vertex in triangle_to_vertex.values():
        if not vertex.has_parent:
            probabilities.update(propogate_probabilities(vertex))

    # probability normalization
    prob_sum = 0
    for k, v in probabilities.items():
        if math.isnan(v):
            del probabilities[k]
        else:
            prob_sum += v
    norm_factor = 1.0/prob_sum

    for k in probabilities.keys():
        probabilities[k] *= norm_factor

    return probabilities


def compute_projected_area(vertices, cm):
    """
    Projects input vertices onto unit sphere and computes the area of the projection
    
    triangle_vertices -- list of three 3-element arrays; each array is a vertex coordinate of the triangle
    cm -- 3-element array representing the center of mass of the mesh being handled
    """
    angles, projected_vertices = [], [np.subtract(vertex, cm) / np.linalg.norm(vertex) for vertex in vertices]
    
    a = math.acos(np.dot(projected_vertices[0], projected_vertices[1]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[1])))
    b = math.acos(np.dot(projected_vertices[0], projected_vertices[2]) / (np.linalg.norm(projected_vertices[0]) * np.linalg.norm(projected_vertices[2])))
    c = math.acos(np.dot(projected_vertices[1], projected_vertices[2]) / (np.linalg.norm(projected_vertices[1]) * np.linalg.norm(projected_vertices[2])))
    s = (a + b + c) / 2

    return 4 * math.atan(math.sqrt(math.tan(s/2)*math.tan((s-a)/2)*math.tan((s-b)/2)*math.tan((s-c)/2)))
    

def propogate_probabilities(vertex):
    """
    Traverses graph of faces with depth first search, setting all non-sink
    probabilities to 0 and all sink probabilities to the sum of all parent
    vertices from vertex to sink vertex

    vertex -- starting vertex from which graph is to be traversed
    prob_sum -- current probability sum
    prob_mapping -- mapping of sink vertices to their propogated probabilities
    """
    prob_sum, prob_mapping = vertex.probability, {}
    while not vertex.is_sink:
        vertex = vertex.child
        if not vertex.marked:
            vertex.marked = True
            prob_sum += vertex.probability
    if tuple(vertex.face) in prob_mapping.keys():
        prob_mapping[tuple(vertex.face)] += prob_sum
    else:
        prob_mapping[tuple(vertex.face)] = prob_sum
    return prob_mapping


def closest_segment(point, line_segments):
    """
    Returns the finite line segment the least distance from the input point

    point -- A 3-element array-like data structure containing x,y,z coordinates
    line_segments -- An array of Segments
    """
    min_dist = sys.maxint
    for segment in line_segments:
        proj_point = project(point, segment)
        if on_line_segment(proj_point, segment):
            dist = distance_between(point, proj_point)
        else:
            min_endpoint_distance = min(distance_between(proj_point, segment.endpoint_1), distance_between(proj_point, segment.endpoint_2))
            dist = math.sqrt((distance_between(point, proj_point))**2 + (min_endpoint_distance)**2)

        if dist < min_dist:
            min_dist, min_seg = dist, segment
    return min_seg


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
    vector_segment = np.subtract(segment.endpoint_1, segment.endpoint_2)
    scale_factor = (np.dot(vector_segment, np.subtract(point, segment.endpoint_1)) / np.dot(vector_segment, vector_segment))
    return np.add(segment.endpoint_1, scale_factor*vector_segment)


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

    def __init__(self, probability, face, child=None):
        """
        Create a Vertex with given probability and children
        
        probability -- float probability of toppling
        children -- array of vertices pointed at by this vertex
        """
        self.probability = probability
        self.child = child
        self.face = face
        self.is_sink = True if not self.child else False
        self.has_parent = False
        self.num_parents = 0

    def add_edge(self, child):
        """
        Connects this vertex to the input child vertex

        child -- Vertex object
        """
        self.is_sink = False
        self.child = child
        child.has_parent = True
        child.num_parents += 1


##############
#TESTING AREA#
##############
# filename = "/Users/Nikhil/Desktop/UC Berkeley /GPIS/data/test/meshes/Co_clean.obj"
filename = "/Users/Nikhil/Desktop/pyramid.obj"
ob = obj_file.ObjFile(filename)
mesh = ob.read()
mesh.remove_unreferenced_vertices()
prob_mapping = compute_stable_poses(mesh)
print(prob_mapping)
print(sum([val for val in prob_mapping.values()]))
