"""
Computes the statistical distribution of stable poses for a polyhedron
Author: Nikhil Sharma
"""

import math
import sys
import numpy as np
import mesh as m

#	A function for computing the statistical distribution of stable poses of a polyhedron.
def compute_stable_poses(mesh):
	"""
	Computes convex hull of the input mesh and returns stable final faces along with 
	corresponding probabilities.

	mesh -- a 3D Mesh object
	"""
	convex_hull, cm = m.convex_hull(mesh), m.compute_centroid()

	# mapping each edge in the convex hull to the two faces it borders
	triangles, vertices, edge_to_faces, triangle_to_vertex, top_vertex = convex_hull.triangles(), convex_hull.vertices(), {}, {}, 0
	for triangle in triangles:
		triangle_vertices = [vertices[i] for i in triangle]
		e1, e2, e3 = Segment(triangle_vertices[0], triangle_vertices[1]), Segment(triangle_vertices[0], triangle_vertices[2]), Segment(triangle_vertices[1], triangle_vertices[2])
		for edge in [e1, e2, e3]:
			if (edge.endpoint_1, edge.endpoint_2) in edge_to_faces.keys():
				edge_to_faces[(edge.endpoint_1, edge.endpoint_2)].append(triangle)
			else:
				edge_to_faces[(edge.endpoint_1, edge.endpoint_2)] = [triangle]
		p_i = compute_projected_area(triangle_vertices, cm) / (4 * math.pi)
		v = Vertex(p_i, triangle)
		triangle_to_vertex[triangle] = v

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

		if not proj_cm_in_triangle:
			s1, s2, s3 = Segment(triangle_vertices[0], triangle_vertices[1]), Segment(triangle_vertices[0], triangle_vertices[2]), Segment(triangle_vertices[1], triangle_vertices[2])
			closest_edge = closest_segment(proj_cm, [s1, s2, s3])
			for face in edge_to_faces[(closest_edge.endpoint_1, closest_edge.endpoint_2)]:
				if list(face) != list(triangle):
					topple_face = face
			predecessor, successor = triangle_to_vertex[triangle], triangle_to_vertex[topple_face]
			predecessor.add_edge(successor)
			if top_vertex:
				if top_vertex is successor:
					top_vertex = predecessor
			else: 
				top_vertex = predecessor

	return dfs_sum_probs(top_vertex, 0, {})


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
	

def dfs_sum_probs(start_vertex, prob_sum, prob_mapping):
	"""
	Traverses graph of faces with depth first search, setting all non-sink
	probabilities to 0 and all sink probabilities to the sum of all parent
	vertices from start_vertex to sink vertex

	start_vertex -- starting vertex from which graph is to be traversed
	prob_sum -- current probability sum
	prob_mapping -- mapping of sink vertices to their propogated probabilities
	"""
	start_vertex.marked = True
	prob_sum = prob_sum + start_vertex.probability
	if not start_vertex.is_sink:
		start_vertex.probability = 0
		for child in start_vertex.children:
			if not child.marked:
				dfs_sum_probs(child, prob_sum, prob_mapping)
	else:
		prob_mapping[start_vertex.face] = prob_sum
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

	def __init__(self, probability, face, children=[]):
		"""
		Create a Vertex with given probability and children
		
		probability -- float probability of toppling
		children -- array of vertices pointed at by this vertex
		"""
		self.probability = probability
		self.children = children
		self.face = face
		self.is_sink = True if not self.children else False

	def add_edge(self, child):
		"""
		Connects this vertex to the input child vertex

		child -- Vertex object
		"""
		self.is_sink = False
		self.children = self.children + [child]


