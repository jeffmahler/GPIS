class Mesh:
	"""A Mesh is a three-dimensional shape representation"""

	def __init__(self, vertices, triangles, normals=None, metadata=None):
		self.vertices_ = vertices
		self.triangles_ = triangles
		self.normals_ = normals
		self.metadata_ = metadata

	def get_vertices(self):
		return self.vertices_

	def get_triangles(self):
		return self.triangles_

	def get_normals(self):
		if self.normals_:
			return self.normals_
		return None #"Mesh does not have a list of normals."

	def get_metadata(self):
		if self.metadata_:
			return self.metadata_
		return "No metadata available."

	def set_vertices(self, vertices):
		self.vertices_ = vertices

	def set_triangles(self, triangles):
		self.triangles_ = triangles

	def set_normals(self, normals):
		self.normals_ = normals

	def set_metadata(self, metadata):
		self.metadata_ = metadata

