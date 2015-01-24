import bpy.ops.import_scene
from converters import import_obj, load

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
		return "Mesh does not have a list of normals."

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

class MeshFile:
	"""A MeshFile holds a Mesh in one of various file formats"""
	convertible_modes = ["OBJ", "SKP", "3DS", "OFF", "HDF5"]

	def __init__(self, filepath, mode):
		self.filepath_ = filepath
		if mode.upper() in convertible_modes:
			self.mode_ = mode.upper() #entered as a String and not including the .
		else:
			self.mode_ = None
			print("Cannot convert files of type " + mode)

	def get_filepath(self):
		return self.filepath_

	def get_mode(self):
		return self.mode_

	def open(self, filepath, mode):
		if mode == "OBJ":
			try:
				import_obj(filepath)
			except:
				bpy.ops.import_scene.obj(filepath)
		if mode == "SKP":
			#implement
		if mode == "3DS":
			bpy.ops.import_scene.autodesk_3ds(filepath)
		if mode == "OFF":
			load(filepath)
		if mode == "HDF5":
			#implement
		else:
			return "Cannot open files of type " + mode

	def read(self):
		#implement function

	def write(self, mesh):
		#implement function
