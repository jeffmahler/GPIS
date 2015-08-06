import mesh
import numpy as np
import sklearn.decomposition

class MeshCleaner:
	RescalingTypeMin = 0
	RescalingTypeMed = 1
	RescalingTypeMax = 2
	RescalingTypeAbsolute = 2

	def __init__(self, mesh):
		self.mesh_ = mesh

	def mesh(self):
		return self.mesh_

	def remove_bad_tris(self):
		''' Remove triangles with illegal out-of-bounds references '''
		new_tris = []
		num_v = len(self.mesh_.vertices())
		for t in self.mesh_.triangles():
			if (t[0] >= 0 and t[0] < num_v and t[1] >= 0 and t[1] < num_v and t[2] >= 0 and t[2] < num_v and
				t[0] != t[1] and t[0] != t[2] and t[1] != t[2]):
				new_tris.append(t)

		self.mesh_.set_triangles(new_tris)
		return self.mesh_

	def remove_unreferenced_vertices(self):
		'''
		Clean out vertices (and normals) not referenced by any triangles.
		'''
		vertex_array = np.array(self.mesh_.vertices())
		num_v = vertex_array.shape[0]

		# fill in a 1 for each referenced vertex
		reffed_array = np.zeros([num_v, 1])
		for f in self.mesh_.triangles():
			if f[0] < num_v and f[1] < num_v and f[2] < num_v:
				reffed_array[f[0]] = 1
				reffed_array[f[1]] = 1
				reffed_array[f[2]] = 1

		# trim out vertices that are not referenced
		reffed_v_old_ind = np.where(reffed_array == 1)
		reffed_v_old_ind = reffed_v_old_ind[0]
		reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1 # counts number of reffed v before each ind

		try:
			self.mesh_.set_vertices(vertex_array[reffed_v_old_ind, :].tolist())
			if self.mesh_.normals():
				normals_array = np.array(self.mesh_.normals())
				self.mesh_.set_normals(normals_array[reffed_v_old_ind, :].tolist())
		except IndexError:
			return False

		# create new face indices
		new_triangles = []
		for f in self.mesh_.triangles():
			new_triangles.append([reffed_v_new_ind[f[0]], reffed_v_new_ind[f[1]], reffed_v_new_ind[f[2]]] )
		self.mesh_.set_triangles(new_triangles)
		return True

	def standardize_pose(self):
		'''
		Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
		centroid and the principal axes are aligned with the vertical Z, Y, and X axes.

		Returns:
		Nothing. Modified the mesh in place (for now)
		'''
		self.mesh_.center_vertices_avg()
		vertex_array_cent = np.array(self.mesh_.vertices())

		# find principal axes
		pca = sklearn.decomposition.PCA(n_components = 3)
		pca.fit(vertex_array_cent)

		# count num vertices on side of origin wrt principal axes
		comp_array = pca.components_
		norm_proj = vertex_array_cent.dot(comp_array.T)
		opposite_aligned = np.sum(norm_proj < 0, axis = 0)
		same_aligned = np.sum(norm_proj >= 0, axis = 0)
		pos_oriented = 1 * (same_aligned > opposite_aligned) # trick to turn logical to int
		neg_oriented = 1 - pos_oriented

		# create rotation from principal axes to standard basis
		target_array = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) # Z+, Y+, X+
		target_array = target_array * pos_oriented + -1 * target_array * neg_oriented
		R = np.linalg.solve(comp_array, target_array)
		R = R.T

		# rotate vertices, normals and reassign to the mesh
		vertex_array_rot = R.dot(vertex_array_cent.T)
		vertex_array_rot = vertex_array_rot.T
		self.mesh_.set_vertices(vertex_array_rot.tolist())
		self.mesh_.center_vertices_bb()

		if self.mesh_.normals():
			normals_array = np.array(self.normals_)
			normals_array_rot = R.dot(normals_array.T)
			self.mesh_.set_normals(normals_array_rot.tolist())

	def rescale_vertices(self, scale, rescaling_type=RescalingTypeMin):
		'''
		Rescales the vertex coordinates so that the minimum dimension (X, Y, Z) is exactly min_scale

		Params:
		scale: (float) scale of the ,esj
                rescaling_type: (int) which dimension to scale along; if not absolute then the min,med,max dim is scaled to be exactly scale
		Returns:
		Nothing. Modified the mesh in place (for now)
		'''
		vertex_array = np.array(self.mesh_.vertices())
		min_vertex_coords = np.min(self.mesh_.vertices(), axis=0)
		max_vertex_coords = np.max(self.mesh_.vertices(), axis=0)
		vertex_extent = max_vertex_coords - min_vertex_coords

		# find minimal dimension
		if rescaling_type == MeshCleaner.RescalingTypeMin:
			dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
                        relative_scale = vertex_extent[dim]
		elif rescaling_type == MeshCleaner.RescalingTypeMed:
			dim = np.where(vertex_extent == np.med(vertex_extent))[0][0]
                        relative_scale = vertex_extent[dim]
		elif rescaling_type == MeshCleaner.RescalingTypeMax:
			dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
                        relative_scale = vertex_extent[dim]
		elif rescaling_type == MeshCleaner.RescalingTypeAbsolute:
                        relative_scale = 1.0

		# compute scale factor and rescale vertices
		scale_factor = scale / relative_scale 
		vertex_array = scale_factor * vertex_array
		self.vertices_ = vertex_array.tolist()

