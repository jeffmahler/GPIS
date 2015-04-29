import IPython
import numpy as np
import os
from PIL import Image, ImageDraw
import sklearn.decomposition
import sys

import obj_file

class CameraParams:
        '''
        Encapsulates camera parameters and the operations we want to do with them
        '''
        def __init__(self, height, width, fx, fy=None, cx=None, cy=None):
                '''
                Init camera parameters
                
                Params:
                   height: (int or float) height of image
                   width: (int of flaot) width of image
                   fx: (float) x focal length of camera in pixels
                   fy: (float) y focal length of camera in pixels
                   cx: (float) optical center of camera in pixels along x axis
                   cy: (float) optical center of camera in pixels along y axis
                '''
                self.height_ = height
                self.width_ = width
                self.fx_ = fx

                # set focal, camera center automatically if under specified
                if fy is None:
                        self.fy_ = fx
                else:
                        self.fy_= fy
                if cx is None:
                        self.cx_ = float(width) / 2
                else:
                        self.cx_ = cx
                if cy is None:
                        self.cy_ = float(height) / 2
                else:
                        self.cy_ = cy
                # set camera projection matrix
                self.K_ = np.array([[self.fx_,        0, self.cx_],
                                    [       0, self.fy_, self.cy_],
                                    [       0,        0,        1]])

        def get_height(self):
                return self.height_

        def get_width(self):
                return self.width_

        def get_proj_matrix(self):
                return self.K_
        
        def project(self, points):
                '''
                Projects a set of points into the camera given by these parameters
                
                Params:
                   points: (3xN numpy array of floats) 3D points to project
                Returns:
                   2xN numpy float array of 2D image coordinates
                   1xN binary numpy array indicating whether or not point projected outside of image
                '''
                # check valid data
                if points.shape[0] != 3:
                        raise Exception('Incorrect data dimension. CameraParams project must be supplied a 3xN numpy float array.')

                points_proj = self.K_.dot(points)
                point_depths = np.tile(points_proj[2,:], [3, 1])
                points_proj = np.divide(points_proj, point_depths)
                points_proj = np.round(points_proj)

                # find valid indices
                valid = (points[0,:] >= 0) & (points[1,:] >= 0) & (points[0,:] < self.width_) & (points[1,:] < self.height_)

                return points_proj[:2,:].astype(np.int), valid

class Mesh:
	"""
        A Mesh is a three-dimensional shape representation
        
        Params:
           vertices:  (list of 3-lists of float)
           triangles: (list of 3-lists of ints)
           normals:   (list of 3-lists of float)
           metadata:  (dictionary) data like category, etc
           pose:      (4x4 numpy array) pose of mesh in world basis
        """
	def __init__(self, vertices, triangles, normals=None, metadata=None, pose=None):
		self.vertices_ = vertices
		self.triangles_ = triangles
		self.normals_ = normals
		self.metadata_ = metadata
                if pose is None:
                        # assume identity pose if none provided
                        self.pose_ = np.eye(4)
                else:
                        self.pose_ = pose

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

        def project_binary(self, camera_params, camera_pose):
                '''
                Project the triangles of the mesh into a binary image which is 1 if the mesh is
                visible at that point in the image and 0 otherwise.
                The image is assumed to be taken from a camera with specified params at
                a given relative pose to the mesh.

                Params:
                   camera_params: CameraParams object
                   camera_pose: 4x4 pose matrix (camera in mesh basis)
                Returns:
                   PIL binary image (1 = mesh projects to point, 0 = does not)
                '''
                vertex_array = np.array(self.vertices_)
                R = camera_pose[:3,:3]
                t = camera_pose[:3,3:]
                t_rep = np.tile(t, [1, 3]) # replicated t vector for fast transformations
                
                height = camera_params.get_height()
                width = camera_params.get_width()
                img_shape = [int(height), int(width)]
                fill_img = Image.fromarray(np.zeros(img_shape).astype(np.uint8)) # init grayscale image
                img_draw = ImageDraw.Draw(fill_img)

                # project each triangle and fill in image
                for f in self.triangles_:
                        verts_mesh_basis = np.array([vertex_array[f[0],:], vertex_array[f[1],:], vertex_array[f[2],:]])

                        # transform to camera basis
                        verts_cam_basis = R.dot(verts_mesh_basis.T) + t_rep
                        # project points to the camera imaging plane and snap to image borders
                        verts_proj, valid = camera_params.project(verts_cam_basis)
                        verts_proj[0, verts_proj[0,:] < 0] = 0
                        verts_proj[1, verts_proj[1,:] < 0] = 0
                        verts_proj[0, verts_proj[0,:] >= width]  = width - 1
                        verts_proj[1, verts_proj[1,:] >= height] = height - 1

                        # fill in area on image
                        img_draw.polygon([tuple(verts_proj[:,0]), tuple(verts_proj[:,1]), tuple(verts_proj[:,2])], fill=255)
                return fill_img
        
        def remove_unreferenced_vertices(self):
                '''
                Clean out vertices (and normals) not referenced by any triangles.
                '''
                vertex_array = np.array(self.vertices_)
                num_v = vertex_array.shape[0]

                # fill in a 1 for each referenced vertex
                reffed_array = np.zeros([num_v, 1])
                for f in self.triangles_:
                        reffed_array[f[0]] = 1
                        reffed_array[f[1]] = 1
                        reffed_array[f[2]] = 1

                # trim out vertices that are not referenced
                reffed_v_old_ind = np.where(reffed_array == 1)
                reffed_v_old_ind = reffed_v_old_ind[0]
                reffed_v_new_ind = np.cumsum(reffed_array).astype(np.int) - 1 # counts number of reffed v before each ind

                try:
                        self.vertices_ = vertex_array[reffed_v_old_ind, :].tolist()
                        if self.normals_:
                                normals_array = np.array(self.normals_)
                                self.normals_ = normals_array[reffed_v_old_ind, :].tolist()
                except IndexError:
                        return False

                # create new face indices
                new_triangles = []
                for f in self.triangles_:
                        new_triangles.append([reffed_v_new_ind[f[0]], reffed_v_new_ind[f[1]], reffed_v_new_ind[f[2]]] )
                self.triangles_ = new_triangles
                return True

        def image_to_3d_coords(self):
                '''
                Flip x and y axes (if created from image this might help)
                '''
                if len(self.vertices_) > 0:
                        vertex_array = np.array(self.vertices_)
                        new_vertex_array = np.zeros(vertex_array.shape)
                        new_vertex_array[:,0] = vertex_array[:,1]
                        new_vertex_array[:,1] = vertex_array[:,0]
                        new_vertex_array[:,2] = vertex_array[:,2]
                        self.vertices_ = new_vertex_array.tolist()
                        return True
                else:
                        return False
        
        def center_vertices_avg(self):
                '''
                Centers vertices at average vertex
                '''
                vertex_array = np.array(self.vertices_)
                centroid = np.mean(vertex_array, axis = 0)
                vertex_array_cent = vertex_array - centroid
                self.vertices_ = vertex_array_cent.tolist()

        def center_vertices_bb(self):
                '''
                Centers vertices at center of bounding box
                '''
                vertex_array = np.array(self.vertices_)
                min_vertex = np.min(vertex_array, axis = 0)
                max_vertex = np.max(vertex_array, axis = 0)
                centroid = (max_vertex + min_vertex) / 2
                vertex_array_cent = vertex_array - centroid
                self.vertices_ = vertex_array_cent.tolist()

        def normalize_vertices(self):
                '''
                Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
                centroid and the principal axes are aligned with the vertical Z, Y, and X axes.

                Returns:
                   Nothing. Modified the mesh in place (for now)
                '''
                self.center_vertices_bb()
                vertex_array_cent = np.array(self.vertices_)
                
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
                self.vertices_ = vertex_array_rot.tolist()

                if self.normals_:
                        normals_array = np.array(self.normals_)
                        normals_array_rot = R.dot(vertex_array_cent.T)
                        self.normals_ = normals_array_rot.tolist()

        def rescale_vertices(self, min_scale):
                '''
                Rescales the vertex coordinates so that the minimum dimension (X, Y, Z) is exactly min_scale

                Params:
                   min_scale: (float) the scale of the min dimension
                Returns:
                   Nothing. Modified the mesh in place (for now)
                '''
                vertex_array = np.array(self.vertices_)
                min_vertex_coords = np.min(self.vertices_, axis=0)
                max_vertex_coords = np.max(self.vertices_, axis=0)
                vertex_extent = max_vertex_coords - min_vertex_coords

                # find minimal dimension
                min_dim = np.where(vertex_extent == np.min(vertex_extent))
                min_dim = min_dim[0][0]

                # compute scale factor and rescale vertices
                scale_factor = min_scale / vertex_extent[min_dim] 
                vertex_array = scale_factor * vertex_array
                self.vertices_ = vertex_array.tolist()

        def make_image(self, filename, rot):
            proj_img = self.project_binary(cp, T)
            file_root, file_ext = os.path.splitext(filename)
            proj_img.save(file_root + ".jpg")                

            oof = obj_file.ObjFile(filename)
            oof.write(self)
                
if __name__ == '__main__':
        # test various aspects of mesh cleanup
        with open("paths", "r") as ifile:
            for line in ifile: 
                print("converting " + line) 
                filename = line
                of = obj_file.ObjFile(filename)
                m = of.read()

                cp = CameraParams(480., 640., 525., 525.)

                T = np.eye(4)
                T[:3,3] = np.array([0, 0, 3.0])
                min_dim = 0.5
                
                m.remove_unreferenced_vertices()

                m.normalize_vertices()
                m.rescale_vertices(min_dim)

                m.make_image(filename, T)

        T = np.eye(4)
        theta = np.pi / 2
        R = np.array([[np.cos(theta), 0, -np.sin(theta)],
                      [0, 1, 0],
                      [np.sin(theta), 0, np.cos(theta)]])
        T[:3,:3] = R
        T[:3,3] = np.array([0, 0, 2.5])
        min_dim = 0.5
        
        m.remove_unreferenced_vertices()


#        IPython.embed()

