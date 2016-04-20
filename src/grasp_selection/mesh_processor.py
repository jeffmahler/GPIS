"""
Encapsulates mesh cleaning & preprocessing pipeline for database generation
Authors: Mel Roderick and Jeff Mahler
"""
import glob
import IPython
import logging
import mesh
import numpy as np
import os
import sklearn.decomposition

import feature_file
import obj_file
import stp_file
import sdf_file
import xml.etree.cElementTree as et

OBJ_EXT = '.obj'
SDF_EXT = '.sdf'
STP_EXT = '.stp'
FTR_EXT = '.ftr'
DEC_TAG = '_dec'
PROC_TAG = '_proc'

class MeshProcessor:
    RescalingTypeMin = 0
    RescalingTypeMed = 1
    RescalingTypeMax = 2
    RescalingTypeRelative = 3
    RescalingTypeDiag = 4
    
    def __init__(self, filename):
        file_path, file_root = os.path.split(filename)
        file_root, file_ext = os.path.splitext(file_root)
        self.file_path_ = file_path
        self.file_root_ = file_root
        self.file_ext_ = file_ext

    @property
    def file_path(self):
        return self.file_path_

    @property
    def file_root(self):
        return self.file_root_

    @property
    def file_ext(self):
        return self.file_ext_
    
    @property
    def filename(self):
        return os.path.join(self.file_path, self.file_root + self.file_ext)

    @property
    def mesh(self):
        return self.mesh_

    @property
    def sdf(self):
        return self.sdf_

    @property
    def stable_poses(self):
        return self.stable_poses_

    @property
    def shot_features(self):
        return self.shot_features_

    @property
    def orig_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + self.file_ext_)

    @property
    def obj_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + PROC_TAG + OBJ_EXT)

    @property
    def off_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + PROC_TAG + OFF_EXT)

    @property
    def sdf_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + PROC_TAG + SDF_EXT)

    @property
    def stp_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + PROC_TAG + STP_EXT)

    @property
    def shot_filename(self):
        return os.path.join(self.file_path_, self.file_root_ + PROC_TAG + FTR_EXT)

    def generate_graspable(self, config):
        """ Generates a graspable object """
        self.load_mesh(config['preproc_script'])
        self.mesh_.set_density(config['obj_density'])
        self.clean_mesh(config['obj_scale'], config['obj_rescaling_type'])
        self.generate_sdf(config['sdf_dim'], config['sdf_padding'])
        self.generate_stable_poses(config['stp_min_prob'])
        self.generate_shot_features()
        return self.mesh, self.sdf, self.stable_poses, self.shot_features
        
    def load_mesh(self, script_to_apply=None):
        """ Loads the mesh from the file by first converting to an obj and then loading """        
        # convert to an obj file using meshlab
        if script_to_apply is None:
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(self.filename, self.obj_filename)
        else:
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\" -s \"%s\"' %(self.filename, self.obj_filename, script_to_apply) 
        os.system(meshlabserver_cmd)
        logging.info('MeshlabServer Command: %s' %(meshlabserver_cmd))

        if not os.path.exists(self.obj_filename):
            raise ValueError('Meshlab conversion failed for %s' %(self.obj_filename))
        
        # read mesh from obj file
        of = obj_file.ObjFile(self.obj_filename)
        self.mesh_ = of.read()
        return self.mesh_ 

    def clean_mesh(self, scale, rescaling_type):
        """ Runs all cleaning ops at once """
        self._remove_bad_tris()
        self._remove_unreferenced_vertices()
        self._standardize_pose()
        self._rescale_vertices(scale, rescaling_type)

    def _remove_bad_tris(self):
        """ Remove triangles with illegal out-of-bounds references """
        new_tris = []
        num_v = len(self.mesh_.vertices())
        for t in self.mesh_.triangles():
            if (t[0] >= 0 and t[0] < num_v and t[1] >= 0 and t[1] < num_v and t[2] >= 0 and t[2] < num_v and
                t[0] != t[1] and t[0] != t[2] and t[1] != t[2]):
                new_tris.append(t)
        self.mesh_.set_triangles(new_tris)
        return self.mesh_

    def _remove_unreferenced_vertices(self):
        """ Clean out vertices (and normals) not referenced by any triangles. """
        # convert vertices to an array
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
            if self.mesh_.normals() is not None:
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

    def _standardize_pose(self):
        """
        Transforms the vertices and normals of the mesh such that the origin of the resulting mesh's coordinate frame is at the
        centroid and the principal axes are aligned with the vertical Z, Y, and X axes.
        
        Returns:
        Nothing. Modified the mesh in place (for now)
        """
        self.mesh_.center_vertices_bb()
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

        if self.mesh_.normals() is not None:
            normals_array = np.array(self.mesh_.normals_)
            normals_array_rot = R.dot(normals_array.T)
            self.mesh_.set_normals(normals_array_rot.tolist())

    def _rescale_vertices(self, scale, rescaling_type=RescalingTypeMin):
        """
        Rescales the vertex coordinates so that the minimum dimension (X, Y, Z) is exactly min_scale
        
        Params:
        scale: (float) scale of the mesh
        rescaling_type: (int) which dimension to scale along; if not absolute then the min,med,max dim is scaled to be exactly scale
        Returns:
        Nothing. Modified the mesh in place (for now)
        """
        vertex_array = np.array(self.mesh_.vertices())
        min_vertex_coords = np.min(self.mesh_.vertices(), axis=0)
        max_vertex_coords = np.max(self.mesh_.vertices(), axis=0)
        vertex_extent = max_vertex_coords - min_vertex_coords

        # find minimal dimension
        if rescaling_type == MeshProcessor.RescalingTypeMin:
            dim = np.where(vertex_extent == np.min(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == MeshProcessor.RescalingTypeMed:
            dim = np.where(vertex_extent == np.med(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == MeshProcessor.RescalingTypeMax:
            dim = np.where(vertex_extent == np.max(vertex_extent))[0][0]
            relative_scale = vertex_extent[dim]
        elif rescaling_type == MeshProcessor.RescalingTypeRelative:
            relative_scale = 1.0
        elif rescaling_type == MeshProcessor.RescalingTypeDiag:
            diag = np.linalg.norm(vertex_extent)
            relative_scale = diag / 3.0 # make the gripper size exactly one third of the diagonal

        # compute scale factor and rescale vertices
        scale_factor = scale / relative_scale 
        vertex_array = scale_factor * vertex_array
        self.mesh_.vertices_ = vertex_array.tolist()
        self.mesh_._compute_bb_center()
        self.mesh_._compute_centroid()
        self.mesh_.set_center_of_mass(self.mesh_.bb_center_)
        
    def generate_sdf(self, dim, padding):
        """ Converts mesh to an sdf object """
        # write the mesh to file
        of = obj_file.ObjFile(self.obj_filename)
        of.write(self.mesh_)

        # create the SDF using binary tools
        sdfgen_cmd = '/home/jmahler/Libraries/SDFGen/bin/SDFGen \"%s\" %d %d' %(self.obj_filename, dim, padding)
        os.system(sdfgen_cmd)
        logging.info('SDF Command: %s' %(sdfgen_cmd))

        if not os.path.exists(self.sdf_filename):
            logging.warning('SDF computation failed for %s' %(self.sdf_filename))
            return None
        os.system('chmod a+rwx \"%s\"' %(self.sdf_filename) )

        # read the generated sdf
        sf = sdf_file.SdfFile(self.sdf_filename)
        self.sdf_ = sf.read()
        return self.sdf_

    def generate_stable_poses(self, min_prob = 0.05):
        """ Computes mesh stable poses """
        # hacky as hell but I'm tired of redoing this
        stpf = stp_file.StablePoseFile()
        stpf.write_mesh_stable_poses(self.mesh_, self.stp_filename, min_prob=min_prob)
        self.stable_poses_ = stpf.read(self.stp_filename)
        return self.stable_poses_

    def generate_shot_features(self):
        """ Extracts SHOT features """
        # extract shot features to the filesystem
        shot_os_call = 'bin/shot_extractor %s %s' %(self.obj_filename, self.shot_filename)
        os.system(shot_os_call)

        # read the features back in
        lff = feature_file.LocalFeatureFile(self.shot_filename)
        self.shot_features_ = lff.read()
        return self.shot_features_

    def convex_pieces(self, config):
        """ Generate a list of meshes consitituting the convex pieces of the mesh """
        # get volume
        orig_volume = self.mesh_.get_total_volume()
        
        # convert to off
        meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(self.obj_filename, self.off_filename) 
        os.system(meshlabserver_cmd)
        logging.info('MeshlabServer OFF Conversion Command: %s' %(meshlabserver_cmd))

        if not os.path.exists(off_filename):
            logging.warning('Meshlab conversion failed for %s' %(off_filename))
            return
        
        # create convex pieces
        cvx_decomp_command = config['hacd_cmd_template'] %(self.off_filename,
                                                           config['min_num_clusters'],
                                                           config['max_concavity'],
                                                           config['invert_input_faces'],
                                                           config['extra_dist_points'],
                                                           config['add_faces_points'],
                                                           config['connected_components_dist'],
                                                           config['target_num_triangles'])
        logging.info('CV Decomp Command: %s' %(cvx_decomp_command))
        os.system(cvx_decomp_command)        

        # convert each wrl to an obj and an stl
        convex_piece_files = glob.glob('%s_dec_hacd_*.wrl' %(os.path.join(self.file_path_, self.file_root_)))
        convex_piece_meshes = []
        total_volume = 0.0

        for convex_piece_file in convex_piece_files:
            file_root, file_ext = os.path.splitext(convex_piece_file)
            obj_filename = file_root + '.obj'
            stl_filename = file_root + '.stl'
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(convex_piece_file, obj_filename) 
            os.system(meshlabserver_cmd)
            meshlabserver_cmd = 'meshlabserver -i \"%s\" -o \"%s\"' %(convex_piece_file, stl_filename) 
            os.system(meshlabserver_cmd)

            of = obj_file.ObjFile(obj_filename)
            convex_piece = of.read()
            total_volume += convex_piece.get_total_volume()
            convex_piece_meshes.append(of.read())

        root = et.Element('robot', name="test")

        # get the masses and moments of inertia
        effective_density = orig_volume / total_volume
        prev_piece_name = None
        for convex_piece, filename in zip(convex_piece_meshes, convex_piece_files):
            convex_piece.set_center_of_mass(np.zeros(3))
            convex_piece.set_density(self.mesh_.density * effective_density)
            
            # write to xml
            piece_name = 'link_%s'%(file_root)
            file_path_wo_ext, file_ext = os.path.splitext(filename)
            file_path, file_root = os.path.split(file_path_wo_ext)
            I = convex_piece.inertia
            link = et.SubElement(root, 'link', name=piece_name)

            inertial = et.SubElement(link, 'inertial')
            origin = et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
            mass = et.SubElement(inertial, 'mass', value='%f'%convex_piece.mass)
            inertia = et.SubElement(inertial, 'inertia', ixx='%f'%I[0,0], ixy='%f'%I[0,1], ixz='%f'%I[0,2],
                                    iyy='%f'%I[1,1], iyz='%f'%I[1,2], izz='%f'%I[2,2])
            
            visual = et.SubElement(link, 'visual')
            origin = et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
            geometry = et.SubElement(visual, 'geometry')
            mesh = et.SubElement(geometry, 'mesh', filename=file_path_wo_ext+'.stl')
            material = et.SubElement(visual, 'material', name='')
            color = et.SubElement(material, 'color', rgba="0.75 0.75 0.75 1")

            collision = et.SubElement(link, 'collision')
            origin = et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")            
            geometry = et.SubElement(collision, 'geometry')
            mesh = et.SubElement(geometry, 'mesh', filename=file_path_wo_ext+'.stl')

            if prev_piece_name is not None:
                joint = et.SubElement(root, 'joint', name='%s_joint'%(piece_name), type='fixed')
                origin = et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
                parent = et.SubElement(joint, 'parent', link=prev_piece_name)
                child = et.SubElement(joint, 'child', link=piece_name)

            prev_piece_name = piece_name

            """
            txt_filename = file_root + '.txt'
            f = open(txt_filename, 'w')
            f.write('mass: %f\n' %(convex_piece.mass))
            f.write('inertia: ' + str(convex_piece.inertia) + '\n')
            f.close()
            """

        tree = et.ElementTree(root)
        tree.write('test.URDF')
        exit(0)

        return convex_piece_meshes

