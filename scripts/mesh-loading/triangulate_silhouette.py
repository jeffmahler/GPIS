import IPython
import numpy as np
import os
from PIL import Image, ImageDraw
import sklearn.decomposition
import sys

import skimage.morphology as morph
import matplotlib
import matplotlib.pyplot as plt

import mesh
import obj_file

def adjacent_black_pixels(i, j, pixels, lower_bound):
    num_black = 0
    diffs = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    for d in diffs:
        if pixels[i+d[0]][j+d[1]] <= lower_bound:
            num_black += 1
    '''
    for m in range(-1, 2):
        for n in range(-1, 2):
            if pixels[i+m][j+n] <= lower_bound:
                num_black += 1
    '''
    return num_black

class MeshConverter2D:
    def __init__(self):
        self.occupied_thresh_ = 50
        self.upper_bound_ = 255
        # what to do?

    def convert_binary_image_to_sdf(self, binary_img, vis = False):
        binary_data = np.array(binary_img)
        skel, sdf_in = morph.medial_axis(binary_data, return_distance = True)
        useless_skel, sdf_out = morph.medial_axis(self.upper_bound_ - binary_data, return_distance = True)
        
        sdf = sdf_out - sdf_in

        # display the sdf and skeleton
        if vis:
            dist_on_skel = sdf * skel
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(binary_data, cmap=plt.cm.gray, interpolation='nearest')
            ax1.axis('off')
            ax2.imshow(dist_on_skel, cmap=plt.cm.spectral, interpolation='nearest')
            ax2.contour(binary_data, [0.5], colors='w')
            ax2.axis('off')

            fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0, right=1)
            plt.show()

            plt.imshow(sdf)
            plt.show()

            plt.imshow(skel)
            plt.show()

        return sdf, skel

    def convert_binary_image_to_mesh(self, binary_img, extrusion = 1000, min_dim = 5):
        '''
        Converts a binary image in file "filename" to a mesh
        
        Params:
           binary_img: (2d numpy arry) binary image for silhouette (255 = occupied, 0 = not occupied)
        Returns:
           mesh object with specified depth
           bool, whether or not triangulation was successful (since it doesn't work for certain topologies)
        '''
        # get occupied indices from binary image
        binary_map = np.array(binary_img)
        occ_ind = np.where(binary_map > self.occupied_thresh_)
        occ_coords = zip(occ_ind[0], occ_ind[1]) 

        # todo: duplicate at multiple depths
        front_verts, front_tris, front_ind_map = self.create_mesh_face(occ_coords, extrusion / 2, binary_map.shape, cw = True)
        back_verts, back_tris, back_ind_map = self.create_mesh_face(occ_coords, -extrusion / 2, binary_map.shape, cw = False)
        verts, tris = self.join_vert_tri_lists(front_verts, front_tris, back_verts, back_tris)
        num_verts = len(front_verts)
        back_ind_map = back_ind_map + num_verts

        # todo: connect boundaries
        boundary_img = self.find_boundary(binary_img)
        success = self.add_boundary_tris(boundary_img, verts, tris, front_ind_map, back_ind_map)

        # convert to mesh and return
        m = mesh.Mesh(verts, tris)
        unreffed_success = m.remove_unreferenced_vertices()
        succcess = success and unreffed_success
        coord_conversion = m.image_to_3d_coords()
        success = success and coord_conversion
#        m.normalize_vertices()
        # m.rescale_vertices(min_dim)
        return m, success

    def join_vert_tri_lists(self, verts1, tris1, verts2, tris2):
        '''
        Joins a list of vertices and triangles
        
        Params:
           verts1: (list of 3-lists of float)
           tris1:  (list of 3-lists of ints)
           verts2: (list of 3-lists of float)
           tris2:  (list of 3-lists of ints)
        Returns:
           verts: (list of 3-lists of float) joined list of vertices
           tris:  (list of 3-lists of ints) joined list of triangles
        '''
        num_verts1 = len(verts1)

        # simple append for verts
        verts = list(verts1)
        verts.extend(verts2)

        # offset and append triangle (vertex indices)
        tris = list(tris1)
        tris2_offset = [[num_verts1 + t[0], num_verts1 + t[1], num_verts1 + t[2]] for t in tris2]
        tris.extend(tris2_offset)
        return verts, tris

    def add_boundary_tris(self, boundary_img, verts, tris, front_ind_map, back_ind_map):
        '''
        Connects front and back faces along the boundary, modifying tris IN PLACE
        NOTE: Right now this only works for points topologically equivalent to a sphere
              Can be extended by parsing back over untriangulated boundary points

        Params:
           boundary_img: (numpy array) 255 if point on boundary, 0 otherwise
           verts: (list of 3-lists of float)
           tris: (list of 3-lists of integers)
           front_ind_map: (numpy 2d array) maps vertex coords to the indices of their front face vertex in list  
           back_ind_map: (numpy 2d array) maps vertex coords to the indices of their back face vertex in list  
        '''
        remaining_boundary = np.copy(boundary_img)
        boundary_ind = np.where(remaining_boundary == self.upper_bound_)
        boundary_coords = zip(boundary_ind[0], boundary_ind[1])
        if len(boundary_coords) == 0:
            return False

        # setup inital vars
        tris_arr = np.array(tris)

        visited_map = np.zeros(boundary_img.shape)
        another_visit_avail = True

        # make sure to start with a reffed tri
        reffed = False
        visited_marker = 128
        i = 0
        while not reffed and i < len(boundary_coords):
            cur_coord = boundary_coords[i]
            visited_map[cur_coord[0], cur_coord[1]] = 1
            front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
            back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
            ref_tris = np.where(tris_arr == front_ind)
            ref_tris = ref_tris[0]
            reffed = (ref_tris.shape[0] > 0)
            remaining_boundary[cur_coord[0], cur_coord[1]] = visited_marker
            i = i+1

        coord_visits = [cur_coord]
        cur_dir_angle = np.pi / 2 # start straight down

        # loop around boundary and add faces connecting front and back
        while another_visit_avail:
            front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
            back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
            ref_tris = np.where(tris_arr == front_ind)
            ref_tris = ref_tris[0]
            num_reffing_tris = ref_tris.shape[0]
            
            # get all possible cadidates from neighboring tris
            another_visit_avail = False
            candidate_next_coords = []
            for i in xrange(num_reffing_tris):
                reffing_tri = tris[ref_tris[i]]
                for j in xrange(3):
                    v = verts[reffing_tri[j]]
                    if boundary_img[v[0], v[1]] == self.upper_bound_ and visited_map[v[0], v[1]] == 0:
                        candidate_next_coords.append([v[0], v[1]])
                        another_visit_avail = True

            # get the "rightmost" next point
            num_candidates = len(candidate_next_coords)
            if num_candidates > 0:
                # calculate candidate directions
                directions = []
                next_dirs = np.array(candidate_next_coords) - np.array(cur_coord)
                dir_norms = np.linalg.norm(next_dirs, axis = 1)
                next_dirs = next_dirs / np.tile(dir_norms, [2, 1]).T

                # calculate angles relative to positive x axis
                new_angles = np.arctan(next_dirs[:,0] / next_dirs[:,1])
                negative_ind = np.where(next_dirs[:,1] < 0)
                negative_ind = negative_ind[0]
                new_angles[negative_ind] = new_angles[negative_ind] + np.pi

                # compute difference in angles
                angle_diff = new_angles - cur_dir_angle
                correction_ind = np.where(angle_diff <= -np.pi)
                correction_ind = correction_ind[0]
                angle_diff[correction_ind] = angle_diff[correction_ind] + 2 * np.pi

                # choose the next coordinate with the maximum angle diff (rightmost)
                next_ind = np.where(angle_diff == np.max(angle_diff))
                next_ind = next_ind[0]

                cur_coord = candidate_next_coords[next_ind[0]]
                cur_dir_angle = new_angles[next_ind[0]]

                # add triangles (only add if there is a new candidate)
                next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
                next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
                tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
                tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])

                # mark coordinate as visited
                visited_map[cur_coord[0], cur_coord[1]] = 1
                coord_visits.append(cur_coord)
                remaining_boundary[cur_coord[0], cur_coord[1]] = visited_marker

        # add edge back to first coord
        cur_coord = coord_visits[0]
        next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
        next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
        tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
        tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])

        # check success 
        success = (np.sum(remaining_boundary == self.upper_bound_) == 0)
#        IPython.embed()
        return success

    def find_boundary(self, im):

        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
        arr = [[0 for _ in range(width)] for _ in range(height)]

        for i in range(1, len(pixels) - 1):
            for j in range(1, len(pixels[0]) - 1):
                if (pixels[i][j] > self.occupied_thresh_) and (pixels[i][j] <= self.upper_bound_) and (adjacent_black_pixels(i, j, pixels, self.occupied_thresh_) > 0):
                    arr[i][j] = 255
        return np.array(arr)


    def create_mesh_face(self, occ_coords, depth, index_shape, cw = True):
        '''
        Creates a 2D mesh face of vertices and triangles from the given coordinates at a specified depth
        
        Params:
           occ_coords: (list of vertices (3-tuples)) the coordinates of vertices
           depth: (float) the depth at which to place the face
           index_shape: (2-tuple) the shape of the numpy grid on which the vertices lie
           cw: (bool) clockwise or counterclockwise orientation
        Returns:
           verts: (list of verts)
           tris: (list of tris)
        '''
        # get mesh vertices
        verts = []
        tris = []
        ind_map = -1 * np.ones(index_shape) # map vertices to indices in vert list
        for coord in occ_coords:
            verts.append([coord[0], coord[1], depth])
            ind_map[coord[0], coord[1]] = len(verts) - 1

        # get mesh triangles
        # rule: vertex adds triangles that it is the 90 degree corner of
        for coord in occ_coords:
            coord_right = [coord[0] + 1, coord[1]]
            coord_left  = [coord[0] - 1, coord[1]]
            coord_below = [coord[0], coord[1] + 1]
            coord_above = [coord[0], coord[1] - 1]
            cur_ind = ind_map[coord[0], coord[1]]

            # add tri above left
            if coord_left[0] >= 0 and coord_above[1] >= 0:
                left_ind = ind_map[coord_left[0], coord_left[1]]
                above_ind = ind_map[coord_above[0], coord_above[1]]

                # check if valid vertices and add
                if left_ind > -1 and above_ind > -1:
                    if cw:
                        tris.append([int(cur_ind), int(left_ind), int(above_ind)])
                    else:
                        tris.append([int(cur_ind), int(above_ind), int(left_ind)])                        
                elif above_ind > -1:
                    # try to patch area
                    coord_left_above = [coord[0] - 1, coord[1] - 1]
                    if coord_left_above[0] > 0 and coord_left_above[1] > 0:
                        left_above_ind = ind_map[coord_left_above[0], coord_left_above[1]]

                        # check validity
                        if left_above_ind > -1:
                            if cw:
                                tris.append([int(cur_ind), int(left_above_ind), int(above_ind)])
                            else:
                                tris.append([int(cur_ind), int(above_ind), int(left_above_ind)])                                

            # add tri below right
            if coord_right[0] < index_shape[1] and coord_below[1] < index_shape[0]:
                right_ind = ind_map[coord_right[0], coord_right[1]]
                below_ind = ind_map[coord_below[0], coord_below[1]]

                # check if valid vertices and add
                if right_ind > -1 and below_ind > -1:
                    if cw:
                        tris.append([int(cur_ind), int(right_ind), int(below_ind)])
                    else:
                        tris.append([int(cur_ind), int(below_ind), int(right_ind)])
                elif below_ind > -1:
                    # try to patch area
                    coord_right_below = [coord[0] + 1, coord[1] + 1]
                    if coord_right_below[0] < index_shape[0] and coord_right_below[1] < index_shape[1]:
                        right_below_ind = ind_map[coord_right_below[0], coord_right_below[1]]

                        # check validity
                        if right_below_ind > -1:
                            if cw:
                                tris.append([int(cur_ind), int(right_below_ind), int(below_ind)])
                            else:
                                tris.append([int(cur_ind), int(below_ind), int(right_below_ind)])

        return verts, tris, ind_map

def create_binary_rect(img_height, img_width, rect_height, rect_width):
    binary_img = np.zeros([img_height, img_width])
    img_center = np.array([img_height / 2, img_width / 2])

    rect_h_low = img_center[0] - rect_height / 2
    rect_h_high = img_center[0] + rect_height / 2

    rect_w_low = img_center[1] - rect_width / 2
    rect_w_high = img_center[1] + rect_width / 2
 
    binary_img[rect_h_low:rect_h_high, rect_w_low:rect_w_high] = 255
    return binary_img
    

if __name__ == '__main__':
    filename = sys.argv[1]
    dec_filename = sys.argv[2]
    make_rect = int(sys.argv[3])

    im_width  = 200.0
    im_height = 200.0
    rect_width = 12.0
    rect_height = 20.0

    if make_rect:
        binary_img = create_binary_rect(im_height, im_width, rect_height, rect_width)
        binary_img = Image.fromarray(binary_img)
        m = MeshConverter2D()
        sdf, skel = m.convert_binary_image_to_sdf(binary_img, False)
        bin_mesh, succeeded = m.convert_binary_image_to_mesh(binary_img)    

    else:
        binary_img = Image.open(filename)
        binary_img = binary_img.resize([int(im_height), int(im_width)])
        
        m = MeshConverter2D()
        bin_mesh, succeeded = m.convert_binary_image_to_mesh(binary_img)    

        '''
        cp = mesh.CameraParams(im_height, im_width, 52.5, 52.5)
        T = np.eye(4)
        T[:3,3] = np.array([0, 0, 1.5])
        proj_bin_img = bin_mesh.project_binary(cp, T)
        '''
        sdf, skel = m.convert_binary_image_to_sdf(binary_img)


    np.savetxt("out.csv", sdf, delimiter=",", header="%d %d"%(sdf.shape[0], sdf.shape[1]))

    out_filename = 'tmp_high_res.obj'
    dec_script_path = 'scripts/meshlab/decimation.mlx'
    of = obj_file.ObjFile(out_filename)
    of.write(bin_mesh)

    meshlabserver_cmd = 'meshlabserver -i %s -o %s -s %s' %(out_filename, dec_filename, dec_script_path)
    os.system(meshlabserver_cmd)
