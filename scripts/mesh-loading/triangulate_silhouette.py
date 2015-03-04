import IPython
import numpy as np
import os
from PIL import Image, ImageDraw
import sklearn.decomposition
import sys

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

    def convert_binary_image_to_mesh(self, binary_img, extrusion = 100):
        '''
        Converts a binary image in file "filename" to a mesh
        
        Params:
           binary_img: (2d numpy arry) binary image for silhouette (255 = occupied, 0 = not occupied)
        Returns:
           mesh object with specified depth
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
        self.add_boundary_tris(boundary_img, verts, tris, front_ind_map, back_ind_map)

        # convert to mesh and return
        m = mesh.Mesh(verts, tris)
        m.remove_unreferenced_vertices()
        return m

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
        boundary_ind = np.where(boundary_img == self.upper_bound_)
        boundary_coords = zip(boundary_ind[0], boundary_ind[1])

        # setup inital vars
        cur_coord = boundary_coords[0]
        tris_arr = np.array(tris)

        visited_map = np.zeros(boundary_img.shape)
        visited_map[cur_coord[0], cur_coord[1]] = 1
        another_visit_avail = True

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


            # add triangles
            next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
            next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
            tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
            tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])

            # mark coordinate as visited
            visited_map[cur_coord[0], cur_coord[1]] = 1
            coord_visits.append(cur_coord)

        # add edge back to first coord
        cur_coord = coord_visits[0]
        next_front_ind = front_ind_map[cur_coord[0], cur_coord[1]]
        next_back_ind = back_ind_map[cur_coord[0], cur_coord[1]]
        tris.append([int(front_ind), int(back_ind), int(next_front_ind)])
        tris.append([int(back_ind), int(next_back_ind), int(next_front_ind)])


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

if __name__ == '__main__':
    filename = sys.argv[1]
    binary_img = Image.open(filename)
    m = MeshConverter2D()
    bin_mesh = m.convert_binary_image_to_mesh(binary_img)

    out_filename = 'test.obj'
    dec_filename = 'test_dec.obj'
    dec_script_path = 'scripts/meshlab/decimation.mlx'
    of = obj_file.ObjFile(out_filename)
    of.write(bin_mesh)

    meshlabserver_cmd = 'meshlabserver -i %s -o %s -s %s' %(out_filename, dec_filename, dec_script_path)
    os.system(meshlabserver_cmd)

