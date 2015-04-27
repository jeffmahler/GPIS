import math
import numpy as np
import PIL
from PIL import Image
import os
import random
import sys

import mesh
import obj_file as of
import triangulate_silhouette as ts

import IPython

formats = ['_clean.obj']
bad_categories = ['helicopter', 'clothes_hanger', 'mnt', 'terastation', 'shape_data', 'YCB', 'meshes', 'textured_meshes', 'kinbody', '']

#global transforms
theta = np.pi / 2
t = [0, 0, 5.0]
Rx = np.array([[1, 0, 0],
               [0, np.cos(theta), -np.sin(theta)],
               [0, np.sin(theta), np.cos(theta)]])
Ry = np.array([[np.cos(theta), 0, -np.sin(theta)],
               [0, 1, 0],
               [np.sin(theta), 0, np.cos(theta)]])
Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]])
Rxp = Rx.T
Ryp = Ry.T
Rzp = Rz.T
R_list = [Rx, Rxp, Ry, Ryp, Rz, Rzp]

def extract_category(root):
    head, tail = os.path.split(root)
    while head != '/' and tail in bad_categories:
        head, tail = os.path.split(head)
    success = (tail not in bad_categories)
    return success, tail

def create_parallel_jaw_grippers(img_height, img_width, rect_height, rect_width, div = 4):
    binary_img = np.zeros([img_height, img_width])
    img_center = np.array([img_height / 2, img_width / 2])
    img_top = np.array([img_height / div, img_width / div])
    img_bottom = np.array([(div - 1 ) * img_height / div, (div - 1 ) * img_width / div])

    # create top
    rect_h_low = img_top[0] - rect_height / 2
    rect_h_high = img_top[0] + rect_height / 2
    rect_w_low = img_center[1] - rect_width / 2
    rect_w_high = img_center[1] + rect_width / 2
 
    binary_img[rect_h_low:rect_h_high, rect_w_low:rect_w_high] = 255

    # create bottom
    rect_h_low = img_bottom[0] - rect_height / 2
    rect_h_high = img_bottom[0] + rect_height / 2
    rect_w_low = img_center[1] - rect_width / 2
    rect_w_high = img_center[1] + rect_width / 2
 
    binary_img[rect_h_low:rect_h_high, rect_w_low:rect_w_high] = 255

    # create mesh and sdf
    binary_img = Image.fromarray(binary_img.astype(np.uint8))
    mesh_converter = ts.MeshConverter2D()
    sdf, skel = mesh_converter.convert_binary_image_to_sdf(binary_img)
    binary_mesh, triangulation_succeeded = mesh_converter.convert_binary_image_to_mesh(binary_img)    
    return sdf, binary_mesh, binary_img

def create_parallel_jaw(img_height, img_width, rect_height, rect_width):
    binary_img = np.zeros([img_height, img_width])
    img_center = np.array([img_height / 2, img_width / 2])

    # create top
    rect_h_low = img_center[0] - rect_height / 2
    rect_h_high = img_center[0] + rect_height / 2
    rect_w_low = img_center[1] - rect_width / 2
    rect_w_high = img_center[1] + rect_width / 2
 
    binary_img[rect_h_low:rect_h_high, rect_w_low:rect_w_high] = 255

    # create mesh and sdf
    binary_img = Image.fromarray(binary_img.astype(np.uint8))
    mesh_converter = ts.MeshConverter2D()
    sdf, skel = mesh_converter.convert_binary_image_to_sdf(binary_img)
    binary_mesh, triangulation_succeeded = mesh_converter.convert_binary_image_to_mesh(binary_img)    
    return sdf, binary_mesh, binary_img

def random_poses():
    poses = []
    for i in range(6):
        T = np.eye(4)
        T[:3,3] = t
        T[:3,:3] = R_list[i]
        poses.append(T)
    random.shuffle(poses)
    return poses

def process_mesh(filename, camera_params, num_projections, im_size = 50, min_padding = 25, max_padding = 100):
    mesh_reader = of.ObjFile(filename)
    mesh_converter = ts.MeshConverter2D()
    m = mesh_reader.read()

    sdfs = []
    extruded_meshes = []
    images = []
    i = 0
    poses = random_poses()
    while i < 6 and len(sdfs) < num_projections:
        projection_succeeded = True

        # project into image
        pose = poses[i]
        proj_img = m.project_binary(camera_params, pose)
        binary_array = np.array(proj_img)

        # center_image around object
        row_ind, col_ind = np.where(binary_array > 0)
        if row_ind.shape[0] == 0 or col_ind.shape[0] == 0:
            projection_succeeded = False
            i = i+1
            continue

        min_row = np.min(row_ind)
        max_row = np.max(row_ind)
        min_col = np.min(col_ind)
        max_col = np.max(col_ind)

        # check that proj is not on borders
        if min_row <= 0 and min_col <= 0 and max_row >= binary_array.shape[0] - 1 and max_col >= binary_array.shape[1] - 1:
            projection_succeeded = False

        # compute padding
        padding = max_padding
        padding = min(padding, min_row) 
        padding = min(padding, binary_array.shape[0] - max_row) 
        padding = min(padding, min_col) 
        padding = min(padding, binary_array.shape[1] - max_col) 
        if padding < min_padding:
            projection_succeeded = False

        # center image
        min_row = min_row - padding
        max_row = max_row + padding
        min_col = min_col - padding
        max_col = max_col + padding
        binary_array_centered = binary_array[min_row:max_row, min_col:max_col]

        # resize
        resized_img = Image.fromarray(binary_array_centered)
        resized_img = resized_img.resize([int(im_size), int(im_size)],  PIL.Image.ANTIALIAS)

        # convert and store
        sdf, skel = mesh_converter.convert_binary_image_to_sdf(resized_img)
        extruded_mesh, triangulation_succeeded = mesh_converter.convert_binary_image_to_mesh(resized_img)
        if projection_succeeded and triangulation_succeeded:
            sdfs.append(sdf)
            extruded_meshes.append(extruded_mesh)
            images.append(resized_img)
        i = i+1

    # check for successs
    success = (len(sdfs) == num_projections)
    return sdfs, extruded_meshes, images, success

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 3:
        print 'Error: Not enough args'
        print 'Usage: python generate_caging_dataset.py <root_dir> <out_dir> <shapes_per_category>'
        print ''
        exit()
        
    # parse cmd line args
    root_dir = sys.argv[1]
    out_dir = sys.argv[2]
    val_size = 0.3
    test_size = 0.3
    im_size = 50
    shapes_per_cat = 1
    num_projections = 1
    if argc > 3:
        val_size = float(sys.argv[3])
    if argc > 4:
        test_size = float(sys.argv[4])
    if argc > 5:
        im_size = int(sys.argv[5])
    if argc > 6:
        shapes_per_cat = int(sys.argv[6])
    if argc > 7:
        num_projections = int(sys.argv[7])
    print 'Validation', val_size
    print 'Test', test_size

    train_list_filename = 'train_filenames.txt'
    val_list_filename = 'val_filenames.txt'
    test_list_filename = 'test_filenames.txt'
    template = 'poisson'

    # camera params
    cam_height = 500
    cam_width  = 500 
    cam_focal  = 10000.
    camera_params = mesh.CameraParams(cam_height, cam_width, cam_focal, cam_focal)    

    cat_counts = {}
    data_filenames = []

    # create parallel jaw gripper
    grip_width = im_size / 8
    grip_height = im_size / 12
    sdf, extruded_mesh, image = create_parallel_jaw(im_size, im_size, grip_height, grip_width)

    # write to directory
    out_gripper_filename = 'parallel_jaw'
    out_gripper_mesh_filename = os.path.join(out_dir, out_gripper_filename + '.obj')
    out_gripper_sdf_filename = os.path.join(out_dir, out_gripper_filename + '.csv')
    out_gripper_image_filename = os.path.join(out_dir, out_gripper_filename + '.jpg')
    mesh_writer = of.ObjFile(out_gripper_mesh_filename)
    mesh_writer.write(extruded_mesh)
    np.savetxt(out_gripper_sdf_filename, sdf, delimiter=',', header="%d %d"%(sdf.shape[0], sdf.shape[1]))
    image.save(out_gripper_image_filename)

    # loop through files from root dir and create cage data for each
    for root, dirs, files in os.walk(root_dir):
        # extract category, and skip ones known to cause issues
        extract_success, category = extract_category(root)
        if not extract_success:
            continue

        # set counts to zero
        print 'Category', category
        if category not in cat_counts.keys():
            cat_counts[category] = 0
        
        # parse files in this dir
        random.shuffle(files)
        i = 0
        while i < len(files) and cat_counts[category] < shapes_per_cat:
            name = files[i]
            for format in formats:
                if name.endswith(format) and name.find(template) > 0:
                    filename = os.path.join(root, name)
                    print 'Processing ', filename

                    # process the mesh with the number of camera parameters
                    sdfs, extruded_meshes, images, processing_succeeded = \
                        process_mesh(filename, camera_params, num_projections, im_size, min_padding = cam_height / 10)
                    
                    # save to output dir
                    if processing_succeeded:
                        fileroot, file_ext = os.path.splitext(name)
                        for j in range(len(sdfs)):
                            # gen filenames
                            new_filename = category + '_' + fileroot + '_%d'%(j)
                            out_mesh_filename = os.path.join(out_dir, new_filename + '.obj')
                            out_sdf_filename = os.path.join(out_dir, new_filename + '.csv')
                            out_image_filename = os.path.join(out_dir, new_filename + '.jpg')
                            
                            # write files
                            mesh_writer = of.ObjFile(out_mesh_filename)
                            mesh_writer.write(extruded_meshes[j])
                            np.savetxt(out_sdf_filename, sdfs[j], delimiter=',', header="%d %d"%(sdfs[j].shape[0], sdfs[j].shape[1]))
                            images[j].save(out_image_filename)
                            data_filenames.append(os.path.join(out_dir, new_filename))
                        cat_counts[category] += 1
            i += 1
    
    # randomly assign to train, test, validation
    random.shuffle(data_filenames)
    total_files = len(data_filenames)
    num_val = int(math.ceil(val_size * total_files))
    num_test = int(math.ceil(test_size * total_files))
    num_train = int(total_files - num_val - num_test)

    print 'Num train', num_train
    print 'Num val', num_val
    print 'Num test', num_test

    # write filenames to a .txt file
    train_list_file = open(os.path.join(out_dir, train_list_filename), 'w')
    for i in range(0, num_train):
        train_list_file.write(data_filenames[i] + '\n')
    train_list_file.close()

    val_list_file = open(os.path.join(out_dir, val_list_filename), 'w')
    for i in range(num_train, num_train + num_val):
        val_list_file.write(data_filenames[i] + '\n')
    val_list_file.close()

    test_list_file = open(os.path.join(out_dir, test_list_filename), 'w')
    for i in range(num_train + num_val, total_files):
        test_list_file.write(data_filenames[i] + '\n')
    test_list_file.close()
