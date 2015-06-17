import os
import sys

from PIL import Image, ImageDraw
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import mesh
import obj_file as obj
import triangulate_silhouette as ts

import IPython

from collections import namedtuple

EXTENSIONS = ['.png', '.jpg']

class ParallelJawGrasp(object):
    def __init__(self, x, y, theta, width):
        self.x = x
        self.y = y
        self.theta = theta
        self.width = width

def point_in_polygon(target, poly):
    """x,y is the point to test. poly is a list of tuples comprising the polygon."""
    point = namedtuple("Point", ("x", "y"))
    line = namedtuple("Line", ("p1", "p2"))
    target = point(*target)

    inside = False
    # Build list of coordinate pairs
    # First, turn it into named tuples

    poly = map(lambda p: point(*p), poly)

    # Make two lists, with list2 shifted forward by one and wrapped around
    list1 = poly
    list2 = poly[1:] + [poly[0]]
    poly = map(line, list1, list2)

    for l in poly:
        p1 = l.p1
        p2 = l.p2

        if p1.y == p2.y:
            # This line is horizontal and thus not relevant.
            continue
        if max(p1.y, p2.y) < target.y <= min(p1.y, p2.y):
            # This line is too high or low
            continue
        if target.x < max(p1.x, p2.x):
            # Ignore this line because it's to the right of our point
            continue
        # Now, the line still might be to the right of our target point, but 
        # still to the right of one of the line endpoints.
        rise = p1.y - p2.y
        run =  p1.x - p2.x
        try:
            slope = rise/float(run)
        except ZeroDivisionError:
            slope = float('inf')

        # Find the x-intercept, that is, the place where the line we are
        # testing equals the y value of our target point.

        # Pick one of the line points, and figure out what the run between it
        # and the target point is.
        run_to_intercept = target.x - p1.x
        x_intercept = p1.x + run_to_intercept / slope
        if target.x < x_intercept:
            # We almost crossed the line.
            continue

        inside = not inside

    return inside

def trace_image(trace_im, im_size, scale):
    trace_im_array = np.array(trace_im)

    # get user input coords
    user_input = [(0,0)]
    coords = []
    plt.figure(1)
    plt.clf()

    while len(user_input) > 0:

        # plot figure
        plt.figure(1)
        plt.imshow(trace_im_array, extent=[0, im_size, 0, im_size])

        # plot coords for interactivity
        for i in range(len(coords)-1):
            x = np.array([coords[i][0], coords[i+1][0]])
            y = np.array([coords[i][1], coords[i+1][1]])
            plt.scatter(coords[i][0], coords[i][1], marker = '*', c = 'y', s = 200)
            plt.plot(x, y, 'y', linewidth=5)

        if len(coords) > 0:
            plt.scatter(coords[-1][0], coords[-1][1], marker = '*', c = 'y', s = 200)

        # update
        user_input = plt.ginput()
        if len(user_input) > 0:
            user_input[0] = (round(user_input[0][0]), round(user_input[0][1]))
            coords.extend(user_input)

    # plot final shape
    plt.figure(1)
    plt.imshow(trace_im_array, extent=[0, im_size, 0, im_size])
    for i in range(len(coords)-1):
        x = np.array([coords[i][0], coords[i+1][0]])
        y = np.array([coords[i][1], coords[i+1][1]])
        plt.scatter(coords[i][0], coords[i][1], marker = '*', c = 'y', s = 200)
        plt.plot(x, y, 'y', linewidth=5)
        
    if len(coords) > 0:
        x = np.array([coords[-1][0], coords[0][0]])
        y = np.array([coords[-1][1], coords[0][1]])
        plt.scatter(coords[-1][0], coords[-1][1], marker = '*', c = 'y', s = 200)
        plt.plot(x, y, 'y', linewidth=5)
        
    # create binary image from poly
    new_im_size = (scale*im_size, scale*im_size)
    poly = []
    for c in coords:
        p = (scale * int(c[0]), new_im_size[1] - scale * int(c[1]))
        poly.append(p)
            
    binary_im = Image.new('L', new_im_size, 0)
    ImageDraw.Draw(binary_im).polygon(poly, outline=1, fill=1)
    binary_im = binary_im.resize((im_size, im_size), Image.ANTIALIAS)
    binary_im_arr = 255*np.array(binary_im)

    plt.figure(2)
    plt.imshow(binary_im_arr, cmap = cm.Greys_r)
    plt.show()

    new_binary_im = Image.fromarray(np.uint8(binary_im_arr))    
    
    return poly, new_binary_im    

def trace_grasp_axes(trace_im, im_size, scale):
    trace_im_array = np.array(trace_im)

    # get user input coords
    user_input = [(0,0)]
    coords = []
    plt.figure(1)
    plt.clf()

    while len(user_input) > 0:

        # plot figure
        plt.figure(1)
        plt.imshow(trace_im_array, extent=[0, im_size, 0, im_size])

        # plot coords for interactivity
        for i in range(len(coords)-1):
            x = np.array([coords[i][0], coords[i+1][0]])
            y = np.array([coords[i][1], coords[i+1][1]])
            plt.scatter(coords[i][0], coords[i][1], marker = '*', c = 'y', s = 200)

            # lines between every other
            if i % 2 == 0:
                plt.plot(x, y, 'y', linewidth=5)

        if len(coords) > 0:
            plt.scatter(coords[-1][0], coords[-1][1], marker = '*', c = 'y', s = 200)

        # update
        user_input = plt.ginput()
        if len(user_input) > 0:
            user_input[0] = (round(user_input[0][0]), round(user_input[0][1]))
            coords.extend(user_input)

    # compute grasp centers and angles
    grasps = []
    for i in range(0, len(coords), 2):
        # center vertices and compute grasp
        x1 = np.array(coords[i]) - im_size / 2
        x2 = np.array(coords[i+1]) - im_size / 2
        center = (x1 + x2) / 2
        center[0] = -center[0]
        v = x2 - x1
        angle = np.arctan(v[0] / v[1])
        width = np.linalg.norm(v)
        grasp = ParallelJawGrasp(center[1], center[0], angle, width)
        grasps.append(grasp)
    return grasps

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 2:
        print 'Error: Filename of in dir and out dir required'

    root_dir = sys.argv[1]
    out_dir = sys.argv[2]
    im_size = 100
    scale = 1
    if argc > 3:
        im_size = int(sys.argv[3])
    if argc > 4:
        scale = int(sys.argv[4])

    # load image
    data_filenames = []
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            for ext in EXTENSIONS:
                if name.endswith(ext):
                    print 'Tracing image', name
                    
                    # open image
                    filename, ext = os.path.splitext(name)
                    image_filename = os.path.join(root, name)
                    trace_im = Image.open(image_filename)

                    # get mesh for object
                    poly, new_binary_im = trace_image(trace_im, im_size, scale)

                    # convert to a mesh
                    mesh_converter = ts.MeshConverter2D()
                    binary_mesh, triangulation_succeeded = mesh_converter.convert_binary_image_to_mesh(new_binary_im)
                    binary_sdf, skel = mesh_converter.convert_binary_image_to_sdf(new_binary_im)
                    if not triangulation_succeeded:
                        print 'Failed to triangulate polygon'
                        exit()

                    # get grasps
                    grasps = trace_grasp_axes(trace_im, im_size, scale)

                    # write results to file
                    mesh_filename = '%s.obj' %(filename)
                    sdf_filename = '%s.csv' %(filename)
                    grasp_filename = '%s_grasps.txt' %(filename)
                    mesh_filename = os.path.join(out_dir, mesh_filename)
                    sdf_filename = os.path.join(out_dir, sdf_filename)
                    grasp_filename = os.path.join(out_dir, grasp_filename)

                    of = obj.ObjFile(mesh_filename)
                    of.write(binary_mesh)
                    np.savetxt(sdf_filename, binary_sdf, delimiter=',', header="%d %d"%(binary_sdf.shape[0], binary_sdf.shape[1]))
                    data_filenames.append(os.path.join(out_dir, filename))
                    
                    grasp_file = open(grasp_filename, 'w')
                    for grasp in grasps:
                        grasp_file.write('%.4f %.4f %.4f %.4f\n'%(grasp.x, grasp.y, grasp.theta, grasp.width))
                    grasp_file.close()

    # write filenames file
    num_files = len(data_filenames)
    list_filename = 'filenames.txt'
    list_file = open(os.path.join(out_dir, list_filename), 'w')
    for i in range(0, num_files):
        list_file.write(data_filenames[i] + '\n')
    list_file.close()
