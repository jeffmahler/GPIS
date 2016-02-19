import numpy as np
from mayavi import mlab
import time
import wx
import pdb
from tvtk.api import tvtk
import pdb
import mlab_3D_to_2D
import os
import sys

other_code_dir = "../src/grasp_selection"
other_code_dir = os.getcwd() + "/" + other_code_dir
sys.path.insert(0, other_code_dir)

import obj_file as of
import mesh

################################################################################
# Disable the rendering, to get bring up the figure quicker:
figure = mlab.gcf()
mlab.clf()
figure.scene.disable_render = True

#Load mesh here

filename = "/Users/Sherdil/Research/GPIS/data/test/meshes/Co_clean.obj"
ofile = of.ObjFile(filename)
msh = ofile.read()
colored_msh = None


msh.visualize((0.0, 0.0, 1.0))



# Every object has been created, we can reenable the rendering.
figure.scene.disable_render = False
################################################################################




#Sherdil Code start


engine = mlab.get_engine()
scene = engine.scenes[0]
vtk_scene = scene.scene
interactor = vtk_scene.interactor
render_window = vtk_scene.render_window

draw_box_start_coords = [0]*2
draw_box_triggered = False
draw_mode = False
default_interaction_style = interactor.interactor_style



def box_bounding(vtk_obj, event):
    """ Picker callback: this get called when on pick events.
    """
    global draw_box_triggered

    current_mouse_position = interactor.event_position

    if not draw_mode:
        return

    if (not draw_box_triggered):
        draw_box_start_coords[0] = current_mouse_position[0]
        draw_box_start_coords[1] = current_mouse_position[1]
        draw_box_triggered = True

    else:
        project_3D(draw_box_start_coords, current_mouse_position)
        draw_box_triggered = False



def draw_box(vtk_obj, event):

    if not draw_mode:
        return

    if draw_box_triggered:
        render_window.render()
        current_mouse_position = interactor.event_position
        num_shaded_pixels = (abs(current_mouse_position[0] - draw_box_start_coords[0]) + 1)*(abs(current_mouse_position[1] - draw_box_start_coords[1]) + 1)
        render_window.set_pixel_data(draw_box_start_coords[0], draw_box_start_coords[1], current_mouse_position[0], current_mouse_position[1], [1,1,1]*num_shaded_pixels, 1)


def select_mode(vtk_obj, event):
    global draw_mode
    key_code = vtk_obj.GetKeyCode()

    #We want the "e" key to trigger a switch in the edit mode.
    if (key_code != 'e'):
        return
    
    if (not draw_mode): 

        pdb.set_trace

        mlab.title('LABEL MODE ON')
        draw_mode = True
        interactor.interactor_style = tvtk.InteractorStyleImage()

    else:
        mlab.title('')
        draw_mode = False
        interactor.interactor_style = default_interaction_style



def project_3D(box_start, box_end):

    global msh
    global colored_msh

    x = np.array([i[0] for i in msh.vertices()])
    y = np.array([i[1] for i in msh.vertices()])
    z = np.array([i[2] for i in msh.vertices()])
    W = np.ones(x.shape)
    triangles = msh.triangles()

    hmgns_world_coords = np.column_stack((x, y, z, W))
    comb_trans_mat = mlab_3D_to_2D.get_world_to_view_matrix(figure.scene)
    view_coords = mlab_3D_to_2D.apply_transform_to_points(hmgns_world_coords, comb_trans_mat)
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
    view_to_disp_mat = mlab_3D_to_2D.get_view_to_display_matrix(figure.scene)
    disp_coords = mlab_3D_to_2D.apply_transform_to_points(norm_view_coords, view_to_disp_mat)

    closest_triangles = np.empty((figure.scene.get_size()[0], figure.scene.get_size()[1]), dtype="object")

    in_triangle_indicies = []

    for t in range(len(triangles)):

        cur_triangle = triangles[t]
        screen_x_data = []
        screen_y_data = []

        for i in cur_triangle:

            screen_x_cord = disp_coords[:, 0][i]
            #For some reason, the script we stole thinks the origin is the upper left. It's actually the 
            #lower left, so we need to adjust here!
            screen_y_cord = figure.scene.get_size()[1] - disp_coords[:, 1][i]

            screen_x_data.append(screen_x_cord)
            screen_y_data.append(screen_y_cord)

        screen_cord_A = (screen_x_data[0], screen_y_data[0])
        screen_cord_B = (screen_x_data[1], screen_y_data[1])
        screen_cord_C = (screen_x_data[2], screen_y_data[2])

        valid_triangle = False
        for coord in [screen_cord_A, screen_cord_B, screen_cord_C]:

            if bounding_box_check(coord[0], coord[1], box_start, box_end):
                valid_triangle = True


        if not valid_triangle:
            continue


        #find "bounding box" for this triangle on the display screen.
        min_screen_x = int(min(screen_x_data))
        max_screen_x = int(max(screen_x_data))
        min_screen_y = int(min(screen_y_data))
        max_screen_y = int(max(screen_y_data))

        for x in range(min_screen_x, max_screen_x+1):
            for y in range(min_screen_y, max_screen_y+1):

                in_triangle = in_triangle_checker(screen_cord_A, screen_cord_B, screen_cord_C, (x, y))
                in_bounding_box = bounding_box_check(x, y, box_start, box_end)

                if in_triangle and in_bounding_box:

                    previous_closest_triangle = closest_triangles[x, y]
                    z_depth = max([norm_view_coords[:, 2][i] for i in cur_triangle])
                    if (not previous_closest_triangle) or (previous_closest_triangle and z_depth < previous_closest_triangle[1]):
                        closest_triangles[x, y] = (t, z_depth)


    for i in range(figure.scene.get_size()[0]):
        for j in range(figure.scene.get_size()[1]):
            if closest_triangles[i, j]:
                closest_traingle_index = closest_triangles[i, j][0]
                in_triangle_indicies.append(closest_traingle_index)


    removed_triangles = [triangles[t] for t in in_triangle_indicies]
    new_triangles = [triangles[t] for t in range(len(triangles)) if t not in in_triangle_indicies]

    msh = mesh.Mesh3D(msh.vertices(), new_triangles, msh.normals())
    if not colored_msh and len(removed_triangles) > 0:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles, msh.normals())
    elif colored_msh:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles + colored_msh.triangles(), msh.normals())

    mlab.clf()
    msh.visualize((0.0, 0.0, 1.0))
    if colored_msh:
        colored_msh.visualize((1.0, 0.0, 0.0))

    mlab.title('LABEL MODE ON')



def in_triangle_checker(vertex_A, vertex_B, vertex_C, checked_point):

    A = np.array(list(vertex_A) + [0])
    B = np.array(list(vertex_B) + [0])
    C = np.array(list(vertex_C) + [0])
    P = np.array(list(checked_point) + [0])

    w = np.subtract(P, A)
    u = np.subtract(B, A)
    v = np.subtract(C, A)

    v_cross_w = np.cross(v, w)
    v_cross_u = np.cross(v, u)
    u_cross_w = np.cross(u, w)
    u_cross_v = np.cross(u, v)

    sign_r = np.dot(v_cross_w , v_cross_u)
    sign_t = np.dot(u_cross_w, u_cross_v)

    if sign_r < 0 or sign_t < 0:
        return False

    norm_v_cross_w = np.linalg.norm(v_cross_w)
    norm_u_cross_w = np.linalg.norm(u_cross_w)
    norm_v_cross_u = np.linalg.norm(v_cross_u)

    r = norm_v_cross_w/norm_v_cross_u
    t = norm_u_cross_w/norm_v_cross_u

    return (r + t <= 1)


def bounding_box_check(x, y, box_start, box_end):

    x_in_box = x > min(box_start[0], box_end[0]) and x < max(box_start[0], box_end[0])
    y_in_box = y > min(box_start[1], box_end[1]) and y < max(box_start[1], box_end[1])

    return x_in_box and y_in_box


interactor.add_observer('LeftButtonPressEvent', box_bounding)
interactor.add_observer('MouseMoveEvent', draw_box)
interactor.add_observer('KeyPressEvent', select_mode)




mlab.show()
