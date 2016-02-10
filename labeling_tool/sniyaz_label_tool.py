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

    in_vertex_indicies = []
    for i in range(len(x)):

        screen_x_cord = disp_coords[:, 0][i]
        #For some reason, the script we stole thinks the origin is the upper left. It's actually the 
        #lower left, so we need to adjust here!
        screen_y_cord = figure.scene.get_size()[1] - disp_coords[:, 1][i]
        
        x_in_box = screen_x_cord > min(box_start[0], box_end[0]) and screen_x_cord < max(box_start[0], box_end[0])
        y_in_box = screen_y_cord > min(box_start[1], box_end[1]) and screen_y_cord < max(box_start[1], box_end[1])

        if x_in_box and y_in_box:
            in_vertex_indicies.append(i)


    #Label any triangle with A (even one) vertex in the bounding box
    #removed_triangles = [t for t in triangles if len(set(t).intersection(in_vertex_indicies)) != 0]
    #new_triangles = [t for t in triangles if len(set(t).intersection(in_vertex_indicies)) == 0]

    #Label only triangles that have EVERY vertex in the bounding box.
    removed_triangles = [t for t in triangles if len(set(t).intersection(in_vertex_indicies)) == 3]
    new_triangles = [t for t in triangles if len(set(t).intersection(in_vertex_indicies)) < 3]


    msh = mesh.Mesh3D(msh.vertices(), new_triangles, msh.normals())
    if not colored_msh and len(removed_triangles) > 0:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles, msh.normals())
    else:
        colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles + colored_msh.triangles(), msh.normals())

    mlab.clf()
    msh.visualize((0.0, 0.0, 1.0))
    if colored_msh:
        colored_msh.visualize((1.0, 0.0, 0.0))

    mlab.title('LABEL MODE ON')



interactor.add_observer('LeftButtonPressEvent', box_bounding)
interactor.add_observer('MouseMoveEvent', draw_box)
interactor.add_observer('KeyPressEvent', select_mode)




mlab.show()
