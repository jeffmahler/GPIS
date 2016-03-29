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
import pickle

other_code_dir = "../src/grasp_selection"
other_code_dir = os.getcwd() + "/" + other_code_dir
sys.path.insert(0, other_code_dir)

import obj_file as of
import mesh


def box_bounding(vtk_obj, event):
    """ Picker callback: this get called when on pick events.
    """
    global draw_box_triggered

    current_mouse_position = interactor.event_position

    if not draw_mode and not undo_mode:
        return

    if (not draw_box_triggered):
        draw_box_start_coords[0] = current_mouse_position[0]
        draw_box_start_coords[1] = current_mouse_position[1]
        draw_box_triggered = True

    else:
        project_3D(draw_box_start_coords, current_mouse_position)
        draw_box_triggered = False



def draw_box(vtk_obj, event):

    if not draw_mode and not undo_mode:
        return

    if draw_box_triggered:
        render_window.render()
        current_mouse_position = interactor.event_position
        num_shaded_pixels = (abs(current_mouse_position[0] - draw_box_start_coords[0]) + 1)*(abs(current_mouse_position[1] - draw_box_start_coords[1]) + 1)
        render_window.set_pixel_data(draw_box_start_coords[0], draw_box_start_coords[1], current_mouse_position[0], current_mouse_position[1], [1,1,1]*num_shaded_pixels, 1)


def select_mode(vtk_obj, event):
    global draw_mode
    global undo_mode
    global draw_box_triggered
    key_code = vtk_obj.GetKeyCode()

    #We want the "e" key to trigger a switch in the edit mode.
    if (key_code == 'e' and view_mode == 'l' and not undo_mode):
        if (not draw_mode): 

            mlab.title('LABEL MODE ON')
            draw_mode = True
            interactor.interactor_style = tvtk.InteractorStyleImage()

        else:
            mlab.title('')
            draw_box_triggered = False
            draw_mode = False
            interactor.interactor_style = default_interaction_style

    elif (key_code == 'u' and view_mode == 'l' and not draw_mode):
        if (not undo_mode):
            mlab.title('UNDO MODE ON')
            undo_mode = True
            interactor.interactor_style = tvtk.InteractorStyleImage()
        else:
            mlab.title('')
            draw_box_triggered = False
            undo_mode = False
            interactor.interactor_style = default_interaction_style


    #Write out a bianry mask for the vertices if the user decides to quit.
    elif (key_code == 'q'):

        if (view_mode == 'l' and colored_msh != None):
            save_file = open(save_mask_name, 'w')
            triangle_list = [original_triangles.index(t) for t in colored_msh.triangles()]
            np.save(save_file, triangle_list)
            save_file.close()

        sys.exit()



def project_3D(box_start, box_end):

    global msh
    global colored_msh

    x = np.array([i[0] for i in msh.vertices()])
    y = np.array([i[1] for i in msh.vertices()])
    z = np.array([i[2] for i in msh.vertices()])
    W = np.ones(x.shape)

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

    message = ""

    if (undo_mode):
        
        if (not colored_msh):
            return

        message = "UNDO MODE ON"
        triangles = colored_msh.triangles()
        undone_triangles, still_masked_triangles = triangle_grabber(triangles, in_vertex_indicies)
        msh = mesh.Mesh3D(msh.vertices(), msh.triangles() + undone_triangles, msh.normals())
        colored_msh = mesh.Mesh3D(msh.vertices(), still_masked_triangles, msh.normals())

    else:
        message = 'LABEL MODE ON'
        triangles = msh.triangles()
        removed_triangles, new_triangles = triangle_grabber(triangles, in_vertex_indicies)
        msh = mesh.Mesh3D(msh.vertices(), new_triangles, msh.normals())
        if not colored_msh and len(removed_triangles) > 0:
            colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles, msh.normals())
        else:
            colored_msh = mesh.Mesh3D(msh.vertices(), removed_triangles + colored_msh.triangles(), msh.normals())

    mlab.clf()
    msh.visualize((0.0, 0.0, 1.0))
    if colored_msh:
        colored_msh.visualize((1.0, 0.0, 0.0))

    mlab.title(message)

def triangle_grabber(triangles, vertex_list):

    #Label any triangle with A (even one) vertex in the bounding box
    # masked_triangles = [t for t in triangles if len(set(t).intersection(vertex_list)) != 0]
    # not_masked_traingles = [t for t in triangles if len(set(t).intersection(vertex_list)) == 0]

    #Label only triangles that have EVERY vertex in the bounding box.
    masked_triangles = [t for t in triangles if len(set(t).intersection(vertex_list)) == 3]
    not_masked_traingles = [t for t in triangles if len(set(t).intersection(vertex_list)) < 3]
    return (masked_triangles, not_masked_traingles)

def load_saved_mask(mask_file):

    global msh
    global colored_msh

    read_file = open(mask_file, 'r')
    triangle_index_list = np.load(read_file)
    read_file.close()
    masked_triangles = [msh.triangles()[i] for i in triangle_index_list]
    non_masked_triangles = [msh.triangles()[i] for i in range(len(msh.triangles())) if i not in triangle_index_list]
    msh = mesh.Mesh3D(msh.vertices(), non_masked_triangles, msh.normals())
    colored_msh = mesh.Mesh3D(msh.vertices(), masked_triangles, msh.normals())
    mlab.clf()
    msh.visualize((0.0, 0.0, 1.0))
    colored_msh.visualize((1.0, 0.0, 0.0))


#APPLICATION CORE


################################################################################
# Disable the rendering, to get bring up the figure quicker:
figure = mlab.gcf()
mlab.clf()
figure.scene.disable_render = True

#Arguemnts exptected: pythonw sniyaz_lableing_tool mode mesh_to_load output_file_name 
#mode: l for labeling, r to read mask
#output_file_name: if labeling the loaded mesh, as opposed to just reading it

#Load mesh here
view_mode = sys.argv[1]
mesh_filename = os.path.abspath(sys.argv[2])
save_mask_name = sys.argv[3]


ofile = of.ObjFile(mesh_filename)
msh = ofile.read()
colored_msh = None
msh.visualize((0.0, 0.0, 1.0))



# Every object has been created, we can reenable the rendering.
figure.scene.disable_render = False
################################################################################


if (view_mode == 'r'):
    msh, colored_msh = load_saved_mask()


engine = mlab.get_engine()
scene = engine.scenes[0]
vtk_scene = scene.scene
interactor = vtk_scene.interactor
render_window = vtk_scene.render_window

draw_box_start_coords = [0]*2
draw_box_triggered = False
draw_mode = False
undo_mode = False
default_interaction_style = interactor.interactor_style
original_triangles = msh.triangles()




interactor.add_observer('LeftButtonPressEvent', box_bounding)
interactor.add_observer('MouseMoveEvent', draw_box)
interactor.add_observer('KeyPressEvent', select_mode)




mlab.show()
