"""
This example shows how to use a callback to select a red ball amongst white
balls.

The example uses the figure method 'on_mouse_pick', added in Mayavi
3.4, to register a callback when the left mouse is pressed. The callback
is called with a picker, enabling to identify the object selected.
Specificaly, actors are selected, each object is represented on the scene
via actors. The selected actors can be found in 'picker.actors'. In this
example, we have plotted red balls and white ball. We want to select the
red balls, and thus test if any actor in picker.actors corresponds to an
actor of red balls.

To identify which ball has been selected, we use the point id. However,
each ball is represented by several points. Thus we need to retrieve the
number of points per ball, and divide the point id by this number.

We use an outline to display which ball was selected by positioning it on
the corresponding ball.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup.org>
# Copyright (c) 2009, Enthought, Inc.
# License: BSD style.

import numpy as np
from mayavi import mlab
import time
import wx
import pdb
from tvtk.api import tvtk
import pdb
import mlab_3D_to_2D

################################################################################
# Disable the rendering, to get bring up the figure quicker:
figure = mlab.gcf()
mlab.clf()
figure.scene.disable_render = True

# Creates two set of points using mlab.points3d: red point and
# white points
x1, y1, z1 = np.random.random((3, 10))
red_glyphs = mlab.points3d(x1, y1, z1, color=(1, 0, 0),
                resolution=20)

x2, y2, z2 = np.random.random((3, 10))
white_glyphs = mlab.points3d(x2, y2, z2, color=(0.9, 0.9, 0.9),
                resolution=20)

# Add an outline to show the selected point and center it on the first
# data point.
outline = mlab.outline(line_width=3)
outline.outline_mode = 'full'
outline.bounds = (x1[0]-0.1, x1[0]+0.1,
                  y1[0]-0.1, y1[0]+0.1,
                  z1[0]-0.1, z1[0]+0.1)

# Every object has been created, we can reenable the rendering.
figure.scene.disable_render = False
################################################################################


# Here, we grab the points describing the individual glyph, to figure
# out how many points are in an individual glyph.
glyph_points = red_glyphs.glyph.glyph_source.glyph_source.output.points.to_array()


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

    W = np.ones(x1.shape)
    hmgns_world_coords = np.column_stack((x1, y1, z1, W))
    comb_trans_mat = mlab_3D_to_2D.get_world_to_view_matrix(figure.scene)
    view_coords = mlab_3D_to_2D.apply_transform_to_points(hmgns_world_coords, comb_trans_mat)
    norm_view_coords = view_coords / (view_coords[:, 3].reshape(-1, 1))
    view_to_disp_mat = mlab_3D_to_2D.get_view_to_display_matrix(figure.scene)
    disp_coords = mlab_3D_to_2D.apply_transform_to_points(norm_view_coords, view_to_disp_mat)

    for i in range(len(x1)):

        screen_x_cord = disp_coords[:, 0][i]
        #For some reason, the script we stole think the origin is the upper left. It's actually the 
        #lower left, so we need to adjust here!
        screen_y_cord = figure.scene.get_size()[1] - disp_coords[:, 1][i]
        
        x_in_box = screen_x_cord > min(box_start[0], box_end[0]) and screen_x_cord < max(box_start[0], box_end[0])
        y_in_box = screen_y_cord > min(box_start[1], box_end[1]) and screen_y_cord < max(box_start[1], box_end[1])

        if x_in_box and y_in_box:
            outline.bounds = (x1[i]-0.1, x1[i]+0.1,y1[i]-0.1, y1[i]+0.1, z1[i]-0.1, z1[i]+0.1)
            break


    
    #print(original_mouse_position)

    #def end_draw_box():


    #interactor.add_observer('KeyPressEvent', your_function)



    # while(True):
    #     current_mouse_posiiton = interactor.event_position
    #     print(current_mouse_posiiton)
    #     wx.Yield()
        
        
        




interactor.add_observer('LeftButtonPressEvent', box_bounding)
interactor.add_observer('MouseMoveEvent', draw_box)
interactor.add_observer('KeyPressEvent', select_mode)
# Decrease the tolerance, so that we can more easily select a precise
# point.



mlab.show()
