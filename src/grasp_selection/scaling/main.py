'''
3D Rotating Monkey Head
========================

This example demonstrates using OpenGL to display a rotating monkey head. This
includes loading a Blender OBJ file, shaders written in OpenGL's Shading
Language (GLSL), and using scheduled callbacks.

The monkey.obj file is an OBJ file output from the Blender free 3D creation
software. The file is text, listing vertices and faces and is loaded
using a class in the file objloader.py. The file simple.glsl is
a simple vertex and fragment shader written in GLSL.
'''

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.resources import resource_find
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.graphics.transformation import Matrix
from kivy.graphics.opengl import *
from kivy.graphics import *
from objloader import ObjFile
from glob import glob
from random import shuffle
import pickle

# Grasp: general class for grasps (to be added to in other programs)
# Should match Grasp class/objects loaded from external file
class Grasp:
    def __init__(self, image_path, obj_file, label=0, scale=1):
        self.image = image_path
        self.obj_file = obj_file
        self.label = label
        self.scale = scale

    def mark_good(self):
        self.label = 1

    def mark_bad(self):
        self.label = -1

    def mark_undecided(self):
        self.label = 0

    def mark_scale(self, scale_val):
        self.scale = scale_val


class Renderer(Widget):
    def __init__(self, grasp, **kwargs):
        self.grasp = grasp
        self.canvas = RenderContext(compute_normasl_mat=True)
        self.canvas.shader.source = resource_find('simple.glsl')
        self.scene = ObjFile(resource_find(self.grasp.obj_file))
        super(Renderer, self).__init__(**kwargs)
        with self.canvas:
            self.cb = Callback(self.setup_gl_context)
            PushMatrix()
            self.setup_scene()
            PopMatrix()
            self.cb = Callback(self.reset_gl_context)
        self.update_glsl_cw()
        #Clock.schedule_interval(self.update_glsl_cw, 1 / 60.)

    def setup_gl_context(self, *args):
        glEnable(GL_DEPTH_TEST)

    def reset_gl_context(self, *args):
        glDisable(GL_DEPTH_TEST)

    def update_glsl_zoom_in(self, *args):
        self.zoom.scale += .005

    def update_glsl_zoom_out(self, *args):
        if self.zoom.scale > 0:
            self.zoom.scale -= .005

    def update_glsl_cw(self, *args):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['projection_mat'] = proj
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.rot.angle -= 1

    def update_glsl_ccw(self, *args):
        asp = self.width / float(self.height)
        proj = Matrix().view_clip(-asp, asp, -1, 1, 1, 100, 1)
        self.canvas['projection_mat'] = proj
        self.canvas['diffuse_light'] = (1.0, 1.0, 0.8)
        self.canvas['ambient_light'] = (0.1, 0.1, 0.1)
        self.rot.angle += 1

    def setup_scene(self):
        Color(1, 1, 1, 1)
        PushMatrix()
        Translate(0, 0, -3)
        Translate(1.4,0,0)
        self.rot = Rotate(1, 0, 1, 0)
        self.zoom = Scale(1)
        m = list(self.scene.objects.values())[0]
        UpdateNormalMatrix()
        self.mesh = Mesh(
            vertices=m.vertices,
            indices=m.indices,
            fmt=m.vertex_format,
            mode='triangles',
        )
        PopMatrix()

# NextButton: removes current images, shows new ones
class NextButton(Button):
    grasps = ObjectProperty(None)

    def on_press(self):
        remaining_grasps = self.grasps[1:]
        self.grasps = remaining_grasps
        gui.root.remove_widget(gui.renderer)
        if remaining_grasps[0]:
            gui.renderer = Renderer(remaining_grasps[0])
            gui.root.add_widget(gui.renderer)

# SaveButton: saves grasp objects into file_name
class SaveButton(Button):
    source = StringProperty(None)
    labeled_grasps = ObjectProperty(None)

    def on_press(self):
        file_name = "../grasp_db/labeled_grasps.dat"
        file_object = open(file_name, 'wb')
        pickle.dump(self.labeled_grasps, file_object)
        file_object.close()

# HelpButton: opens help screen when clicked
class HelpButton(Button):
    def on_release(self):
        help_screen = HelpScreenButton()
        gui.root.add_widget(help_screen)


# HelpScreenButton: Displays help stuff, closes when clicked
class HelpScreenButton(Button):
    def on_release(self):
        gui.root.remove_widget(self)

class ZoomInButton(Button):
    def on_press(self):
        Clock.schedule_interval(gui.renderer.update_glsl_zoom_in, 1 / 60.)

    def on_release(self):
        Clock.unschedule(gui.renderer.update_glsl_zoom_in)
        gui.renderer.grasp.mark_scale(gui.renderer.zoom.scale)


class ZoomOutButton(Button):
    def on_press(self):
        Clock.schedule_interval(gui.renderer.update_glsl_zoom_out, 1 / 60.)

    def on_release(self):
        Clock.unschedule(gui.renderer.update_glsl_zoom_out)
        gui.renderer.grasp.mark_scale(gui.renderer.zoom.scale)      


class RotCWButton(Button):
    def on_press(self):
        #self.update_glsl()
        Clock.schedule_interval(gui.renderer.update_glsl_cw, 1 / 60.)

    def on_release(self):
        Clock.unschedule(gui.renderer.update_glsl_cw)


class RotCCWButton(Button):
    def on_press(self):
        #self.update_glsl()
        Clock.schedule_interval(gui.renderer.update_glsl_ccw, 1 / 60.)

    def on_release(self):
        Clock.unschedule(gui.renderer.update_glsl_ccw)


class RendererApp(App):
    def build(self):
        root = self.root
        Config.set ('input','mouse','mouse,disable_multitouch') # disable multi-touch
        file_name = '../grasp_db/unlabeled_grasps.dat'
        file_object = open(file_name, 'rb') # binary encoding
        self.grasps = pickle.load(file_object)
        shuffle(self.grasps) # randomly sort images

        self.renderer = Renderer(self.grasps[0])
        save_button = SaveButton(labeled_grasps = self.grasps)
        next_button = NextButton(grasps = self.grasps)
        help_button = HelpButton()
        rot_buttonCW = RotCWButton(pos_hint = {'x':.58,'y':.15}, size_hint = (.1,.07), text = 'CW')        
        rot_buttonCCW = RotCCWButton(pos_hint = {'x':.7,'y':.15}, size_hint = (.1,.07), text = 'CCW')
        zoom_in_button = ZoomInButton(pos_hint = {'x':.58,'y':.05}, size_hint = (.1,.07), text = 'Larger')
        zoom_out_button = ZoomOutButton(pos_hint = {'x':.7,'y':.05}, size_hint = (.1,.07), text = 'Smaller')
        compare_image = Image(pos_hint = {'top':.7,'x':.1}, size_hint = (.3, .2), source = 'shoe_compare.png')
        compare_image2 = Image(pos_hint = {'top':.5,'x':.15}, size_hint = (.2, .5), source = 'shoe_compare2.png')
        root.add_widget(self.renderer)
        root.add_widget(save_button)
        root.add_widget(next_button)
        root.add_widget(help_button)
        root.add_widget(zoom_in_button)
        root.add_widget(zoom_out_button)
        root.add_widget(rot_buttonCW)
        root.add_widget(rot_buttonCCW)
        root.add_widget(compare_image)
        root.add_widget(compare_image2)
        return root

gui = RendererApp()

if __name__ == "__main__":
    gui.run()
