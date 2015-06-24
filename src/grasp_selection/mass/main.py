from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.resources import resource_find
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from glob import glob
from random import shuffle
import pickle

# Grasp: general class for grasps (to be added to in other programs)
# Should match Grasp class/objects loaded from external file
class Grasp:
    def __init__(self, image_path, obj_file, label=0, scale=1, mass=1):
        self.image = image_path
        self.obj_file = obj_file
        self.label = label
        self.scale = scale
        self.mass = mass

    def mark_good(self):
        self.label = 1

    def mark_bad(self):
        self.label = -1

    def mark_undecided(self):
        self.label = 0

    def mark_scale(self, scale_val):
        self.scale = scale_val

    def mark_mass(self, mass):
        self.mass = mass


# NextButton: removes current images, shows new ones
class NextButton(Button):
    grasps = ObjectProperty(None)

    def on_press(self):
        remaining_grasps = self.grasps[1:]
        self.grasps = remaining_grasps
        gui.root.remove_widget(gui.current_image)
        gui.display_image(gui.next_grasp)
        # if remaining_grasps[0]:
        #     gui.renderer = Renderer(remaining_grasps[0])
        #     gui.root.add_widget(gui.renderer)

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


class MassApp(App):
    def build(self):
        root = self.root
        Config.set ('input','mouse','mouse,disable_multitouch') # disable multi-touch
        file_name = '../grasp_db/unlabeled_grasps.dat'
        file_object = open(file_name, 'rb') # binary encoding
        self.grasps = pickle.load(file_object)
        shuffle(self.grasps) # randomly sort images
        self.next_grasp = 0
        self.display_image(self.next_grasp)

        save_button = SaveButton(labeled_grasps = self.grasps)
        next_button = NextButton(grasps = self.grasps)
        help_button = HelpButton()
        text_input = TextInput(text='', size_hint = (.2, .05), pos_hint = {'top': .2, 'x': .6})
        compare_image = Image(pos_hint = {'top':.7,'x':.1}, size_hint = (.3, .6), source = 'bottle_comparison.jpg')
        root.add_widget(save_button)
        root.add_widget(next_button)
        root.add_widget(help_button)
        root.add_widget(text_input)
        root.add_widget(compare_image)
        return root

    def display_image(self, next_num):
        self.current_image = Image(source = self.grasps[next_num].image, pos_hint = {'top': .75, 'x': .5}, size_hint = (.4,.4))
        self.root.add_widget(self.current_image)
        self.next_grasp += 1

gui = MassApp()

if __name__ == "__main__":
    gui.run()
