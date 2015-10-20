# Formatting for all classes in graspselection.kv
# This file must be named main.py to work
# GraspSelectionApp (near end) is the central class

from kivy.app import App	
from kivy.uix.gridlayout import GridLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.properties import StringProperty, NumericProperty, ObjectProperty
from kivy.config import Config
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


# select_animate: turns ImageButtons green/red when selected
def select_animate(instance, rgba, speed):
	animation = Animation(background_color=rgba, d=speed)
	animation.start(instance)


# ImageButton: image objects in grid - all same size
class ImageButton(Button):
	source = StringProperty(None) #instance attributes (just how kivy works)
	grasp = ObjectProperty(None)

	# callback - triggered when button is pressed
	def on_press(self):
		if self.grasp.label == 0:
			self.grasp.mark_good()
			select_animate(self,[.5,1,.5,1],.4)
		elif self.grasp.label == 1:
			self.grasp.mark_bad()
			select_animate(self,[1,.5,.5,1],.4)
		else:
			self.grasp.mark_undecided()
			select_animate(self,[1,1,1,1],.4)
			

# SelectionGrid: grid for 16 ImageButton objects
class SelectionGrid(GridLayout):
	pass


# HelpButton: opens help screen when clicked
class HelpButton(Button):
	def on_release(self):
		help_screen = HelpScreenButton()
		gui.root.add_widget(help_screen)


# HelpScreenButton: Displays help stuff, closes when clicked
class HelpScreenButton(Button):
	def on_release(self):
		gui.root.remove_widget(self)


# NextButton: removes current images, shows new ones
class NextButton(Button):
	old_layout = ObjectProperty(None)
	grasps = ObjectProperty(None)

	def on_press(self):
		remaining_grasps = self.grasps[16:]
		self.grasps = remaining_grasps
		disappear_image = Animation(background_color=[0,0,0,0], d=.8) # fade out
		for button in range(len(gui.current_buttons)):
			disappear_image.start(gui.current_buttons[button])
		gui.root.remove_widget(self.old_layout)
		new_layout = gui.display_images(self.grasps)
		gui.root.add_widget(new_layout)

# SaveButton: saves grasp objects into file_name
class SaveButton(Button):
	source = StringProperty(None)
	labeled_grasps = ObjectProperty(None)

	def on_press(self):
		file_name = "labeled_grasps/labeled_grasps.dat"
		file_object = open(file_name, 'wb')
		pickle.dump(self.labeled_grasps, file_object)
		file_object.close()
		disappear_button = Animation(background_color=[0,0,0,0], d=.4) + Animation(background_color=[.6,.6,.6,1], d=.4) 
		disappear_button.start(gui.save_button)


# FinalSaveButton: pops when 'Next' is pressed and there are no more grasps
class FinalSaveButton(Button):
	def __init__(self):
		super(FinalSaveButton, self).__init__()
		appear_button = Animation(background_color=[1,1,1,1], d=1) # fade in
		appear_button.start(self)

	def on_press(self):
		self.text = 'Thank you!'
		file_name = "labeled_grasps.dat"
		file_object = open(file_name, 'wb')
		pickle.dump(gui.grasps, file_object)
		file_object.close()


# GraspSelectionApp: Builds application, central class
class GraspSelectionApp(App):
	current_buttons = ObjectProperty([])
	grasps = ObjectProperty(None)

	# build: runs at beginning (doesn't need to be called)
	def build(self):
		root = self.root
		Config.set ('input','mouse','mouse,disable_multitouch') # disable multi-touch
		file_name = 'unlabeled_grasps.dat'
		file_object = open(file_name, 'rb') # binary encoding
		self.grasps = pickle.load(file_object)
		shuffle(self.grasps) # randomly sort images
		layout = self.display_images(self.grasps) # grid of images
		root.add_widget(layout)

		# create and add buttons
		help_button = HelpButton()
		next_button = NextButton(old_layout = layout, grasps = self.grasps)
		self.save_button = SaveButton(labeled_grasps = self.grasps)
		root.add_widget(help_button)
		root.add_widget(next_button)
		root.add_widget(self.save_button)
		return root

	# display_images: makes new grid of images
	def display_images(self, remaining_grasps):
		appear_image = Animation(background_color=[1,1,1,1], d=.8) # fade in
		layout = SelectionGrid(cols=4, pos_hint = {'center_x':.5, 'top':.8}, size_hint = (.75,.75))
		self.current_buttons = []
		num_images = len(remaining_grasps)
		display_num = min(num_images,16)
		for i in range(display_num):
			self.current_buttons.append(ImageButton(grasp = remaining_grasps[i]))
			layout.add_widget(self.current_buttons[i])
			appear_image.start(self.current_buttons[i])
		if self.current_buttons == []: # if no more images to show
			layout = FinalSaveButton()
		return layout
	

gui = GraspSelectionApp()

# makes gui
if __name__ == '__main__':
	gui.run()