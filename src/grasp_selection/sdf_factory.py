'''
Facilitates the creation of SDFs from multiple methods
Author: Jeff Mahler
'''
import sdf_file

class SdfFactory():
    def __init__(self):
        todo = 1

    def sdfFromFile(self, file_name):
        sf = sdf_file.SdfFile(file_name)
        return sf.read()

    def sdfFromImages(self, color_images, depth_images):
        '''
        Creates an sdf from 2D color and depth images
        '''
        todo = 1
