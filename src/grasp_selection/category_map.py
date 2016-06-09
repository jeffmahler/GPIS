"""
Class to make indexing object categories from the filesystem easy
Authors: Jeff Mahler and Mel Roderick
"""
from abc import ABCMeta, abstractmethod
import os

# Invalid categories
INVALID_CATEGORIES = ['mnt', 'terastation', 'shape_data', 'textured_meshes', 'processed', '',
                      'amazon_picking_challenge', 'BigBIRD', 'YCB']

class CategoryMap:
    def __init__(self):
        self.object_dict_ = {}

    def category(self, key):
        """Get an object category for a given object key"""
        if key in self.object_dict:
            return self.object_dict[key]
        else:
            return None

    @property
    def object_keys(self):
        """Get a dictionary of objects: keys: object ids, values: object category"""
        return self.object_dict_.keys()

    @property
    def object_dict(self):
        """Get a dictionary of objects: keys: object ids, values: object category"""
        return self.object_dict_

class Cat50CategoryMap(CategoryMap):
    def __init__(self, path_to_root):
        CategoryMap.__init__(self)
        self.create_categories(path_to_root)

    def create_categories(self, path_to_root):
        for category in os.listdir(path_to_root):
            for key in os.listdir(os.path.join(path_to_root, category)):
                self.object_dict_[key] = category

class SHRECCategoryMap(CategoryMap):
    def __init__(self, path_to_index):
        self.sorted_keys_ = []
        CategoryMap.__init__(self)
        self.create_categories(path_to_index)

    def create_categories(self, path_to_index):
        with open(path_to_index) as file:
            for skip in xrange(3):
                next(file)

                current_cat = ''
                for line in file:
                    split = line.split()
                    if len(split) == 3:
                        current_cat = split[0]
                    elif len(split) == 1:
                        object_key = 'M'+split[0]
                        self.object_dict_[object_key] = current_cat
                        self.sorted_keys_.append(object_key)

    @property
    def sorted_keys(self):
        return self.sorted_keys_

class BerkeleyCategoryMap(CategoryMap):
    def category(self, key):
        head, tail = os.path.split(key)
        while head != '/' and tail in INVALID_CATEGORIES:
            head, tail = os.path.split(head)
        return tail
                
