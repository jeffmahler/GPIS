from abc import ABCMeta, abstractmethod

class ObjectDatabase:
	@abstractmethod
	def categories(self):
		"""Get a dictionary of categories: keys: object categories, values: list of object ids"""
		pass

	@abstractmethod
	def object_ids(self):
		"""Get a dictionary of objects: keys: object ids, values: object category"""
		pass

class Cat50ObjectDatabase(ObjectDatabase):
	def __init__(self, path_to_index):
		self.categories_, self.object_ids_ = self.create_categories(path_to_index)

	def create_categories(self, path_to_index):
		object_ids = {}
		categories = {}
		with open(path_to_index) as file:
			for line in file:
				split = line.split()
				file_id = split[0]
				category = split[1]
				if category in categories:
					categories[category].append(file_id)
				else:
					categories[category] = [file_id]
				if file_id in object_ids:
					object_ids[file_id].append(category)
				else:
					object_ids[file_id] = [category]
		return categories, object_ids

	def categories(self):
		return self.categories_

	def object_ids(self):
		return self.object_ids_