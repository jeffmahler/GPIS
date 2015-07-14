def scatter_plot_feature_objects(plt, feature_objects, color):
	x = map(lambda obj: obj.feature_vector[0], feature_objects)
	y = map(lambda obj: obj.feature_vector[1], feature_objects)
	return plt.scatter(x, y, s=50, c=color)

def plot_feature_objects_data(feature_objects, name_to_category, categories):
	objs = []
	labels = ['Others']
	plt.figure()
	colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(categories)))
	objs.append(scatter_feature_objects(plt, training_feature_objects+test_feature_objects, '#eeeeff'))

	i = 0
	for category in categories:
		filter_for_cat = lambda obj: name_to_category[obj.name] == category
		sub_fobjs = filter(filter_for_cat, training_feature_objects)
		objs.append(scatter_feature_objects(plt, sub_fobjs, colors[i]))
		labels.append(category)
		i += 1

	plt.legend(objs, labels)
	plt.show()

def plot_training_vs_test_data(training_feature_objects, test_feature_objects, name_to_category, categories):
	objs = []
	labels = ['Others']
	plt.figure()
	colors = plt.get_cmap('hsv')(np.linspace(0.5, 1.0, len(categories)*2))
	objs.append(scatter_feature_objects(plt, training_feature_objects+test_feature_objects, '#eeeeff'))

	i = 0
	for category in categories:
		filter_for_cat = lambda obj: name_to_category[obj.name] == category
		sub_training_fobjs = filter(filter_for_cat, training_feature_objects)
		sub_test_fobjs = filter(filter_for_cat, test_feature_objects)
		objs.append(scatter_feature_objects(plt, sub_training_fobjs, colors[i]))
		objs.append(scatter_feature_objects(plt, sub_test_fobjs, colors[i+1]))
		labels.extend([category+'_training', category+'_test'])
		i += 2

	plt.legend(objs, labels)
	plt.show()

def plot_data_with_indices(training_feature_objects, test_feature_objects, name_to_category, cat_list, indices, train_vs_test=False):
	categories = map(lambda index: cat_list[index], indices)
	if train_vs_test:
		plot_training_vs_test_data(training_feature_objects, test_feature_objects, name_to_category, categories)
	else:
		plot_feature_objects_data(training_feature_objects+test_feature_objects, name_to_category, categories)