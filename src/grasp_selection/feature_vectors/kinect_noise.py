from PIL import Image
import numpy as np
import scipy.ndimage.filters as filters
import scipy.stats as stat
from scipy.stats import multivariate_normal as m_normal
# import scipy.ndimage.filters.median_filter as median_filter
import math
import random
import copy

img_path = "/Users/MelRod/Desktop/elmers|depth_0_0_0.png"
min_distance = 0
max_distance = 0.5

img = Image.open(img_path)
pix = img.load()

width = img.size[0]
height = img.size[1]

sample_deviation = 1
sample_radius = 1
def sample_gaussians():
	global gaussians
	s = sample_deviation
	r = sample_radius
	gaussians = np.zeros((2*r + 1, 2*r + 1))
	for i in range(-r, r+1):
		for j in range(-r, r+1):
			gaussians[i+r, j+r] = m_normal.pdf([i, j], [0, 0], [s, s])
sample_gaussians()

def create_depths_from_pix(pix):
	depths = np.zeros((width, height))
	for x in range(0, width):
		for y in range(0, height):
			depths[x, y] = depth_at_point(x, y, pix)
	return depths

def depth_at_point(x, y, pix):
	pixel = pix[x, y]
	if pixel[3] == 0:
		return 0

	r = max_distance - min_distance
	ave = (pixel[0] + pixel[1] + pixel[2]) / 3.0

	return r * ave/255.0 + min_distance

def depth_to_L(z):
	return ((z - min_distance)/(max_distance - min_distance))*255

def depths_to_image(depths):
	img = Image.new("L", (width, height))
	pix = img.load()
	for x in range(0, width):
		for y in range(0, height):
			pix[x, y] = depth_to_L(depths[x, y])
	return img

def in_image(x, y):
	return x >= 0 and x < width and y >= 0 and y < height

def translate_point(x, y, z):
	# TODO change values
	f_x = 933.3352;
	f_y = 787.501575;
	t_x = 0.05;

	# p = np.matrix([x, y, 1]).reshape(3,1)
	# K = np.matrix([
	# 	[f_x, 0, width/2],
	# 	[0, f_y, height/2],
	# 	[0, 0, 1]])
	
	# t = np.matrix([t_x/z, 0, 0]).reshape(3,1)
	# new_p = np.add(p, np.dot(K, t))
	# new_x = int(round(p[0][0]))
	# new_y = int(round(p[1][0]))
	# return new_x, new_y

	return x + f_x*t_x/z, y


def shift_depths(depths):
	new_depths = np.zeros((width, height))
	for x in range(0, width):
		for y in range(0, height):
			z = depths[x, y]
			new_x, new_y = translate_point(x, y, z)
			
			if in_image(new_x, new_y):
				z_at = new_depths[new_x, new_y]
				if z_at == 0 or z_at > z:
					new_depths[new_x, new_y] = z
	return new_depths

def dilate(depths):
	new_depths = np.zeros((width, height))
	for x in range(0, width):
		for y in range(0, height):
			if depths[x, y] != 0:
				new_depths[x, y] = depths[x, y]
			elif are_any_neighbors_full(x, y, depths):
				new_depths[x, y] = gaussian_sample_neighbors(x, y, depths)
	return new_depths

def are_any_neighbors_full(x, y, depths):
	r = sample_radius
	for i in range(x - r, x + r + 1):
		for j in range(y - r, y + r + 1):
			if i == x and j == y:
				continue
			if in_image(i, j) and depths[i, j] != 0:
				return True
	return False

def gaussian_sample_neighbors(x, y, depths):
	r = sample_radius

	non_norm_depth = 0
	norm = 0
	for i in range(-r, r+1):
		for j in range(-r, r+1):
			if (i == 0 and j == 0) or not in_image(x+i, y+j):
				continue
			z = depths[x+i, y+j]
			if z == 0:
				continue
			p = gaussians[i, j]
			non_norm_depth += p * z
			norm += p
	return non_norm_depth/norm

def erode(depths):
	new_depths = np.zeros((width, height))
	for x in range(0, width):
		for y in range(0, height):
			if depths[x, y] == 0 or are_any_neighbors_empty(x, y, depths):
				new_depths[x, y] = 0
			else:
				new_depths[x, y] = depths[x, y]
	return new_depths

def are_any_neighbors_empty(x, y, depths):
	r = 2
	for i in range(-r, r+1):
		for j in range(-r, r+1):
			if i == x and j == y:
				continue
			if in_image(x+i, y+j) and depths[x+i, y+j] == 0:
				return True
	return False

def median(depths):
	new_depths = np.zeros((width, height))
	for x in range(0, width):
		for y in range(0, height):
			new_depths[x, y] = median_neighbors(x, y, depths)
	return new_depths

def median_neighbors(x, y, depths):
	r = 2
	arr = []
	for i in range(-r, r+1):
		for j in range(-r, r+1):
			if in_image(x+i, y+j):
				arr.append(depths[x+i, y+j])
	return np.median(arr)

def noisy_depth_at(x, y, depths):
	z = depths[x, y]
	if z == 0:
		return 0

	normal = np.cross([1, 0, depths[x+1, y] - z], [0, 1, depths[x, y+1] - z])
	np.linalg.norm(normal)
	theta = math.acos(math.fabs(normal[2]))

	sigma_L_px = 0.8 + 0.035 * theta / (math.pi/2 - theta)
	sigma_z = 0.0012 + 0.0019 * (z - 0.4)**2 + 0.0001/math.sqrt(z) * (theta/(math.pi/2 - theta))**2

	new_x = random.gauss(x, sigma_L_px)
	new_y = random.gauss(y, sigma_L_px)

	new_x, new_y = clamp_coord(new_x, new_y)
	return random.gauss(depths[new_x, new_y], sigma_z)

def add_noise(depths):
	new_depths = np.zeros((width, height))
	for x in range(0, width - 1):
		for y in range(0, height - 1):
			new_depths[x, y] = noisy_depth_at(x, y, depths)
	return new_depths

def clamp_coord(x, y):
	if x < 0:
		x = 0
	elif x >= width:
		x = width - 1

	if y < 0:
		y = 0
	elif y >= height:
		y = height - 1

	return x, y

print "CREATING..."
depths = create_depths_from_pix(pix)
print "CREATED"
print "SHIFTING..."
depths = shift_depths(depths)
print "SHIFTED"
# print "DILATING..."
# depths = dilate(depths)
# print "DILATED"
# print "ERODING..."
# depths = erode(depths)
# print "ERODED"
print "MEDIANING..."
depths = filters.median_filter(depths, size=(13, 5))
print "MEDIANED"
print "ADDING NOISE..."
depths = add_noise(depths)
print "DONE"
new_img = depths_to_image(depths)
new_img.show()

