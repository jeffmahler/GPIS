from PIL import Image
from pylab import *

lower_bound, upper_bound = 50, 255

def adjacent_black_pixels(i, j, pixels):
    num_black = 0
    for m in range(-1, 2):
        for n in range(-1, 2):
            if pixels[i+m][j+n] <= lower_bound:
                num_black += 1
    return num_black

#Returns a list list of the edges of the given image
def find_boundary(img_path):
    im = Image.open(img_path)
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
    arr = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(1, len(pixels) - 1):
        for j in range(1, len(pixels[0]) - 1):
            if (pixels[i][j] > lower_bound) and (pixels[i][j] <= upper_bound) and (adjacent_black_pixels(i, j, pixels) > 0):
                arr[i][j] = 255
    return arr

