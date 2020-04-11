#Requirements
#check on yolo resizing
#shapes should be between 10%-33% of the longer dimension of the image
#create some stationary iamges form the objects
#image creation with labels and bounding boxes
#create background, image(circles, squares, rectangles), and occlusions
#first stick to gray for background and black for occlusions

#one function to generate the objects

#yolo takes square images from 320 to 608, with a step of 32
from PIL import Image, ImageDraw, ImageFilter
image_size = 320
def create_shape(percentage, shape):
    shape_size = int(image_size*percentage)


#one function to generate the backgrounds
def create_background(color = 'gray'):
    pass

#one function to generate the occlusions
def create_occlusion(orientation, shape_size, color = 'black'):
    pass

#one function to combine images together (for stationary objects)
def combine_sb(shape, background, rotate_degree = 0):
    pass

#another function to add in labels and bounding boxes
def labels(image):
    pass