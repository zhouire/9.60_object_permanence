#Requirements
#check on yolo resizing
#shapes should be between 10%-33% of the longer dimension of the image
#create some stationary iamges form the objects
#image creation with labels and bounding boxes
#create background, image(circles, squares, rectangles), and occlusions
#first stick to gray for background and black for occlusions


#yolo takes square images from 320 to 608, with a step of 32
from PIL import Image, ImageDraw, ImageFilter
image_size = 320

#one function to generate the objects
def create_shape(percentage, shape, color = 'white'):
    if shape == 'circle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.ellipse([(0, 0),(shape_size, shape_size)], fill=color)
        return shape
    elif shape == 'rectangle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(int(shape_size/3), shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.rectangle([(0, 0),(int(shape_size/3), shape_size)], fill=color)
        return shape
    elif shape == 'triangle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.polygon([(int(shape_size/2), 0),(0, shape_size),(shape_size, shape_size)], fill=color)
        return shape
    else:
        raise Exception('not a valid shape type-choose between circle, rectangle, triangle')

#one function to generate the backgrounds
def create_background(color = 'gray'):
    background =  Image.new('RGBA',(image_size, image_size), color)
    return background
    
#one function to generate the occlusions
def create_occlusion(orientation, shape_size, color = 'black'):
    if orientation=='horizontal':
        occ = Image.new('RGBA',(shape_size, int(shape_size/3), ), (0, 0, 0, 0))
        draw = ImageDraw.Draw(occ)
        draw.rectangle([(0, 0),(shape_size, int(shape_size/3))], fill=color)
        return occ
    
    elif  orientation=='vertical':
        occ = Image.new('RGBA',(int(shape_size/3), shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(occ)
        draw.rectangle([(int(shape_size/3), shape_size), (0, 0),], fill=color)
        return occ
    else:
        raise Exception('not a valid shape type-choose between horizontal or vertical')


#one function to combine images together (for stationary objects)
def combine_sb(shape, background, rotate_degree = 0):
    pass

#another function to add in labels and bounding boxes
def labels(image):
    pass

shape = create_shape(0.33, 'triangle', color='purple')
shape_size = shape.size
test = create_occlusion('vertical', max(shape_size), color = 'black')
test.show()