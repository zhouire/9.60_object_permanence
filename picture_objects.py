#Requirements
#check on yolo resizing
#shapes should be between 10%-33% of the longer dimension of the image
#create some stationary iamges form the objects
#image creation with labels and bounding boxes
#create background, image(circles, squares, rectangles), and occlusions
#first stick to gray for background and black for occlusions


#yolo takes square images from 320 to 608, with a step of 32
from PIL import Image, ImageDraw, ImageFilter
import random
image_size = 320

#one function to generate the objects
def create_shape(percentage, shape_type, color = 'white', rotate = False):
    if shape_type == 'circle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.ellipse([(0, 0),(shape_size, shape_size)], fill=color)
        return shape, shape_type
    elif shape_type == 'square':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.rectangle([(0, 0),(shape_size, shape_size)], fill=color)
        if rotate:
            degree = random.randint(0,360)
            shape = shape.rotate(degree, expand=True)
        return shape, shape_type
    elif shape_type == 'triangle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.polygon([(int(shape_size/2), 0),(0, shape_size),(shape_size, shape_size)], fill=color)
        if rotate:
            degree = random.randint(0,360)
            shape = shape.rotate(degree, expand=True)
        return shape, shape_type
    else:
        raise Exception('not a valid shape type-choose between circle, rectangle, triangle')

#one function to generate the backgrounds
def create_background(color = 'gray'):
    background =  Image.new('RGBA',(image_size, image_size), color)
    return background
    
#one function to generate the occlusions
def create_occlusion(orientation, image_size, percentage, color = 'black'):
    if orientation=='horizontal':
        occ = Image.new('RGBA',(image_size, int(image_size*percentage)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(occ)
        draw.rectangle([(0, 0),(image_size, int(image_size*percentage))], fill=color)
        return occ
    
    elif orientation=='vertical':
        occ = Image.new('RGBA',(int(image_size*percentage), image_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(occ)
        draw.rectangle([(0, 0), (int(image_size*percentage), image_size)], fill=color)
        return occ
    else:
        raise Exception('not a valid shape type-choose between horizontal or vertical')


#one function to combine images together (for stationary objects)
# place shape on background in a random location
def combine_sb(shape, background, shape_loc = None, rotate_degree = 0):
    bs = background.size
    ss = shape[0].size

    print(bs, ss)

    if not shape_loc:
        rand_x = random.randint(0, bs[0]-ss[0]-1)
        rand_y = random.randint(0, bs[1]-ss[1]-1)
        loc = (rand_x,rand_y)
    else:
        loc = shape_loc

    background.paste(shape[0], loc, shape[0])
    height = ss[1]
    width = ss[0]
    top_left = loc
    bottom_right = (loc[0]+width, loc[1]+height)
    bounding_box = (top_left, bottom_right, width, height)
    annotation = (bounding_box, shape[1])
    #background.show()
    #draw = ImageDraw.Draw(background)
    #draw.rectangle([bounding_box[0], bounding_box[1]], fill=(255, 255, 255, 100))
    return background, annotation



#testing to see if stuff works
shape = create_shape(0.33, 'triangle', color='purple', rotate=False)
shape_size = shape[0].size
background = create_background()
test = combine_sb(shape, background)

occ = create_occlusion("vertical", image_size, 0.20)

for i in range(15):
    test_copy = test[0].copy()
    test2 = combine_sb((occ, 0), test_copy, ((image_size//14)*i - occ.size[0]//2, 0))
    test2[0].show()

# print(test2[1])
