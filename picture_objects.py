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
def create_shape(percentage, shape_type, color = 'white'):
    if shape_type == 'circle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.ellipse([(0, 0),(shape_size, shape_size)], fill=color)
        return shape, shape_type
    elif shape_type == 'rectangle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(int(shape_size/3), shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.rectangle([(0, 0),(int(shape_size/3), shape_size)], fill=color)
        return shape, shape_type
    elif shape_type == 'triangle':
        shape_size = int(image_size*percentage)
        shape = Image.new('RGBA',(shape_size, shape_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape)
        draw.polygon([(int(shape_size/2), 0),(0, shape_size),(shape_size, shape_size)], fill=color)
        return shape, shape_type
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
        draw.rectangle([(int(shape_size/3), shape_size), (0, 0)], fill=color)
        return occ
    else:
        raise Exception('not a valid shape type-choose between horizontal or vertical')


#one function to combine images together (for stationary objects)
def combine_sb(shape, background, rotate_degree = 0):
    bs = background.size
    ss = shape[0].size
    loc = (int(bs[0]/2)-int(ss[0]/2),int(bs[1]/2)-int(ss[1]/2))
    background.paste(shape[0], loc, shape[0])
    height = ss[1]
    width = ss[0]
    top_left = loc
    bottom_right = (loc[0]+width, loc[1]+height)
    bounding_box = (top_left, bottom_right, width, height)
    annotation = (bounding_box, shape[1])
    return background, annotation


#testing to see if stuff works
shape = create_shape(0.33, 'rectangle', color='purple')
shape_size = shape[0].size
background = create_background()
test = combine_sb(shape, background)
test[0].show()
print(test[1])
