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
def create_shape(percentage, shape_type, color = 'white', rotate = False, image_size = image_size):
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
        raise Exception('not a valid shape type-choose between circle, square, triangle')

#one function to generate the backgrounds
def create_background_solid(color = 'gray', image_size = image_size):
    background =  Image.new('RGBA',(image_size, image_size), color)
    return background

#function to make background with other random objects to improve training
def create_background_random(image_color, noise_level  = None ,color = 'gray', image_size = image_size):
    #pass in image color to avoid overspill with other objects
    if noise_level == None:
        background= create_background_solid(color = 'gray', image_size = image_size)
        return background
    elif noise_level == 'less':
        num_items = random.randint(10,20)
    else:
        num_items = random.randint(25,35)
    size_range = (100/num_items)/2
    background = create_background_solid(color)
    shapes = ['circle', 'square', 'triangle', 'occlusion']
    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple', 'black']
    colors.remove(image_color)
    # if num_items < 16:
    #     num_rows = 3
    # else:
    #      num_rows = 4
    for n in range(num_items):
        size = random.uniform(0,size_range)/100
        shape_pick = shapes[random.randint(0,3)]
        color_pick = colors[random.randint(0,6)]
        if shape_pick != 'occlusion':
            shape = create_shape(size, shape_pick, color = color_pick, image_size = image_size)[0]
        else:
            rand = random.randint(0,1)
            if rand == 0:
                shape = create_occlusion('vertical', image_size, size, color = color_pick)
            else:
                perc = random.uniform(0, 0.33)
                shape = create_occlusion('vertical', image_size, perc, color = color_pick)
        degree = random.randint(0,360)
        shape = shape.rotate(degree, expand=True)
        #ss = shape.size
        # print(ss)
        # print(image_size-int(ss[0])-1)
        # print(image_size-int(ss[1])-1)
        rand_x = random.randint(0, image_size)
        rand_y = random.randint(0, image_size)
        loc = (rand_x,rand_y)
        background.paste(shape, loc, shape)
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
def combine_sb(shape, background, shape_loc = None):
    bs = background.size
    ss = shape[0].size

    #print(bs, ss)

    if not shape_loc:
        rand_x = random.randint(0, bs[0]-ss[0]-1)
        rand_y = random.randint(0, bs[1]-ss[1]-1)
        loc = (rand_x,rand_y)
    else:
        loc = shape_loc

    background.paste(shape[0], loc, shape[0])
    height = ss[1]
    width = ss[0]
    #top_left = loc
    #bottom_right = (loc[0]+width, loc[1]+height)
    center_x = loc[0]+width//2
    center_y = loc[1]+height//2
    bounding_box = (center_x, center_y, width, height)
    if shape[1] == 'circle':
        one_hot = [1,0,0]
    elif shape[1] == 'square':
        one_hot = [0,1,0]
    elif shape[1] == 'triangle':
        one_hot = [0,0,1]
    annotation = (bounding_box, one_hot)
    #background.show()
    #draw = ImageDraw.Draw(background)
    #draw.rectangle([bounding_box[0], bounding_box[1]], fill=(255, 255, 255, 100))
    return [background, annotation]



#testing to see if stuff works
# shape = create_shape(0.33, 'triangle', color='purple', rotate=False)
# # shape_size = shape[0].size
# background = create_background_random('purple', noise_level = 'more',color = 'gray', image_size = image_size)
# background.save('test.png')
# background.show()
# test = combine_sb(shape, background)

# occ = create_occlusion("vertical", image_size, 0.20)

# for i in range(15):
#     test_copy = test[0].copy()
#     test2 = combine_sb((occ, 0), test_copy, ((image_size//14)*i - occ.size[0]//2, 0))
#     test2[0].show()

# print(test2[1])
