import picture_objects
import random
import numpy as np

image_size = 320

def create_pretrain_set(set_size):
    #list of tuples (picture, bounding_box)
    data_set = []

    shapes = ['circle', 'square', 'triangle']
    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']

    for i in range(set_size):
        shape_pick = shapes[random.randint(0,2)]
        color_pick = colors[random.randint(0,6)]
        size = random.randint(10,33)/100
        shape = picture_objects.create_shape(size, shape_pick, color = color_pick, image_size = image_size)
        if i%10 in [0,1,2,3,4]:
            background = picture_objects.create_background_solid(image_size=image_size)
        elif i%10 in [5,6,7]:
            background = picture_objects.create_background_random(color_pick, noise_level  = 'less' ,color = 'gray', image_size = image_size)
        else:
            background = picture_objects.create_background_random(color_pick, noise_level  = 'more' ,color = 'gray', image_size = image_size)
        obj =  picture_objects.combine_sb(shape, background)
        data_set.append(obj)
    return data_set


# pretrain = create_pretrain_set(10)
# for p in pretrain:
#      p[0].show()
#      print(p[1])
