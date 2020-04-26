import picture_objects
import random
import pickle
import numpy as np
import cv2

image_size = 320

# savefile = filepath to save a pickled version of the final dataset (e.g. 'file.p'); None = don't save
def create_pretrain_set(set_size, savefile = None):
    #list of tuples (picture, annotation)
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

        obj = picture_objects.combine_sb(shape, background)

        # if pickling the data, convert images to numpy, and switch from BGRA format to RGB format
        if savefile:
            obj[0] = cv2.cvtColor(np.array(obj[0]), cv2.COLOR_BGRA2RGB)

        data_set.append(obj)

    if savefile:
        file = open(savefile, 'wb')
        pickle.dump(data_set, file)
        file.close()

    else:
        return data_set


#pretrain = create_pretrain_set(1)
#for p in pretrain:
    #p[0].show()
    #print(p[1])
