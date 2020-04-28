import picture_objects
import random
import pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

image_size = 320

# picklefile = filepath to save a pickled version of the final dataset (e.g. 'file.p'); None = don't save
# savepath = directory to save the images making up the dataset
# imagefile = location to save list of filenames
# labelfile = location to save list of corresponding labels
def create_pretrain_set(set_size, picklefile = None, savepath = None, imagefile = None, labelfile = None):
    #list of tuples (picture, annotation)
    data_set = []

    shapes = ['circle', 'square', 'triangle']
    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']

    if imagefile and labelfile:
        image_file = open(imagefile, 'a')
        label_file = open(labelfile, 'a')

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

        if savepath and imagefile and labelfile:
            img_path = savepath + 'pretrainimg_' + str(i) + ".jpg"

            # cnovert image from RGBA to RGB
            img = obj[0].copy()
            rgb_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            rgb_img.save(img_path)

            image_file.write(img_path + '\n')

            # label index, x, y, w, h ([0,1] scale)
            label = str([obj[1][1]] + obj[1][0])
            label_file.write(label + '\n')

        # if pickling the data, convert images to numpy, and switch from BGRA format to RGB format
        if picklefile:
            obj[0] = cv2.cvtColor(np.array(obj[0]), cv2.COLOR_BGRA2RGB)

        data_set.append(obj)

    if imagefile and labelfile:
        image_file.close()
        label_file.close()

    if picklefile:
        file = open(picklefile, 'wb')
        pickle.dump(data_set, file)
        file.close()

    else:
        return data_set


pretrain = create_pretrain_set(10000,
                               savepath="data/pretrain/images/",
                               imagefile="data/pretrain/images.txt",
                               labelfile="data/pretrain/labels.txt")


#for p in pretrain:
    #p[0].show()
    #print(p[1])
