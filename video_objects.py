import picture_objects
import random
import numpy as np
import cv2
from PIL import Image

image_size = 320

# output combined image with shape, background, occlusion
# shape_info = (shape object, shape location, shape name)
# occlusion_info = (occlusion, location)
# return: (image, [label idx, x, y, w, h, percent occlusion])
def combine_image(shape_info, occlusion_info, background, percent_occ = False):
    ss = shape_info[0].size
    shape_loc = shape_info[1]
    occ_loc = occlusion_info[1]

    background.paste(shape_info[0], shape_loc, shape_info[0])
    background.paste(occlusion_info[0], occ_loc, occlusion_info[0])

    height = ss[1]
    width = ss[0]
    #top_left = shape_loc
    #bottom_right = (shape_loc[0] + width, shape_loc[1] + height)
    center_x = shape_loc[0]+width//2
    center_y = shape_loc[1]+height//2
    # scale to [0,1]
    bounding_box = [center_x/image_size, center_y/image_size, width/image_size, height/image_size]

    # annotate with bounding box, shape name
    if shape_info[2] == 'circle':
        #one_hot = [1,0,0]
        one_hot_idx = 0
    elif shape_info[2] == 'square':
        #one_hot = [0,1,0]
        one_hot_idx = 1
    elif shape_info[2] == 'triangle':
        #one_hot = [0,0,1]
        one_hot_idx = 2
    annotation = [one_hot_idx] + bounding_box
    if percent_occ:
        po = calc_percent_occ(shape_info, occlusion_info)
        annotation = annotation + po
    # print(annotation)
    # background.show()
    return background, annotation

def calc_percent_occ(shape_info, occlusion_info):
    ss = shape_info[0].size
    shape_width = ss[0]
    shape_height = ss[1]
    shape_loc = shape_info[1]
    shape_tl = (shape_loc[0], shape_loc[1])
    shape_br = (shape_loc[0]+shape_width, shape_loc[1]+shape_height)

    os = occlusion_info[0].size
    occ_loc = occlusion_info[1]
    occ_height = os[1]
    occ_width = os[0]
    occ_tl = (occ_loc[0], occ_loc[1])
    occ_br = (occ_loc[0]+occ_width, occ_loc[1]+occ_height)
    overlap_area = (min(shape_br[0], occ_br[0])-max(shape_tl[0], occ_tl[0]))*(min(shape_br[1], occ_br[1])-max(shape_tl[1], occ_tl[1]))
    
    shape_area = shape_height*shape_width

    return max(overlap_area/shape_area, 0)

# given objects and movement information, compiles into a "video" (list of image objects)
# params:
#   background = Image object representing background
#   shape_info = (Image object representing shape, shape name)
#   occlusion = Image object representing occlusion
#   background = Image object representing background
#   shape_move = ((start_x, start_y), (end_x, end_y)) or single location (x,y) if not moving
#   occlusion_move = one of ['up', 'down', 'left', 'right'] or single location (x,y) if not moving
#   num_frames = number of frames
def compile_video(shape_info, occlusion, background, shape_move, occlusion_move, image_size, num_frames, percent_occ = False):
    # create list of locations if the occlusion is moving
    if type(occlusion_move) == str:
        shape_locs = [shape_move for f in range(num_frames)]

        if occlusion_move == 'up' or occlusion_move == 'down':
            occ_locs = [(0, (image_size//(num_frames-1))*f - occlusion.size[1]//2) for f in range(num_frames)]
            if occlusion_move == 'up':
                occ_locs.reverse()
        else:
            occ_locs = [((image_size//(num_frames-1))*f - occlusion.size[0]//2, 0) for f in range(num_frames)]
            if occlusion_move == 'left':
                occ_locs.reverse()

    # if instead, the shape is moving
    else:
        occ_locs = [occlusion_move for f in range(num_frames)]

        [start_x, start_y], [end_x, end_y] = shape_move
        shape_locs = [((end_x - start_x)//(num_frames-1)*f + start_x, (end_y - start_y)//(num_frames-1)*f + start_y)
                      for f in range(num_frames)]

    # list of images making up the "video":
    video = []
    # list of annotations
    annots = []

    # loop through all frames
    for i in range(num_frames):
        # retain original background after pasting
        background_copy = background.copy()

        img, annot = combine_image((shape_info[0], shape_locs[i], shape_info[1]),
                                   (occlusion, occ_locs[i]),
                                   background_copy, percent_occ)

        video.append(img)
        annots.append(annot)

    return [video, annots]


# generates a full dataset of videos with uniform distribution of different shapes, occlusions, and movement patterns
# params:
#   num_videos = number of videos
#   shape_size_range = tuple range of shape sizes (smallest, largest) by percentage of image size
#   occlusion_size_range = tuple range of occlusion sizes (smallest, largest) by percent of image size
#   full_occlusions = proportion of videos containing some full occlusions
#   num_frames = number of frames in video
#   shapes = [tuple of shape names]
#   occlusions = [tuple of occlusion types]
#   movements = [tuple of movement types]
#   mov_dir = #directions of movement (1 for up or right, 2 for up/down, left/right)
#   savepath = general directory to save videos in (as collections of images)

def generate_dataset(num_videos, shape_size_range, occlusion_size_range, full_occlusions, num_frames,
                     image_size = 320,
                     shapes = ("circle", "square", "triangle"),
                     occlusions = ("horizontal", "vertical"),
                     movements = ("shape", "occlusion"),
                     mov_dir = 2,
                     percent_occ = False,
                     savepath = None):

    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']

    dataset = []

    # meta-txt containing a list of txt files, each of which contains info for one video
    if savepath:
        file_allimg = open(savepath + "images.txt", 'w')
        file_alllabel = open(savepath + "labels.txt", 'w')

    for i in range(num_videos):
        shape_type = random.choice(shapes)
        occlusion_type = random.choice(occlusions)
        moving_obj = random.choice(movements)
        color = random.choice(colors)

        shape_size = random.uniform(shape_size_range[0], shape_size_range[1])
        # if we want full occlusions in the video, make occlusion larger than shape
        if random.random() < full_occlusions:
            occlusion_size = random.uniform(shape_size, occlusion_size_range[1])
        else:
            occlusion_size = random.uniform(occlusion_size_range[0], shape_size)

        shape = picture_objects.create_shape(shape_size, shape_type, color)
        occlusion = picture_objects.create_occlusion(occlusion_type, image_size, occlusion_size)
        background = picture_objects.create_background_solid()

        shape_size = int(image_size*shape_size)
        occlusion_size = int(image_size*occlusion_size)

        mcoord = image_size - shape_size - 1

        if moving_obj == "occlusion":
            if occlusion_type == 'vertical':
                occlusion_move = 'right'
                if mov_dir == 2:
                    occlusion_move = random.choice(['right', 'left'])

                # to ensure a full occlusion if it is possible, place shape fully obscured by one of the occlusion positions
                # occlusion will sweep from one side to another, starting and ending fully off the image
                shape_loc = [0, random.randint(0, mcoord)]
                if occlusion_size > shape_size:
                    shape_loc[0] = int((image_size//(num_frames-1))*random.randint(num_frames//2, num_frames-2) -
                                       occlusion_size//2 +
                                       random.randint(0, occlusion_size - shape_size))
                else:
                    if occlusion_move == 'right':
                        shape_loc[0] = random.randint(mcoord//2, mcoord)
                    else:
                        shape_loc[0] = random.randint(0, mcoord//2)

            else:
                occlusion_move = 'up'
                if mov_dir == 2:
                    occlusion_move = random.choice(['up', 'down'])

                shape_loc = [random.randint(0, mcoord), 0]
                if occlusion_size > shape_size:
                    if occlusion_move == 'down':
                        sf = random.randint(num_frames//2, num_frames-2)
                    else:
                        sf = random.randint(1, num_frames//2)
                    shape_loc[1] = int((image_size//(num_frames-1))*sf -
                                       occlusion_size//2 +
                                       random.randint(0, occlusion_size - shape_size))
                else:
                    if occlusion_move == 'up':
                        shape_loc[1] = random.randint(0, mcoord//2)
                    else:
                        shape_loc[1] = random.randint(mcoord//2, mcoord)

        # case: if the shape is moving, not the occlusion
        else:
            if occlusion_type == 'vertical':
                # shape vertical location can be anywhere
                # shape horizontal placement should start in the left 1/4 of the image and end in the right 1/4.
                shape_loc = [[random.randint(0, mcoord//4), random.randint(0, mcoord)],
                             [random.randint(3*mcoord//4, mcoord), random.randint(0, mcoord)]]

                # if two directions of movement allowed, shuffle the start and end lists
                if mov_dir == 2:
                    random.shuffle(shape_loc)

                # If a full occlusion is necessary, place the occlusion over one of the possible locations of the shape,
                # allowing full exposure of the shape for a few frames before the occlusion
                occlusion_move = [0, 0]
                if occlusion_size > shape_size:
                    occlusion_move[0] = int(((shape_loc[0][0] - shape_loc[1][0])//(num_frames-1)) *
                                            random.randint(num_frames//2, num_frames - 2) -
                                            random.randint(0, occlusion_size - shape_size))

                else:
                    if shape_loc[0][0] < shape_loc[1][0]:
                        occlusion_move[0] = random.randint(mcoord//2, mcoord)
                    else:
                        occlusion_move[0] = random.randint(0, mcoord//2)

            # case: horizontal occlusion
            else:
                shape_loc = [[random.randint(0, mcoord), random.randint(3*mcoord//4, mcoord)],
                            [random.randint(0, mcoord), random.randint(0, mcoord//4)]]

                if mov_dir == 2:
                    random.shuffle(shape_loc)

                # If a full occlusion is necessary, place the occlusion over one of the possible locations of the shape,
                # allowing full exposure of the shape for a few frames before the occlusion
                occlusion_move = [0, 0]
                if occlusion_size > shape_size:
                    occlusion_move[1] = int(((shape_loc[0][1] - shape_loc[1][1]) // (num_frames - 1)) *
                                            random.randint(num_frames // 2, num_frames - 2) -
                                            random.randint(0, occlusion_size - shape_size))

                else:
                    if shape_loc[0][1] < shape_loc[1][1]:
                        occlusion_move[1] = random.randint(mcoord // 2, mcoord)
                    else:
                        occlusion_move[1] = random.randint(0, mcoord // 2)

        # using info, create the "video" and add to list of videos
        # video should be a np 2xn matrix of the "video" and annotations
        video = compile_video(shape, occlusion, background, shape_loc, occlusion_move, image_size, num_frames, percent_occ)

        # save each video as a collection of images, with a frames_n.txt file containing the frames in order, and
        # a labels_n.txt file containing the labels for each frame in order
        # compile meta-lists images.txt and labels.txt containing a list of the txt files defining each video
        if savepath:
            frames, annots = video

            image_file_path = savepath + "frames_" + str(i) + ".txt"
            label_file_path = savepath + "labels_" + str(i) + ".txt"
            image_file = open(image_file_path, 'w')
            label_file = open(label_file_path, 'w')

            for f in range(len(frames)):
                img_path = savepath + "images/video" + str(i) + "_frame" + str(f) + ".jpg"
                label_path = savepath + "labels/video" + str(i) + "_frame" + str(f) + ".txt"

                # convert image from RGBA to RGB
                img = frames[f].copy()
                rgb_img = Image.new("RGB", (image_size, image_size), (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])  # 3 is the alpha channel
                rgb_img.save(img_path)

                image_file.write(img_path + '\n')

                label = annots[f]
                label_file.write(str(label) + '\n')

                # write each label to its own txt file
                label_str = ' '.join(map(str, label))
                newlabel = open(label_path, 'w')
                newlabel.write(label_str)
                newlabel.close()

            image_file.close()
            label_file.close()

            file_allimg.write(image_file_path + '\n')
            file_alllabel.write(label_file_path + '\n')

        dataset.append(video)

    if savepath:
        file_allimg.close()
        file_alllabel.close()

    return dataset


# convert list of images to mp4 video format (for human testing)
def video_format(images, fr, videoname):
    dimensions = np.array(images[0]).shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video = cv2.VideoWriter(videoname, fourcc, fr, dimensions)

    # Appending the images to the video one by one
    for image in images:
        image_RGB = cv2.cvtColor(np.array(image), cv2.COLOR_BGRA2RGB)
        video.write(np.array(image_RGB))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


'''
# testing code
image_size = 320
num_frames = 15


# manual entry shape and occlusion
shape1 = picture_objects.create_shape(0.25, 'triangle', color='purple', rotate=False)
occlusion1 = picture_objects.create_occlusion("vertical", image_size, 0.20)
background1 = picture_objects.create_background_solid()

shape2 = picture_objects.create_shape(0.25, 'triangle', color='purple', rotate=False)
occlusion2 = picture_objects.create_occlusion("horizontal", image_size, 0.20)
background2 = picture_objects.create_background_solid()

# video = compile_video(shape1, occlusion1, background1, ((20, 100), (300, 200)), (160, 0), image_size, num_frames, percent_occ = False)
# video = compile_video(shape2, occlusion2, background2, ((300, 200), (20, 50)), (0, 150), image_size, num_frames, percent_occ = True)
video = compile_video(shape1, occlusion1, background1, (100, 200), 'right', image_size, num_frames)

video_format(video[0], 5, "test.avi")

for i in video[0]:
    i.show()


'''
# testing randomized dataset generation
data = generate_dataset(4, (0.1, 0.25), (0.15, 0.33), 0.5, 10, savepath = "data/videos/")

'''
for d in data:
    for i in d[0]:
        i.show()
'''