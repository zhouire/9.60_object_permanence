import picture_objects
import random
import numpy as np

# output combined image with shape, background, occlusion
# shape_info = (shape object, shape location, shape name)
# occlusion_info = (occlusion, location)
def combine_image(shape_info, occlusion_info, background):
    ss = shape_info[0].size
    shape_loc = shape_info[1]
    occ_loc = occlusion_info[1]

    print(shape_loc, occ_loc)

    background.paste(shape_info[0], shape_loc, shape_info[0])
    background.paste(occlusion_info[0], occ_loc, occlusion_info[0])

    height = ss[1]
    width = ss[0]
    top_left = shape_loc
    bottom_right = (shape_loc[0] + width, shape_loc[1] + height)
    bounding_box = (top_left, bottom_right, width, height)

    # annotate with bounding box, shape name
    annotation = (bounding_box, shape_info[2])

    return background, annotation


# given objects and movement information, compiles into a "video" (list of image objects)
# params:
#   background = Image object representing background
#   shape_info = (Image object representing shape, shape name)
#   occlusion = Image object representing occlusion
#   background = Image object representing background
#   shape_move = ((start_x, start_y), (end_x, end_y)) or single location (x,y) if not moving
#   occlusion_move = one of ['up', 'down', 'left', 'right'] or single location (x,y) if not moving
#   num_frames = number of frames
def compile_video(shape_info, occlusion, background, shape_move, occlusion_move, image_size, num_frames):
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
                                   background_copy)

        video.append(img)
        annots.append(annot)

    return np.array([video, annots])



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

def generate_dataset(num_videos, shape_size_range, occlusion_size_range, full_occlusions, num_frames,
                     image_size = 320,
                     shapes = ("circle", "square", "triangle"),
                     occlusions = ("horizontal", "vertical"),
                     movements = ("shape", "occlusion"),
                     mov_dir = 2):

    colors = ['white', 'red', 'orange', 'yellow', 'green', 'blue', 'purple']

    #TODO: IMPLEMENT SOME SORT OF SAVE FUNCTION INSTEAD OF JUST RETURNING A LIST OF OBJECTS
    dataset = []

    for i in num_videos:
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
        background = picture_objects.create_background()

        mcoord = image_size - shape_size - 1

        if moving_obj == "occlusion":
            if occlusion_type == 'vertical':
                occlusion_move = 'right'
                if mov_dir == 2:
                    occlusion_move = random.choice(['right', 'left'])

                # to ensure a full occlusion if it is possible, place shape fully obscured by one of the occlusion positions
                # occlusion will sweep from one side to another, starting and ending fully off the image
                shape_loc = [0, random.uniform(0, mcoord)]
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

                shape_loc = [random.uniform(0, mcoord), 0]
                if occlusion_size > shape_size:
                    shape_loc[1] = int((image_size//(num_frames-1))*random.randint(num_frames//2, num_frames-2) -
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
        video = compile_video((shape, shape_type), occlusion, background, shape_loc, occlusion_move, image_size, num_frames)

        dataset.append(video)

    return dataset





# testing code
image_size = 320
num_frames = 15
shape = picture_objects.create_shape(0.25, 'triangle', color='purple', rotate=False)
occlusion = picture_objects.create_occlusion("vertical", image_size, 0.20)
background = picture_objects.create_background()

video = compile_video(shape, occlusion, background, ((20, 100), (300, 200)), (160, 0), image_size, num_frames)

for i in video[0]:
    i.show()
