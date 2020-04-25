image_size = 320
num_frames = 15
import picture_objects
import video_objects
import random

colors = ['red', 'blue', 'yellow', 'orange', 'purple', 'white', 'green', 'pink']
shapes = ['circle', 'square', 'triangle']

background = picture_objects.create_background_solid()
occlusion_h = picture_objects.create_occlusion("horizontal", image_size, 0.20)
occlusion_v = picture_objects.create_occlusion("vertical", image_size, 0.20)
image_size = 320
num_frames = 15
#make videos where the object moves
for i in range(8):
    c = random.randint(0,7)
    color = colors[c]  
    if i%3 == 0:
        s = 'circle'
    if i%3 == 1:
        s  = 'square'
    if i%3 == 2:
        s = 'triangle'
    p = random.randint(10,33)/100
    o = random.randint(0,100)%2
    shape = picture_objects.create_shape(p, s, color=color, rotate=False)
    if o == 0: 
        occlusion = occlusion_h 
    else:
        occlusion_v
    sm = random.randint(0,1)
    if sm == 0:
        video = video_objects.compile_video(shape, occlusion_v, background, ((20, 100), (300, 200)), (160, 0), image_size, num_frames)
    else:
        video = video_objects.compile_video(shape, occlusion_h, background, ((300, 200), (20, 50)), (0, 150), image_size, num_frames)
    ist = str(i)
    name = 'HDS'+ist+'.avi'
    video_objects.video_format(video[0], 5, name)


# make videos where the occlusion moves
for i in range(8,24):
    c = random.randint(0,7)
    color = colors[c]  
    if i%3 == 0:
        s = 'circle'
    if i%3 == 1:
        s  = 'square'
    if i%3 == 2:
        s = 'triangle'
    p = random.randint(10,33)/100
    o = random.randint(0,100)%2
    shape = picture_objects.create_shape(p, s, color=color, rotate=False)
    if o == 0: 
        occlusion = occlusion_h 
    else:
        occlusion_v
    sm = random.randint(0,1)
    shape_loc = (random.randint(50,250), random.randint(50,250))
    if o == 1:
        if sm == 0:
            video = video_objects.compile_video(shape, occlusion_v, background, shape_loc, 'right', image_size, num_frames)
        elif sm == 1:
            video = video_objects.compile_video(shape, occlusion_v, background, shape_loc, 'left', image_size, num_frames)
    if o == 0:
        if sm == 0:
            video = video_objects.compile_video(shape, occlusion_h, background, shape_loc, 'up', image_size, num_frames)
        elif sm == 1:
            video = video_objects.compile_video(shape, occlusion_h, background, shape_loc, 'down', image_size, num_frames)
    ist = str(i)
    name = 'HDS'+ist+'.avi'
    video_objects.video_format(video[0], 5, name)



    