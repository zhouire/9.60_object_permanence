import numpy as np
import json
from iou import iou
import matplotlib.pyplot as plt
from bb_draw import bb_draw
from PIL import Image, ImageDraw, ImageFilter

# read data from json file
lstm_json = open("../data/lstm_testvideo_results.json", 'r')
lstm_output = lstm_json.readlines()
lstm_json.close()
lstm_output = json.loads(lstm_output[0])

# create a dictionary mapping imagepath to percent occlusion
percent_occ_file = open("percentocc.txt", 'r')
percent_occ = percent_occ_file.readlines()
percent_occ_file.close()
percent_occ = [json.loads(p) for p in percent_occ]

img_file = open("images.txt", 'r')
images = img_file.readlines()
img_file.close()
images = [i[:-1].split(' ') for i in images]

occlusion_dict = {}
for v in range(len(images)):
    video = images[v]
    occ = percent_occ[v]
    for i in range(len(video)):
        occlusion_dict[video[i]] = occ[i]

# convert all info into a list of images w/ stats
# imagepath, class correctness, iou, percent_occlusion, pred class, pred bbox, target class, target bbox
total_info = []
for i in lstm_output:
    video, class_pred, bbox_pred, class_target, bbox_target, ious = i['video'], i['class_pred'], i['bbox_pred'], \
                                                                    i['class_target'], i['bbox_target'], i['iou']
    for j in range(len(video)):
        class_correct = class_target[j] == class_pred[j]
        info = [video[j][0], class_correct, ious[j], occlusion_dict[video[j][0]], class_pred[j], bbox_pred[j], class_target[j], bbox_target[j]]
        total_info.append(info)

total_info_T = np.array(total_info).T

# make some graphs now
total_percentocc = total_info_T[3]
total_iou = total_info_T[2]
total_labelmatch = total_info_T[1]


standard_digitize = np.digitize(total_percentocc, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
human_digitize = np.digitize(total_percentocc, [0, .18, .37, .65, .87, 1])
standard_bin = [0, 20, 40, 60, 80, 95, 100]
human_bin = [10, 26, 53, 83, 95, 100]

standard_avg_iou = []
human_avg_iou = []

standard_avg_labelmatch = []
human_avg_labelmatch = []

for i in range(1, 8):
    # standard
    cur_idx = standard_digitize == i
    cur_iou = total_iou[cur_idx]
    avg_iou = np.sum(cur_iou)/np.sum(cur_idx)
    standard_avg_iou.append(avg_iou)

    cur_labelmatch = total_labelmatch[cur_idx]
    avg_labelmatch = np.sum(cur_labelmatch) / np.sum(cur_idx)
    standard_avg_labelmatch.append(avg_labelmatch)

for i in range(1, 7):
    # human
    cur_idx = human_digitize == i
    cur_iou = total_iou[cur_idx]
    avg_iou = np.sum(cur_iou) / np.sum(cur_idx)
    human_avg_iou.append(avg_iou)

    cur_labelmatch = total_labelmatch[cur_idx]
    avg_labelmatch = np.sum(cur_labelmatch) / np.sum(cur_idx)
    human_avg_labelmatch.append(avg_labelmatch)

plt.plot(standard_bin, standard_avg_iou, '.-')
plt.title('Average model IOU with different amounts of occlusion')
plt.xlabel('% object occluded')
plt.ylabel('IOU')
plt.show()

plt.plot(standard_bin, standard_avg_labelmatch, '.-')
plt.title('Average model label accuracy with different amounts of occlusion')
plt.xlabel('% object occluded')
plt.ylabel('label accuracy')
plt.show()

print('human bin iou ' + str(human_avg_iou))
print('human bin labelsmatch ' + str(human_avg_labelmatch))
print('standard bin iou ' + str(standard_avg_iou))
print('standard bin labelsmatch ' + str(standard_avg_labelmatch))



# show video with bounding box on each frame
# choose a video
#video_show = lstm_output[4]
video_show = lstm_output[0]
for i in range(len(video_show["video"])):
    imgpath = "../" + video_show["video"][i][0]
    img = Image.open(imgpath)
    bbox = np.array(video_show["bbox_pred"][i])*320

    path = "bboxvideo/video0_frame" + str(i) + ".jpg"

    p = bb_draw(img, bbox)
    p.save(path)
