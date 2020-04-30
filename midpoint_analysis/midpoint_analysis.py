import numpy as np
import json
from iou import iou
import matplotlib.pyplot as plt

# read data from json file
yolo_json = open("results_valid_video_all.json", 'r')
yolo_output = yolo_json.readlines()
yolo_json.close()
yolo_output = json.loads(yolo_output[0])


# we want to get the accuracy of YOLO on different levels of occlusion

# reformat the json to give us a dictionary mapping the image_id to a list [bbox, category_id]
yolo_output_dict = {}
for i in yolo_output:
    imgpath = i['image_id'].replace('\\','/')

    result = None
    if i['bbox']:
        result = i['bbox'] + [i['category_id']]

    yolo_output_dict[imgpath] = result

# load in images.txt, labels.txt, and percent_occ.txt to make a list where each row is [image_path, label, percent_occ]
f_images = open("images.txt", 'r')
f_labels = open("labels.txt", 'r')
f_percent_occ = open("percentocc.txt", 'r')

images = f_images.readlines()
labels = f_labels.readlines()
percent_occ = f_percent_occ.readlines()

f_images.close()
f_labels.close()
f_percent_occ.close()

# reformat to make all of them just flat lists
images = [i[:-1].split(' ') for i in images]
labels = [json.loads(l) for l in labels]
percent_occ = [json.loads(p) for p in percent_occ]

# reshape into a big array nx2, [image, annotation]
# annotation is [x, y, w, h, label]
unified = []
for i in range(len(images)):
    for j in range(len(images[i])):
        # multiply all labels by image size 320 and move label to end of list
        annot = [320*k for k in labels[i][j][1:]] + labels[i][j][:1]
        unified.append([images[i][j], annot, percent_occ[i][j]])


# calculate iou between actual and predicted annotation, also get label correctness
# if there is NO predicted bounding box, IOU is automatically zero
# add an extra column dictating if the model predicted a bounding box at all
for row in unified:
    yolo_pred = yolo_output_dict[row[0]]
    if yolo_pred:
        cur_iou = iou(row[1], yolo_pred)
        row.append(cur_iou[0])
        row.append(cur_iou[1])
        row.append(True)
    else:
        row.append(0)
        row.append(False)
        row.append(False)


# CURRENT STATE OF UNIFIED
# [image path, actual annotation, percent_occlusion, iou, labels match, model predicted an object]
unified_T = np.array(unified).T

# make some graphs now
unified_percentocc = unified_T[2]
unified_iou = unified_T[3]
unified_labelmatch = unified_T[4]
unified_detected = unified_T[5]

# iou vs occlusion
#plt.scatter(unified_percentocc, unified_iou)
#plt.show()

standard_digitize = np.digitize(unified_percentocc, [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1])
human_digitize = np.digitize(unified_percentocc, [0, .18, .37, .65, .87, 1])
standard_bin = [0, 20, 40, 60, 80, 95, 100]
human_bin = [10, 26, 53, 83, 95, 100]

standard_avg_iou = []
human_avg_iou = []

standard_avg_detected = []

standard_avg_labelmatch = []
human_avg_labelmatch = []

for i in range(1, 8):
    # standard
    cur_idx = standard_digitize == i
    cur_iou = unified_iou[cur_idx]
    avg_iou = np.sum(cur_iou)/np.sum(cur_idx)
    standard_avg_iou.append(avg_iou)

    cur_detected = unified_detected[cur_idx]
    avg_detected = np.sum(cur_detected)/np.sum(cur_idx)
    standard_avg_detected.append(avg_detected)

    cur_labelmatch = unified_labelmatch[cur_idx]
    avg_labelmatch = np.sum(cur_labelmatch) / np.sum(cur_idx)
    standard_avg_labelmatch.append(avg_labelmatch)

for i in range(1, 7):
    # human
    cur_idx = human_digitize == i
    cur_iou = unified_iou[cur_idx]
    avg_iou = np.sum(cur_iou) / np.sum(cur_idx)
    human_avg_iou.append(avg_iou)

    cur_labelmatch = unified_labelmatch[cur_idx]
    avg_labelmatch = np.sum(cur_labelmatch) / np.sum(cur_idx)
    human_avg_labelmatch.append(avg_labelmatch)

plt.plot(standard_bin, standard_avg_iou, '.-')
plt.title('Average YOLO-v3 IOU with different amounts of occlusion')
plt.xlabel('% object occluded')
plt.ylabel('IOU')
plt.show()

plt.plot(standard_bin, standard_avg_detected, '.-')
plt.title('Average YOLO-v3 detection with different amounts of occlusion')
plt.xlabel('% object occluded')
plt.ylabel('proportion of objects detected by yolo')
plt.show()

plt.plot(standard_bin, standard_avg_labelmatch, '.-')
plt.title('Average YOLO-v3 label accuracy with different amounts of occlusion')
plt.xlabel('% object occluded')
plt.ylabel('label accuracy')
plt.show()

print('human bin iou ' + str(human_avg_iou))
print('human bin labelsmatch ' + str(human_avg_labelmatch))
print('standard bin iou ' + str(standard_avg_iou))
print('standard bin detection ' + str(standard_avg_detected))
print('standard bin labelsmatch ' + str(standard_avg_labelmatch))



