import os
import torch
import pandas as pd
import numpy as np
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

image_size = 320

# this dataset consists of the features that come out of CNN and YOLO, with output labels [one-hot values, x, y, w, h]
# shape: (seq_len, batch, input_size)
class FeaturesDataset(Dataset):

    # input the json files (paths) form CNN and YOLO, and the image file that dictates the ordering of the images
    # for the videos, and the corresponding labels file
    def __init__(self, cnn_file, yolo_file, video_file, labels_file, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # initialize parameters
        self.inputs = []
        self.paths = []
        self.targets = []
        self.transform = transform

        # read data from json file
        yolo_json = open(yolo_file, 'r')
        yolo_output = yolo_json.readlines()
        yolo_json.close()
        yolo_output = json.loads(yolo_output[0])

        cnn_json = open(cnn_file, 'r')
        cnn_output = cnn_json.readlines()
        cnn_json.close()
        cnn_output = json.loads(cnn_output[0])

        # reformat the yolo json to give us a dictionary mapping the image_id to a list [category_id, bbox, confidence]
        pretrain_output_dict = {}
        for i in yolo_output:
            imgpath = i['image_id'].replace('\\', '/')

            # if YOLO fails to detect anything, just give us a bunch of zeros
            result = [0, 0, 0, 0, 0, 0, 0, 0]
            # check if YOLO actually detects anything
            if i['bbox']:
                # represent category as one-hot
                category = [0, 0, 0]
                category[i['category_id']] = 1
                # put bbox coord in range [0,1] (they are pixel values in yolo output)
                bbox = [c/320 for c in i['bbox']]
                result = category + bbox + [i['score']]

            pretrain_output_dict[imgpath] = result

        # add the CNN json info to the pretraining output dict,
        # value list is now [feature_vec, category_id, bbox, confidence] (flattened)
        for i in cnn_output:
            imgpath = i['image']
            # features is saved in the json as a list in a list
            features = i['features'][0]

            result = features + pretrain_output_dict[imgpath]
            pretrain_output_dict[imgpath] = result

        video_txt = open(video_file, 'r')
        video_list = video_txt.readlines()
        video_txt.close()

        labels_txt = open(labels_file, 'r')
        labels_list = labels_txt.readlines()
        labels_txt.close()

        # loop through all videos, add info to input/target lists
        for i in range(len(video_list)):
            # get rid of newlines at end of each line, and convert to actual python list of image paths/labels
            video = video_list[i][:-1].split(' ')
            labels = json.loads(labels_list[i][:-1])

            # loop through all frames of video, get features from video path
            video_features = [pretrain_output_dict[f] for f in video]

            self.inputs.append(np.array(video_features))
            self.paths.append(video)
            self.targets.append(np.array(labels))

        # convert input/target to numpy arrays
        self.inputs = np.array(self.inputs)
        self.paths = np.array(self.paths)
        self.targets = np.array(self.targets)


    def __len__(self):
        return len(self.inputs)

    # this function grabs a batch of images and labels, and reshapes it to fit LSTM input (seq_len, batch, features)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get batch
        inputs = self.inputs[idx]
        paths = self.paths[idx]
        targets = self.targets[idx]

        # make sure they have 3 dimensions:
        if len(inputs.shape) == 2:
            inputs = np.array([inputs]).shape
        if len(targets.shape) == 2:
            targets = np.array([targets]).shape

        # reshape to (seq_len, batch, features)
        inputs = np.swapaxes(inputs, 0, 1)
        targets = np.swapaxes(targets, 0, 1)

        sample = {'inputs': inputs, 'paths': paths, 'targets': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        inputs, paths, targets = sample['inputs'], sample['paths'], sample['targets']

        return {'inputs': torch.from_numpy(inputs),
                'paths': paths,
                'targets': torch.from_numpy(targets)}