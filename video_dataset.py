# this is the pytorch dataset to pass video frames through the CNN

import os
import torch
import pandas as pd
import numpy as np
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# this dataset consists of x_data = images, y_data = (label_idx, x, y, w, h)
class VideoCNNDataset(Dataset):

    # input is txt files containing a list of image paths and labels, and whether or not the target model is YOLO
    # if working with CNN pretraining, just output the index of the one-hot label vector
    def __init__(self, imagefile, transform = None):
        # assume that the data is formatted so "images" and "labels" are parallel directories with the same filenames except .jpg == .txt
        image_file = open(imagefile, 'r')
        image_paths = image_file.readlines()
        image_file.close()

        self.images = []
        # we also want to have the path to each image for the CNN json output
        self.paths = []
        self.annotation = []
        self.transform = transform

        for p in range(len(image_paths)):
            # get rid of newline at end of line, convert str representation of list back to list
            imgpath = image_paths[p][:-1]
            img = np.array(Image.open(imgpath).convert('RGB'))
            self.images.append(img)
            self.paths.append(imgpath)

            # find and open the associated label txt file, read label
            lblpath = imgpath.replace("images", "labels").replace(".jpg", ".txt")
            labelfile = open(lblpath, 'r')
            label = labelfile.readlines()
            labelfile.close()

            label = list(map(float, label[0].split(' ')))

            # find index of 1 in one-hot label vector (first number in annotation)
            annot = np.array([label[0]])

            self.annotation.append(annot)

        self.images = np.array(self.images)
        self.paths = np.array(self.paths)
        self.annotation = np.array(self.annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        path = self.paths[idx]
        annotation = self.annotation[idx]

        sample = {'image': image, 'path': path, 'annotation': annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, path, annotation = sample['image'], sample['path'], sample['annotation']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'path': path,
                'annotation': torch.from_numpy(annotation)}