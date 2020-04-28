import os
import torch
import pandas as pd
import numpy as np
import cv2
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


# this dataset consists of x_data = images, y_data = (x, y, w, h, label)
class ShapeImageDataset(Dataset):

    # input is txt files containing a list of image paths and labels, and whether or not the target model is YOLO
    # if working with CNN pretraining, just output the index of the one-hot label vector
    def __init__(self, imagefile, labelfile, yolo = True, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        image_file = open(imagefile, 'r')
        label_file = open(labelfile, 'r')
        image_paths = image_file.readlines()
        image_labels = label_file.readlines()
        image_file.close()
        label_file.close()


        self.images = []
        self.annotation = []
        self.transform = transform

        for p in range(len(image_paths)):
            # get rid of newline at end of line, convert str representation of list back to list
            img = np.array(Image.open(image_paths[p][:-1]).convert('RGB'))
            label = json.loads(image_labels[p][:-1])
            self.images.append(img)

            if yolo:
                # NOTE: combining the bounding box and label vectors
                annot = np.array(label)
            else:
                # find index of 1 in one-hot label vector (first number in annotation)
                annot = np.array([label[0]])

            self.annotation.append(annot)

        self.images = np.array(self.images)
        self.annotation = np.array(self.annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        annotation = self.annotation[idx]

        sample = {'image': image, 'annotation': annotation}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, annotation = sample['image'], sample['annotation']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'annotation': torch.from_numpy(annotation)}