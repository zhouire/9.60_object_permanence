import os
import torch
import pandas as pd
import numpy as np
import pickle
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# this dataset consists of x_data = images, y_data = (x, y, w, h, label)
class ShapeImageDataset(Dataset):

    # input is a file path containing a pickled image dataset, and whether or not the target model is YOLO
    # if working with CNN pretraining, just output the index of the one-hot label vector
    def __init__(self, pickle_file, yolo = True, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        file = open(pickle_file, 'rb')
        data = pickle.load(file)
        file.close()

        self.images = []
        self.annotation = []
        self.transform = transform

        for d in data:
            self.images.append(d[0])
            if yolo:
                # NOTE: combining the bounding box and label vectors
                annot = np.concatenate((np.array(d[1][0]), np.array(d[1][1])))
            else:
                # find index of 1 in one-hot label vector
                annot = np.array([d[1][1].index(1)])

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