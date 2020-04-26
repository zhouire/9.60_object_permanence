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

    # input is a file path containing a pickled image dataset
    def __init__(self, pickle_file):
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

        for d in data:
            self.images.append(d[0])
            # NOTE: combining the bounding box and label vectors
            annot = np.concatenate((np.array(d[1][0]), np.array(d[1][1])))
            self.annotation.append(annot)

        self.images = np.array(self.images)
        self.annot = np.array(self.annot)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        annotation = self.annot[idx]

        sample = {'image': image, 'annotation': annotation}

        return sample