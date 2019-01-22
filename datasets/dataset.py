import torch.utils.data as data
import torch
from os import listdir
from os.path import join
import imageio
import numpy as np
from utils import utils

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,): #  input_transform=None, target_transform=None
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = [x for x in listdir(join(image_dir, "hr")) if is_image_file(x)]
        
#         self.input_transform = input_transform
#         self.target_transform = target_transform

    def __getitem__(self, index):
        data = utils.read_data(join(self.image_dir, "lr", self.image_filenames[index]))
        target = utils.read_data(join(self.image_dir, "hr", self.image_filenames[index]))
#         if self.input_transform:
#             data = self.input_transform(data)
#         if self.target_transform:
#             target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.image_filenames)