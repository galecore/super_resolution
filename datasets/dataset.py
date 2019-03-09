import torch.utils.data as data_utils
import torch
from os import listdir
from os.path import join
import imageio
import numpy as np
from PIL import Image
from torchvision.transforms import RandomCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


class DatasetFromFolder(data_utils.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_dir = image_dir
        self.image_filenames = [x for x in listdir(image_dir) if is_image_file(x)]

    def __getitem__(self, index):
        return Image.fromarray(imageio.imread(join(self.image_dir, self.image_filenames[index])))

    def __len__(self):
        return len(self.image_filenames)


class RandomPatchFromFolder(data_utils.Dataset):
    def __init__(self, image_dir, patch_size=64, cr=2):
        super(RandomPatchFromFolder, self).__init__()
        self.dataset_from_folder = DatasetFromFolder(image_dir)
        self.cropper = RandomCrop(size=patch_size*cr)
        self.resizer = Resize(size=patch_size)

    @staticmethod
    def to_tensor(patch):
        return torch.Tensor(np.asarray(patch).swapaxes(0, 2)).float() / 255 - 0.5

    @staticmethod
    def to_image(tensor):
        if type(tensor) == torch.Tensor:
            tensor = tensor.numpy()

        if len(tensor.shape) == 4:
            tensor = tensor.swapaxes(1, 3)
        elif len(tensor.shape) == 3:
            tensor = tensor.swapaxes(0, 2)
        else:
            raise Exception("Predictions have shape not in set {3,4}")

        tensor = (tensor + 0.5) * 255
        tensor[tensor > 255] = 255
        tensor[tensor < 0] = 0
        return tensor.round().astype(int)

    def __getitem__(self, index):
        img = self.dataset_from_folder[index]

        hr_patch = self.cropper(img)
        lr_patch = self.resizer(hr_patch)
        to_tensor = RandomPatchFromFolder.to_tensor
        return to_tensor(lr_patch), to_tensor(hr_patch)

    def __len__(self):
        return len(self.dataset_from_folder)
