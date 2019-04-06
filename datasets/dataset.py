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
    def __init__(self, image_dir, patch_size=96, cr=4, **kwargs):
        super(RandomPatchFromFolder, self).__init__()
        
        self.dataset_from_folder = DatasetFromFolder(image_dir)
        self.cropper = RandomCrop(size=patch_size)
        self.resizer = Resize(size=int(patch_size/cr))

    @staticmethod
    def to_tensor(patch, normalization='lr'):
        tensor = torch.Tensor(np.asarray(patch).swapaxes(0, 2)).float() # from WHC to CHW
        if normalization == 'lr':
            return tensor / 255. # gives image in range [0, 1]
        elif normalization == 'hr':
            return (tensor / 127.5) - 1 # gives image in range [-1, 1]
        else: 
            raise NotImplemented("Type of normalization {} not recognized".format(normalization))
        
    @staticmethod
    def to_image(tensor, denormalize='hr'):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()

        if len(tensor.shape) == 4:
            tensor = tensor.swapaxes(1, 3)
        elif len(tensor.shape) == 3:
            tensor = tensor.swapaxes(0, 2)
        else:
            raise Exception("Predictions have shape not in set {3,4}")
            
        if denormalize == 'hr': # from [-1, 1] to [0, 255]
            tensor = (tensor + 1) * 127.5
        elif denormalize == 'lr': # from [0, 1] to [0, 255]
            tensor = tensor * 255
        else: 
            raise NotImplemented("Type of denormalization {} not recognized".format(denormalize)) 
        tensor[tensor > 255] = 255
        tensor[tensor < 0] = 0
        return tensor.round().astype(int)

    def __getitem__(self, index):
        img = self.dataset_from_folder[index]

        hr_patch = self.cropper(img)
        lr_patch = self.resizer(hr_patch)
        return self.to_tensor(lr_patch, 'lr'), self.to_tensor(hr_patch, 'hr')

    def __len__(self):
        return len(self.dataset_from_folder)
    
    
def ValidationSet(validation_dataset, n_images=50):
    return list(
        map(
            lambda tensors: torch.cat(tuple(
                map(
                    lambda t: t.unsqueeze(0), 
                    tensors)
            ), dim=0), 
         zip(*(validation_dataset[i] for i in range(n_images)))
        )
    )