from os.path import exists, join, basename, isfile, splitext
from os import makedirs, remove, listdir
from shutil import move 
from pathlib import Path
import requests
from zipfile import ZipFile 
import numpy as np
import imageio
from PIL import Image

train_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
val_url = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

def not_empty(directory):
    return len(listdir(directory)) > 0

def download_div2k(destination="data/div2k/"):    
    for fn in ("original", "train/hr", "train/lr", "val/hr", "val/lr"):
        if not exists(join(destination, fn)):
            makedirs(join(destination, fn))
    destination = join(destination, "original")
    
    # download div2k from url and place it somewhere
    if (not_empty(destination)):
        print("div2k already exists")
        return 
    
    print("Downloading data...", end=" ")
    archive = requests.get(train_url).content
    train_archive = join(destination, "train_archive.zip")
    with open(train_archive, 'wb') as f:
        f.write(archive)
    archive = requests.get(val_url).content
    val_archive = join(destination, "val_archive.zip")
    with open(val_archive, 'wb') as f:
        f.write(archive)
    print("Success")

    # extract stuff from .gz package
    print("Extracting data...", end=" ")
    with ZipFile(train_archive) as archive:
        archive.extractall(destination)
        p = Path(join(destination, "DIV2K_train_HR")).absolute()
        p.rename(p.parents[0] / "train")
    with ZipFile(val_archive) as archive:
        archive.extractall(destination)    
        p = Path(join(destination, "DIV2K_valid_HR")).absolute()
        p.rename(p.parents[0] / "val")
    print("Success")
    
    remove(train_archive)
    remove(val_archive)

def is_img_file(path):
    return isfile(path) and any(path.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def np_resize(img, w, h, mode=Image.BICUBIC):
    return np.array(Image.fromarray(img).resize((w, h), mode)) 
    
def patch_img_directory(src, dest, patch_w=64, patch_h=64, cr=2):
    for i, file in enumerate(listdir(src)):
        file = join(src, file)
        if not is_img_file(file): 
            continue
        img = np.asarray(imageio.imread(file))
        print(img.shape)
        img_w, img_h, ch = img.shape

        x = 0
        while((x + patch_w) < img_w):
            y = 0
            while ((y + patch_h) < img_h):
                print("{}.{}.{}".format(i, x//patch_w, y//patch_h))
                
                patch_hr = img[x : x + patch_w, y : y + patch_h]
                patch_lr = np_resize(patch_hr, patch_w//cr, patch_h//cr, Image.BICUBIC)

                patch_fn = "{}_{}_{}.png".format(splitext(basename(file))[0], x//patch_w, y//patch_h)
                imageio.imwrite(join(dest, "hr", patch_fn), patch_hr)
                imageio.imwrite(join(dest, "lr", patch_fn), patch_lr)
                
                y += patch_h
            x += patch_w    
            
def form_dataset(destination="data/div2k/", patch_w=128, patch_h=128, cr=2):
    download_div2k(destination)
    data_destination = join(destination, "original")
    
    print("Forming train set...")
    patch_img_directory(join(data_destination, "train"), join(destination, "train"), patch_w, patch_h, cr)
    print("Forming val set...")
    patch_img_directory(join(data_destination, "val"), join(destination, "val"), patch_w, patch_h, cr)
    print("Success!")
#     for i, file in enumerate(listdir(data_destination)):
#         if not isfile(join(data_destination, file)): continue
#         file = join(data_destination, file)
#         img = Image.open(file)
#         array = np.array(img)
#         p = Path(file)
        
#         x = 0
#         while((x + patch_w) < img.width):
#             y = 0
#             while ((y + patch_h) < img.height):
#                 print("{}.{}.{}".format(i, x//patch_w, y//patch_h))
#                 patch = array[y : y + patch_h, x : x + patch_w]
#                 patch_hr = Image.fromarray(patch, 'RGB')
#                 patch_lr = patch_hr.resize((patch_w//2, patch_h//2), Image.BICUBIC)
                
#                 patch_fn = "{}_{}_{}.png".format(p.name[:-4], x//patch_w, y//patch_h)
#                 patch_lr.save(join(train_destination, "lr", patch_fn))
#                 patch_hr.save(join(train_destination, "hr", patch_fn))
#                 y += patch_h
#             x += patch_w
#     print("Success!")