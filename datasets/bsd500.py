from os.path import exists, join, basename, isfile
from os import makedirs, remove, listdir
import requests
import tarfile

url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

def download_bsd500(destination="data/bsd500/"):
    if (exists(destination)): 
        print("BSD500 already exists")
        return
    
    for fn in ("original", "train", "val", "test"):
        makedirs(join(destination, fn))
    destination = join(destination, "original")
    
    # download bsd 500 from url and place it somewhere
    print("Downloading data...", end=" ")
    archive = requests.get(url).content
    file_path = join(destination, "archive.gz")
    with open(file_path, 'wb') as f:
        f.write(archive)
    print("Success")

    # extract stuff from .gz package
    print("Extracting data...", end=" ")
    with tarfile.open(file_path) as tar:
        for item in tar:
            tar.extract(item, destination)    
    print("Success")
    
    remove(file_path)
    
def form_dataset(destination="data/bsd500/"):
    download_bsd500(destination)
    # patch data here