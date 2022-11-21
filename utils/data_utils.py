"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
from tqdm import tqdm
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_(dir_):
    images = []
    #print(dir)
    
    #assert os.path.isdir(dir), '%s is not a valid directory' % dir
    data = os.listdir(dir_)[:2]
    #print(data)
    for dir in tqdm(data):
           
        dir = os.path.join(dir_ , dir)
        print(dir)
        for root, _, fnames in sorted(os.walk(dir)):
            #print(root , _ , fnames)
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images
def make_dataset_new(dir):
    return [os.path.join(dir , x) for x in os.listdir(dir)]

def make_dataset(dir):
    images = []
    #print(dir)
    for data in tqdm(os.listdir(dir)):
        #print(data)
        data = os.path.join(dir , data)
        for files in os.listdir(data):
            #if is_image_file(files):
            path = os.path.join(data , files)
            images.append(path)
    print(len(images))
    return images
