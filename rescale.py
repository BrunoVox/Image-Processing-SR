import numpy as np 
from skimage.transform import rescale
from skimage.io import imsave, imread
from types import SimpleNamespace
from os import listdir

images = SimpleNamespace()
dsimages = SimpleNamespace()

images.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X4/'
dsimages.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X12/'

images.n_of_files = len(listdir(images.dir))

for i in range(0, images.n_of_files):
    temp_imread = imread('{}{}'.format(images.dir, listdir(images.dir)[i]))
    temp = rescale(temp_imread, 1/3)
    name = list(listdir(images.dir)[i])
    name[5] = '12'
    name = ''.join(name)
    imsave('{}{}'.format(dsimages.dir, name), temp)