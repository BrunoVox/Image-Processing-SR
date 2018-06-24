import numpy as np 
from skimage.transform import rescale
from skimage.io import imsave, imread
from types import SimpleNamespace
from os import listdir

test = SimpleNamespace()
dstest = SimpleNamespace()

test.dir = r'/home/vox/Image-Processing-SR/test/X0/'
dstest.dir = r'/home/vox/Image-Processing-SR/test/X3/'

test.n_of_files = len(listdir(test.dir))

for i in range(0, test.n_of_files):
    temp_imread = imread('{}{}'.format(test.dir, listdir(test.dir)[i]))
    temp = rescale(temp_imread, 1/3)
    name = list(listdir(test.dir)[i])
    name[5] = '3'
    name = ''.join(name)
    imsave('{}{}'.format(dstest.dir, name), temp)