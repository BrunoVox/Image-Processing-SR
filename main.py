import numpy as np 
from types import SimpleNamespace
from skimage.io import imread
from os import listdir
from spams import trainDL

f1 = [-1, 0, 1]
f2 = np.transpose(f1)
f3 = [1, 0, -2, 0 ,1]
f4 = np.transpose(f3)
lmbd = 0.1

lowres = SimpleNamespace()
highres = SimpleNamespace()

lowres.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X4/'
highres.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X2/'

lowres.n_of_files = len(listdir(lowres.dir))
highres.n_of_files = len(listdir(highres.dir))

lowres.reso_matrix = np.zeros((lowres.n_of_files,4))

for i in range(lowres.n_of_files):
    temp_imread = imread('{}{}'.format(lowres.dir, listdir(lowres.dir)[i]))
    temp_shape = np.shape(temp_imread)
    lowres.reso_matrix[i,0] = temp_shape[0]
    lowres.reso_matrix[i,1] = temp_shape[1]
    lowres.reso_matrix[i,2] = temp_shape[2]
    lowres.reso_matrix[i,3] = temp_shape[0] * temp_shape[1]

lowres.signal_forD = np.zeros((int(max(lowres.reso_matrix[:,3])),lowres.n_of_files))

for j in range(lowres.n_of_files):
    temp_imread = imread('{}{}'.format(lowres.dir, listdir(lowres.dir)[j]))
    temp_pixels = np.ravel(temp_imread[:,:,0])
    temp_size = np.shape(temp_pixels)[0]
    for i in range(int(temp_size)):
        lowres.signal_forD[i,j] = temp_pixels[i]

lowres.signal_forD = np.asfortranarray(lowres.signal_forD)
lowres.dictionary = trainDL(lowres.signal_forD, lambda1=0.1, K=512)