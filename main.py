import numpy as np 
from types import SimpleNamespace
from skimage.io import imread
from os import listdir

f1 = [-1, 0, 1]
f2 = np.transpose(f1)
f3 = [1, 0, -2, 0 ,1]
f4 = np.transpose(f3)
lmbd = 0.1

lowres = SimpleNamespace()
highres = SimpleNamespace()

lowres.dir = r'C:/ImageProcessing/DIV2K_train_LR_bicubic_X4/DIV2K_train_LR_bicubic/X4/'
highres.dir = r'C:/ImageProcessing/DIV2K_train_HR/DIV2K_train_HR/'

lowres.n_of_files = len(listdir(lowres.dir))
highres.n_of_files = len(listdir(highres.dir))

# lowres.reso_matrix = np.zeros((lowres.n_of_files,4))
# for i in range(lowres.n_of_files):
#     temp_imread = imread('{}{}'.format(lowres.dir, listdir(lowres.dir)[i]))
#     temp_shape = np.shape(temp_imread)
#     lowres.reso_matrix[i,0] = temp_shape[0]
#     lowres.reso_matrix[i,1] = temp_shape[1]
#     lowres.reso_matrix[i,2] = temp_shape[2]
#     lowres.reso_matrix[i,3] = temp_shape[0] * temp_shape[1]

dict_matrix = np.random.normal(128, 128, (10,10))
print((dict_matrix))