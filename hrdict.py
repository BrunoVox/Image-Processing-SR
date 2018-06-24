import numpy as np 
from lrdict import samples
from types import SimpleNamespace
from skimage.io import imread
from sklearn.feature_extraction.image import extract_patches_2d
from os import listdir
import pickle
import gc
from spams import trainDL, lasso

lmbd = 0.1
highres = SimpleNamespace()
highres.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X4/'
highres.n_of_files = len(listdir(highres.dir))

### HIGH RESOLUTION DICTIONARY ###

number_patches = 100
N = 81
K = highres.n_of_files * number_patches
highres.signal_forD = np.zeros((N,K))

if K > N:
    print('Overcomplete dictionary ; Ok')
else:
    print('Undercomplete dictionary ; Problem')

for i in range(highres.n_of_files):
    temp_imread = imread('{}{}'.format(highres.dir, listdir(highres.dir)[i]), as_gray=True)
    temp_patch = extract_patches_2d(temp_imread, (9,9))

    patch = np.zeros((N, number_patches))
           
    for m in range(number_patches):
        temp_col_patch = np.ravel(temp_patch[samples[m,i].astype(int)])
        for l in range(N):
            patch[l,m] = temp_col_patch[l]
    highres.signal_forD[:,number_patches * i:number_patches + i * number_patches] = patch[:,0:number_patches]

print('Training HR dictionary!!')

highres.signal_forD = np.asfortranarray(highres.signal_forD)
highres.dictionary = trainDL(X=highres.signal_forD, lambda1=lmbd, K=128, numThreads=1)

with open('obj/'+ 'hrdict' + '.pkl', 'wb') as f:
    pickle.dump(highres.dictionary, f, pickle.HIGHEST_PROTOCOL)

with open('obj/'+ 'hrsignal' + '.pkl', 'wb') as f:
    pickle.dump(highres.signal_forD, f, pickle.HIGHEST_PROTOCOL)