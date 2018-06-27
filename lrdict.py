import numpy as np 
from types import SimpleNamespace
from skimage.io import imread
from scipy.misc import imresize
from skimage.color import rgb2ycbcr, rgb2gray
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.signal import convolve
from os import listdir
from random import sample, randint
import pickle
from spams import trainDL, lasso

f1 = [-1,0,1]
f2 = np.transpose(f1)
f3 = [1,0,-2,0,1]
f4 = np.transpose(f3)

lmbd = 0.1
lowres = SimpleNamespace()
lowres.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X12/'
lowres.n_of_files = len(listdir(lowres.dir))

### LOW RESOLUTION DICTIONARY ###

number_patches = 100
N = 9
K = lowres.n_of_files * number_patches
lowres.signal_forD = np.zeros((N,K,4))
lowres.signal = np.zeros((N,K*4))
samples = np.zeros((number_patches, lowres.n_of_files))

if K > N:
    print('Overcomplete dictionary ; Ok')
else:
    print('Undercomplete dictionary ; Problem')

for i in range(lowres.n_of_files):
    temp_imread = imread('{}{}'.format(lowres.dir, listdir(lowres.dir)[i]))
    temp_imread = rgb2gray(temp_imread)
    tis0 = temp_imread.shape[0]
    tis1 = temp_imread.shape[1]
    temp_imread = np.ravel(temp_imread)
    # temp_imread = imresize(temp_imread, 200, interp='bicubic')
    lowres.img1 = convolve(temp_imread, f1, mode='same')
    lowres.img1 = np.reshape(lowres.img1, (tis0,tis1))
    lowres.img2 = convolve(temp_imread, f2, mode='same')
    lowres.img2 = np.reshape(lowres.img2, (tis0,tis1))
    lowres.img3 = convolve(temp_imread, f3, mode='same')
    lowres.img3 = np.reshape(lowres.img3, (tis0,tis1))
    lowres.img4 = convolve(temp_imread, f4, mode='same')
    lowres.img4 = np.reshape(lowres.img4, (tis0,tis1))
    temp_patch1 = extract_patches_2d(lowres.img1, (3,3))
    temp_patch2 = extract_patches_2d(lowres.img2, (3,3))
    temp_patch3 = extract_patches_2d(lowres.img3, (3,3))
    temp_patch4 = extract_patches_2d(lowres.img4, (3,3))

    for m in range(samples.shape[0]):
        while samples[m,i] == 0:
            # patch_index = randint(1, len(temp_patch1) - 1)
            patch_index = randint(1,(temp_patch1.shape[0]-2)*(temp_patch1.shape[1]-2))
            if patch_index not in samples[:,i] and abs(temp_patch1[patch_index].max() / temp_patch1[patch_index].min()) > np.mean(temp_patch1[patch_index]):
                samples[m,i] = patch_index

    patch = np.zeros((N, number_patches, 4))
           
    for m in range(number_patches):
        temp_col_patch1 = np.ravel(temp_patch1[samples[m,i].astype(int)])
        temp_col_patch2 = np.ravel(temp_patch2[samples[m,i].astype(int)])
        temp_col_patch3 = np.ravel(temp_patch3[samples[m,i].astype(int)])
        temp_col_patch4 = np.ravel(temp_patch4[samples[m,i].astype(int)])
        for l in range(N):
            patch[l,m,0] = temp_col_patch1[l]
            patch[l,m,1] = temp_col_patch2[l]
            patch[l,m,2] = temp_col_patch3[l]
            patch[l,m,3] = temp_col_patch4[l]
    lowres.signal_forD[:,number_patches * i:number_patches + i * number_patches,0] = patch[:,0:number_patches,0]
    lowres.signal_forD[:,number_patches * i:number_patches + i * number_patches,1] = patch[:,0:number_patches,1]
    lowres.signal_forD[:,number_patches * i:number_patches + i * number_patches,2] = patch[:,0:number_patches,2]
    lowres.signal_forD[:,number_patches * i:number_patches + i * number_patches,3] = patch[:,0:number_patches,3]

for i in range(0,K*4,4):
    lowres.signal[:,i] = np.ravel(lowres.signal_forD[:,int(i/4),0])
    lowres.signal[:,i+1] = np.ravel(lowres.signal_forD[:,int(i/4),1])
    lowres.signal[:,i+2] = np.ravel(lowres.signal_forD[:,int(i/4),2])
    lowres.signal[:,i+3] = np.ravel(lowres.signal_forD[:,int(i/4),3])

print('Training LR dictionary!!')

lowres.signal = np.asfortranarray(lowres.signal)
lowres.dictionary = trainDL(X=lowres.signal, lambda1=lmbd, K=1024, numThreads=1, batchsize=64, iter=100)

with open('obj/'+ 'lrdict' + '.pkl', 'wb') as f:
    pickle.dump(lowres.dictionary, f, pickle.HIGHEST_PROTOCOL)

with open('obj/'+ 'lrsignal' + '.pkl', 'wb') as f:
    pickle.dump(lowres.signal, f, pickle.HIGHEST_PROTOCOL)

