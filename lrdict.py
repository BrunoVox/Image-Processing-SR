import numpy as np 
from types import SimpleNamespace
from skimage.io import imread
from sklearn.feature_extraction.image import extract_patches_2d
from os import listdir
from random import sample, randint
import pickle
import gc
from spams import trainDL, lasso

lmbd = 0.1
lowres = SimpleNamespace()
lowres.dir = r'/home/vox/Image-Processing-SR/DIV2K_train_LR_bicubic/X12/'
lowres.n_of_files = len(listdir(lowres.dir))

### LOW RESOLUTION DICTIONARY ###

number_patches = 100
N = 9
K = lowres.n_of_files * number_patches
lowres.signal_forD = np.zeros((N,K))
samples = np.zeros((number_patches, lowres.n_of_files))

if K > N:
    print('Overcomplete dictionary ; Ok')
else:
    print('Undercomplete dictionary ; Problem')

for i in range(lowres.n_of_files):
    temp_imread = imread('{}{}'.format(lowres.dir, listdir(lowres.dir)[i]), as_gray=True)
    temp_patch = extract_patches_2d(temp_imread, (3,3))

    for m in range(samples.shape[0]):
        while samples[m,i] == 0:
            patch_index = randint(1, len(temp_patch) - 1)
            if patch_index not in samples[:,i]:
                samples[m,i] = patch_index

    patch = np.zeros((N, number_patches))
           
    for m in range(number_patches):
        temp_col_patch = np.ravel(temp_patch[samples[m,i].astype(int)])
        for l in range(N):
            patch[l,m] = temp_col_patch[l]
    lowres.signal_forD[:,number_patches * i:number_patches + i * number_patches] = patch[:,0:number_patches]

print('Training LR dictionary!!')

lowres.signal_forD = np.asfortranarray(lowres.signal_forD)
lowres.dictionary = trainDL(X=lowres.signal_forD, lambda1=lmbd, K=128, numThreads=1)

with open('obj/'+ 'lrdict' + '.pkl', 'wb') as f:
    pickle.dump(lowres.dictionary, f, pickle.HIGHEST_PROTOCOL)

with open('obj/'+ 'lrsignal' + '.pkl', 'wb') as f:
    pickle.dump(lowres.signal_forD, f, pickle.HIGHEST_PROTOCOL)

### HIGH RESOLUTION DICTIONARY ###

# N = 81
# K = highres.n_of_files * number_patches
# highres.signal_forD = np.zeros((N,K))

# if K > N:
#     print('Overcomplete dictionary ; Ok')
# else:
#     print('Undercomplete dictionary ; Problem')

# for i in range(highres.n_of_files):
#     temp_imread = imread('{}{}'.format(highres.dir, listdir(highres.dir)[i]), as_gray=True)
#     temp_patch = extract_patches_2d(temp_imread, (9,9))

#     patch = np.zeros((N, number_patches))
           
#     for m in range(number_patches):
#         temp_col_patch = np.ravel(temp_patch[samples[m,i].astype(int)])
#         for l in range(N):
#             patch[l,m] = temp_col_patch[l]       
#     highres.signal_forD[:,number_patches * i:number_patches + i * number_patches] = patch[:,0:number_patches]

# highres.signal_forD = np.asfortranarray(highres.signal_forD)
# highres.dictionary = trainDL(X=highres.signal_forD, lambda1=lmbd, K=128, numThreads=1)

### LASSO/ALPHA CALCULATION ###

# alpha = lasso(X=lowres.signal_forD, D=lowres.dictionary, lambda1=lmbd)

# test = SimpleNamespace()
# test.dir = r'/home/vox/Image-Processing-SR/test/'
# test.name = r'dog.jpg'
# test.img = imread('{}{}'.format(test.dir, test.name), as_gray=True)
# test.width = (test.img).shape[0]
# test.height = (test.img).shape[1]

# final_image = np.zeros((test.width, test.height))

# for i in range(test.height):
#     for j in range(test.width):
#         if i == 0 and j != 0 and j != test.width:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i,j + 1] + test.img[i,j - 1] + test.img[i + 1,j + 1] + test.img[i + 1,j - 1]) / 6
#         elif i == 0 and j == 0:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i,j + 1] + test.img[i + 1,j + 1]) / 4
#         elif i == test.height - 1 and j != 0 and j != test.width:
#             mean_pxl = (test.img[i,j] + test.img[i - 1,j] + test.img[i,j + 1] + test.img[i,j - 1] + test.img[i - 1,j + 1] + test.img[i - 1,j - 1]) / 6
#         elif i == test.height - 1 and j == 0:
#             mean_pxl = (test.img[i,j] + test.img[i - 1,j] + test.img[i,j + 1] + test.img[i - 1,j + 1]) / 4
#         elif i == 0 and j == test.width - 1:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i,j - 1] + test.img[i + 1,j - 1]) / 4
#         elif i == test.height - 1 and j == test.width - 1:
#             mean_pxl = (test.img[i,j] + test.img[i - 1,j] + test.img[i,j - 1] + test.img[i - 1,j - 1]) / 4
#         elif i != 0 and i != test.height - 1 and j == 0:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i - 1,j] + test.img[i,j + 1] + test.img[i + 1,j + 1] + test.img[i - 1,j + 1]) / 6
#         elif i != 0 and i != test.height - 1 and j == test.width - 1:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i - 1,j] + test.img[i,j - 1] + test.img[i + 1,j - 1] + test.img[i - 1,j - 1]) / 6
#         elif i != 0 and i != test.height - 1 and j != 0 and j != test.width - 1:
#             mean_pxl = (test.img[i,j] + test.img[i + 1,j] + test.img[i - 1,j] + test.img[i,j + 1] + test.img[i,j - 1] + test.img[i + 1,j + 1] + test.img[i + 1,j - 1] + test.img[i - 1,j + 1] + test.img[i - 1,j - 1]) / 9
        
#         x = np.dot(highres.dictionary, alpha)
#         final_image[i,j] = x + mean_pxl

# plt.imshow(final_image, interpolation='nearest')
# plt.show()