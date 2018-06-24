import numpy as np 
import pickle
from types import SimpleNamespace
from matplotlib import pyplot as plt
from skimage.util import pad
from skimage.io import imread
from scipy.sparse import csc_matrix, isspmatrix
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from spams import lasso

hrdictionary_dir = r'/home/vox/Image-Processing-SR/obj/hrdict.pkl'
hr = SimpleNamespace()
lrdictionary_dir = r'/home/vox/Image-Processing-SR/obj/lrdict.pkl'
lr = SimpleNamespace()

with open(hrdictionary_dir, 'rb') as f:
    hr.dictionary = pickle.load(f)

with open(lrdictionary_dir, 'rb') as f:
    lr.dictionary = pickle.load(f)

lmbd = 0.1
test = SimpleNamespace()
test.dir = r'/home/vox/Image-Processing-SR/test/X9/'
test.name = r'0001x9.jpg'
test.img = imread('{}{}'.format(test.dir, test.name), as_gray=True)
test.img = pad(test.img, 1, 'constant')
test.height = (test.img).shape[0]
test.width = (test.img).shape[1]

final_image = np.zeros((test.width*3-6, test.height*3-6))
patch = extract_patches_2d(test.img, (3,3))
print(patch.shape)
all_patches = np.zeros(((test.height-2) * (test.width-2), 3, 3))
print(all_patches.shape)

for i in range((test.height-2) * (test.width-2)):
    
    patch_temp = patch[i]
    mean_pxl = np.sum(patch_temp)/9
    patch_temp = np.asfortranarray(patch_temp)
    patch_temp = np.reshape(patch_temp, ((9,1)))

    alpha = lasso(X=patch_temp, D=lr.dictionary, lambda1=lmbd)
    alpha = alpha.todense()

    x = lr.dictionary @ alpha
    x = np.reshape(x, ((3,3)))
    all_patches[i] = x + mean_pxl

    prog = i/(test.height * test.width)

final_image = reconstruct_from_patches_2d(all_patches, (test.height,test.width))

plt.imshow(final_image, interpolation='nearest')
plt.show()