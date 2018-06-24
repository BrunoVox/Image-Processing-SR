import numpy as np 
import pickle
from types import SimpleNamespace
from skimage.io import imread
# from matplotlib import pyplot as plt
from spams import lasso

### LASSO/ALPHA CALCULATION ###

lmbd = 0.1
lowres = SimpleNamespace()
highres = SimpleNamespace()
lowres.dictionary_dir = r'/home/vox/Image-Processing-SR/obj/lrdict.pkl'

with open(lowres.dictionary_dir, 'rb') as f:
    lowres.dictionary = pickle.load(f)

lowres.signal_dir = r'/home/vox/Image-Processing-SR/obj/lrsignal.pkl'

with open(lowres.signal_dir, 'rb') as f:
    lowres.signal = pickle.load(f)

lowres.dictionary = np.asfortranarray(lowres.dictionary)

alpha = lasso(X=lowres.signal, D=lowres.dictionary, lambda1=lmbd)

with open('obj/'+ 'alpha' + '.pkl', 'wb') as f:
    pickle.dump(alpha, f, pickle.HIGHEST_PROTOCOL)

# test = SimpleNamespace()
# test.dir = r'/home/vox/Image-Processing-SR/test/'
# test.name = r'dog.jpg'
# test.img = imread('{}{}'.format(test.dir, test.name), as_gray=True)
# test.height = (test.img).shape[0]
# test.width = (test.img).shape[1]

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