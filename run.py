import numpy as np 
import pickle
from types import SimpleNamespace
from skimage.color import ycbcr2rgb, rgb2ycbcr, rgb2gray, gray2rgb
from matplotlib import pyplot as plt
from skimage.transform import resize
import cv2
from skimage.util import pad
from skimage.io import imread, imsave
from scipy.sparse import csc_matrix, isspmatrix
from scipy import sparse
from scipy.signal import convolve2d, convolve
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from spams import lasso, omp

def gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

hrdictionary_dir = r'/home/vox/Image-Processing-SR/obj/hrdict.pkl'
hr = SimpleNamespace()
lrdictionary_dir = r'/home/vox/Image-Processing-SR/obj/lrdict.pkl'
lr = SimpleNamespace()
finalimages_dir = r'/home/vox/Image-Processing-SR/output/'
with open(hrdictionary_dir, 'rb') as f:
    hr.dictionary = pickle.load(f)
with open(lrdictionary_dir, 'rb') as f:
    lr.dictionary = pickle.load(f)
lmbd = 0.1
test = SimpleNamespace()
test.dir = r'/home/vox/Image-Processing-SR/test/X3/'
test.name = r'0001x3.jpg'
test.img = imread('{}{}'.format(test.dir, test.name))
test.img = rgb2gray(test.img)
test.height = (test.img).shape[0]
test.width = (test.img).shape[1]
test.img = np.pad(test.img, 1, 'constant', constant_values=0)

f1 = [-1,0,1]
final_image = np.zeros(((test.height*3+8, test.width*3+8, 2)))  # CORRIGIR
patch = extract_patches_2d(test.img, (3,3))
all_patches = np.zeros(((test.height) * (test.width), 9, 9))
future_patch = np.zeros((9,9))
row = 0
col = 0
t_index = 1
invalid_row = 1
invalid_col = 1
w = np.zeros((9,9))
for i in range(0, patch.shape[0]): #(test.height-2) * (test.width-2)):    
    if invalid_row > 0 and invalid_col > 0:
        patch_temp = patch[i]
        mean_pxl = np.mean(patch_temp)
        patch_temp = np.ravel(patch_temp)#, ((9,1)))
        patch_temp = convolve(patch_temp, f1, mode='same')
        patch_temp = np.reshape(patch_temp, (9,1))
        patch_temp = np.asfortranarray(patch_temp)
        
        hr_patch = all_patches[i,:,:]
        hr_patch = np.multiply(w, hr_patch)
        hr_patch = np.reshape(hr_patch, ((81,1)))
        hr_patch = np.asfortranarray(hr_patch)

        # CONCATENATE MATRICES

        full_dictionary = np.concatenate((lr.dictionary, hr.dictionary))
        full_patches = np.concatenate((patch_temp, hr_patch))

        alpha = lasso(X=full_patches, D=full_dictionary, lambda1=lmbd)
        alpha = alpha.todense()
        x = hr.dictionary @ alpha
        x = np.reshape(x, (9,9))
        all_patches[i] = x + mean_pxl
        
        # alphalr = omp(X=patch_temp, D=lr.dictionary, lambda1=lmbd)
        # alphalr = alphalr.todense()       

        # alphahr = omp(X=hr_patch, D=hr.dictionary, lambda1=lmbd)
        # alphahr = alphahr.todense()

        # if np.sum(abs(alphahr)) > np.sum(abs(alphalr)):
        #     # print('1')
        #     x = hr.dictionary @ alphalr
        #     # x = sparse.csc_matrix(hr.dictionary).multiply(sparse.csc_matrix(alphalr)).todense()
        #     x = np.reshape(x, ((9,9)))
        #     all_patches[i] = x + mean_pxl

        # else:
        #     # print('2')
        #     x = hr.dictionary @ alphahr
        #     # x = sparse.csc_matrix(hr.dictionary).multiply(sparse.csc_matrix(alphahr)).todense()
        #     x = np.reshape(x, ((9,9)))
        #     all_patches[i] = x + mean_pxl
        # # print(row,',',col)
        # # print(np.shape(final_image))
        for m in range(9):
            for n in range(9):
                future_patch[m,n] = all_patches[i,m,n]
                if final_image[row+m,col+n,1] == 0:
                    final_image[row+m,col+n,0] = future_patch[m,n]
                    final_image[row+m,col+n,1] = 1

        # for m in range(9):
        #     for n in range(9):
        #         future_patch[m,n] = all_patches[i,m,n]
        #         if final_image[row+m,col+n,1] == 0:
        #             final_image[row+m,col+n,0] = future_patch[m,n]
        #             final_image[row+m,col+n,1] = 1
            
    prog = i/patch.shape[0]
    
    print(prog*100)

    threshold = (test.width) * (t_index)
    t = (i + 1) / threshold
    if t >= 1:
        invalid_row *= -1
        invalid_col = 1
        t_index += 1
        row += 3
        col = 0
        w = np.zeros((9,9))
        if invalid_row > 0:
            for m in range(9):
                if m <= 2:
                    for n in range(9):                
                        w[m,n] = 1
                else:
                    break
    else:
        invalid_col *= -1
        col += 3
        w = np.zeros((9,9))
        if row == 0 and invalid_col > 0:
            for m in range(9):
                for n in range(9):
                    if n <= 2:
                        w[m,n] = 1
        if row > 0 and invalid_col > 0:
            for m in range(9):
                for n in range(9):
                    if n <= 2 or m <= 2:
                        w[m,n] = 1


# count = -1
# for i in range(4, test.height*3-2, 3):
#     for j in range(4, test.width*3-2, 3):
#         count += 1
#         # for k in range(-4,5):
#         #     for l in range(-4,5):
#         #         if final_image[i+k,j+l] == 0:
#         #             final_image[i+k,j+l] = all_patches[count,k,l]
        
#         final_image[i-4:i+5,j-4:j+5] = all_patches[count,:,:]

# SRimg = np.zeros((final_image.shape[0],final_image.shape[1]))
SRimg = final_image[:,:,0]
ni = 1

for i in range(100):
# while np.any(SRimg) != np.any(old_SRimg):
    
    # old_SRimg = SRimg

    # term1 = test.img - resize(convolve2d(SRimg[:,:], gauss2D((3,3),1), mode='same'), (test.img.shape[0],test.img.shape[1]))
    # term2 = convolve2d(resize(term1, (final_image.shape[0],final_image.shape[1])), np.transpose(gauss2D((3,3),1)), mode='same')

    term1 = test.img - resize(SRimg[:,:], (test.img.shape[0],test.img.shape[1]))
    term2 = resize(term1, (final_image.shape[0],final_image.shape[1]))

    SRimg[:,:] = SRimg[:,:] + ni * (term2 + (SRimg[:,:] - final_image[:,:,0]))
#     # temp_imread = np.ravel(temp_imread)
# test.dir = r'/home/vox/Image-Processing-SR/test/X9/'
# test.name = r'0001x9.jpg'
# test.img = imread('{}{}'.format(test.dir, test.name))
# test.img = rgb2ycbcr(test.img)

# SRimg = new_SRimg

# imgcolors = resize(test.img, (test.height*3+2, test.width*3+2), anti_aliasing_sigma=1.2)

# SRimg[:,:,1] = imgcolors[:,:,1]

# SRimg[:,:,2] = imgcolors[:,:,2]

SRimg = np.delete(SRimg, -1, axis=1)
SRimg = np.delete(SRimg, -1, axis=1)
SRimg = np.delete(SRimg, -1, axis=1)
# SRimg = np.delete(SRimg, -1, axis=1)
SRimg = np.delete(SRimg, 0, axis=1)
SRimg = np.delete(SRimg, 0, axis=1)
SRimg = np.delete(SRimg, 0, axis=1)
SRimg = np.delete(SRimg, 0, axis=1)
SRimg = np.delete(SRimg, -1, axis=0)
SRimg = np.delete(SRimg, -1, axis=0)
SRimg = np.delete(SRimg, -1, axis=0)
SRimg = np.delete(SRimg, -1, axis=0)
SRimg = np.delete(SRimg, 0, axis=0)
SRimg = np.delete(SRimg, 0, axis=0)
SRimg = np.delete(SRimg, 0, axis=0)
SRimg = np.delete(SRimg, 0, axis=0)



SRimg[:,:] = ((SRimg[:,:] - SRimg[:,:].min()) * (1/(SRimg[:,:].max() - SRimg[:,:].min()) * 219)) + 16
SRimg = SRimg.astype('uint8')

# SRimg = gray2rgb(SRimg)

imsave('{}{}'.format(finalimages_dir, 'final_sem_filtro.png'), SRimg)
SRimg[:,:] = convolve2d(SRimg[:,:], gauss2D((3,3),1), mode='same')
imsave('{}{}'.format(finalimages_dir, 'final_com_filtro.png'), SRimg)
plt.imshow(SRimg, cmap='gray')
plt.show()