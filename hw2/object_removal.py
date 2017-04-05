import numpy as np
import scipy
from skimage import morphology, io, img_as_float, color
from scipy import ndimage
import matplotlib.pyplot as plt
import os
import sys
import pdb
import time
from sklearn import feature_extraction as sfe
sys.path.append("/Users/xiaolihe/Documents/Computer-Vision-534/hw2")
# %pdb on

def Getboundarypoints(filled_mask):
    subs_image = morphology.dilation(filled_mask) - filled_mask
    #     pdb.set_trace()
    [row_idxs, col_idxs] = np.where(subs_image != 0)
    # random permutation
    randidx = np.random.permutation(len(row_idxs))
    locs = [(row_idxs[idx],col_idxs[idx]) for idx in randidx]
    return locs

# get confidence value in the template window, and its filled status
def GetNeighborhoodWindow(ridx, colidx, new_img, grad_mag, conf,filled_mask, win_size):
    half_wsize = win_size / 2
    row_range = range(ridx - half_wsize, ridx + half_wsize + 1)
    col_range = range(colidx - half_wsize, colidx + half_wsize + 1)
    tmp_img = np.zeros((win_size, win_size))
    tmp_grad = np.zeros((win_size, win_size))
    tmp_conf = np.zeros((win_size, win_size))
    tmp_filled_mask = np.ones((win_size, win_size)) == False
    # row,column range in the img2bfilled
    if row_range[0]>=0 and col_range[0]>=0 and row_range[-1] < conf.shape[0] and col_range[-1] < conf.shape[1]:
        tmp_img = new_img[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
        tmp_conf = conf[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
        tmp_grad = grad_mag[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
        tmp_filled_mask = filled_mask[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
    else:
        minR = max(0, ridx - half_wsize)
        maxR = min(conf.shape[0], ridx + half_wsize + 1)
        minC = max(0, colidx - half_wsize)
        maxC = min(conf.shape[1], colidx + half_wsize + 1)
        tmp_img[minR - ridx + half_wsize : maxR - ridx + half_wsize,
        minC - colidx + half_wsize : maxC - colidx + half_wsize] = new_img[minR : maxR, minC : maxC]
        tmp_conf[minR - ridx + half_wsize : maxR - ridx + half_wsize,
        minC - colidx + half_wsize : maxC - colidx + half_wsize] = conf[minR : maxR, minC : maxC]
        tmp_grad[minR - ridx + half_wsize : maxR - ridx + half_wsize,
        minC - colidx + half_wsize : maxC - colidx + half_wsize] = grad_mag[minR : maxR, minC : maxC]
        tmp_filled_mask[minR - ridx + half_wsize : maxR - ridx + half_wsize,
        minC - colidx + half_wsize : maxC - colidx + half_wsize] = filled_mask[minR : maxR, minC : maxC]
    return tmp_img, tmp_grad, tmp_conf, tmp_filled_mask


## Find Matches, return best patch
def FindMatches(tmp, sample_img, source_mask, ValidMask, win_size):
    patches_list0 = sfe.image.extract_patches_2d(sample_img, (win_size,win_size))# array, shape = (n_patches, patch_heidth)
    mask_list = sfe.image.extract_patches_2d(source_mask, (win_size,win_size))# array, shape = (n_patches, patch_heidth)
    patches_list = patches_list0[np.sum(mask_list,(1,2))==win_size*win_size]

    dist_filter = (patches_list - tmp)**2 * ValidMask
    SSD = np.asarray([d.sum() for d in dist_filter])
    res_loc_1d, = np.where(SSD == SSD.min())
    # pdb.set_trace()
    return patches_list[res_loc_1d][0]

# filled in target region with the best match, and update correspoinding target_mask, source_mask, confidence value
def filledin(ploc, win_size, new_img, conf, target_mask, source_mask, bestmactchpatch):
    half_wsize = win_size/2
    row_range = range(ploc[0] - half_wsize, ploc[0] + half_wsize)
    col_range = range(ploc[1] - half_wsize, ploc[1] + half_wsize)
    for r in row_range:
        for c in col_range:
            if r >=0 and c>=0 and r<new_img.shape[0] and c< new_img.shape[1]:
                if target_mask[r,c]:
                    new_img[r,c] = bestmactchpatch[r-ploc[0] + win_size/2, c-ploc[1] + win_size/2]
                    target_mask[r,c] = False
                    source_mask[r,c] = True
                    conf[r,c] = conf[r - ploc[0] + win_size/2, c - ploc[1] + win_size/2]


def objectremoval(filename = 'hollywood', filetype = '.jpg', maskname = 'hollywood-mask.bmp', win_size = 9):
    # read sample image
    t = time.time()
    # img_sample0 = io.imread(os.getcwd() + '/Documents/Computer-Vision-534/hw2/Assignment-II-images/' + filename + filetype)
    img_sample0 = io.imread(os.getcwd() + '/Assignment-II-images/' + filename + filetype)
    img_sample0 = color.rgb2gray(img_sample0) # rgb to gray level
    img = img_as_float(img_sample0) # convert image to 0-1 values
    new_img = np.copy(img)
    alpha = 255
    ## load mask
    # mask = img_as_float(color.rgb2gray(io.imread(os.getcwd() + '/Documents/Computer-Vision-534/hw2/Assignment-II-images/' + maskname + '.jpg')))
    mask = img_as_float(color.rgb2gray(io.imread(os.getcwd() + '/Assignment-II-images/' + maskname)))
    mask = mask.astype('bool') # target region: 1;  else: 0
#     plt.imshow(mask,'gray')
#     plt.show()
    target_mask = mask
    source_mask = mask == False
    # initialize confidence term and data term
    conf = (source_mask).astype('double')
    D = (source_mask).astype('double')
    # print conf
    idx = 0

    while target_mask.any():
        print target_mask.sum()
        ## find pixels on the contour
        fill_front = Getboundarypoints(np.logical_not(mask)) #(x,y) location

        ## calculate gradient
        grad_x = ndimage.sobel(img, 0)
        grad_y = ndimage.sobel(img, 1)
        grad_x[target_mask] = 0.0
        grad_y[target_mask] = 0.0
        # magnitude
        grad_mag = np.sqrt(np.array(grad_x)**2 + np.array(grad_y)**2)

        # calculate normals
        norm_x = ndimage.sobel(source_mask, 0)
        norm_y = ndimage.sobel(source_mask, 1)

        Prior = []
        for p in fill_front:
            ## Computing patch priorities
            tmp_img, tmp_grad, tmp_c, tmp_filled_mask = GetNeighborhoodWindow(p[0], p[1],new_img, grad_mag, conf, source_mask, win_size)
            # plt.imshow(tmp_img,'gray')
            # plt.figure()
            # plt.imshow(tmp_filled_mask,'gray')
            # plt.show()
            # Confidence value (C(q) within template, and the area of the template)
            conf[p] = (tmp_c*tmp_filled_mask).sum()/(win_size*win_size)
            # Data (find max gradient, normal direction n_p within template)
            maxgrad_locx,maxgrad_locy = np.where(tmp_grad == tmp_grad.max())
            norm_px,norm_py = norm_x[p], norm_y[p]
            norm_px = np.array([0,0]) if np.isnan(norm_px) else norm_px
            norm_py = np.array([0,0]) if np.isnan(norm_py) else norm_py
            D[p] =  np.linalg.norm(grad_x[maxgrad_locx[0] + p[0] - win_size/2, maxgrad_locy[0] + p[1] - win_size/2] * norm_px +
                                       grad_y[maxgrad_locx[0] + p[0] - win_size/2, maxgrad_locy[0] + p[1] - win_size/2] * norm_py ) / alpha
            # priority
            Prior.append(conf[p]*D[p])

        ## Propagating texture and structure information
        # find max pirority patch
        Prior = np.array(Prior)
        prior_idx, = np.where(Prior == Prior.max())
        prior_ploc = fill_front[prior_idx[np.random.randint(0, len(prior_idx))]] # if there are more than 1 choice, randomly choose 1
        tmp_img, tmp_grad, tmp_c, tmp_filled_mask = GetNeighborhoodWindow(prior_ploc[0], prior_ploc[1],new_img, grad_mag, conf, source_mask, win_size)
        BestMatch_patch = FindMatches(tmp_img, new_img, source_mask, tmp_filled_mask, win_size)

        # fill in the patch and Updating confidence values.
        filledin(prior_ploc, win_size, new_img, conf, target_mask, source_mask, BestMatch_patch)
        # idx += 1
        # if idx%50 == 0:
        #     plt.imshow(new_img,'gray')
        #     plt.show()
    elapsed = time.time() - t
    print 'Total runtime is ',elapsed
    plt.imshow(new_img,'gray')
    plt.title('Object removal for %s with windowsize %d (mask for %s)' %(filename,win_size,maskname))
    plt.savefig('Removal%s_size_%d_mask%s.png' %(filename,win_size,maskname[-1]))
    # plt.show()

objectremoval('test_im3','.jpg','test_im3_mask3.bmp', win_size = 9)
objectremoval('test_im3','.jpg','test_im3_mask2.bmp', win_size = 9)
objectremoval('test_im3','.jpg','test_im3_mask1.bmp', win_size = 9)
# check runtime of each function
# import cProfile
# cProfile.run('objectremoval()')




