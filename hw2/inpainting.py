# coding: utf-8

# In[1]:

# %reset
import numpy as np
import scipy
from skimage import morphology, io, img_as_float
import matplotlib.pyplot as plt
import os
import sys
import pdb
import time
from sklearn import feature_extraction as sfe
plt.interactive(False)
sys.path.append("/Users/xiaolihe/Documents/Computer-Vision-534/hw2")

# In[2]:


# In[ ]:

## growimage
def inpainting(filename, filetype, win_size, sample_winsize = 51):
	"""
	:param filename: sample image
	:param filetype: type of image
	:param win_size: size of synthesis window
	:return: save figures to png file.
	"""
	# read sample image
	t = time.time()
	img_sample0 = io.imread(os.getcwd() + '/Assignment-II-images/' + filename + filetype)
	img_sample = img_as_float(img_sample0)
	img2bfilled = np.copy(img_sample)
	filled_status = img_sample != 0

	# inpainting
	img_new = GrowImage(img_sample, img2bfilled, filled_status, win_size, sample_winsize)
	elapsed = time.time() - t
	print 'Total runtime is ',elapsed

	# show and save image
	plt.imshow(img2bfilled,cmap = 'gray')
	plt.title('Initial patch')
	plt.show()
	plt.imshow(img_sample,'gray')
	plt.title('sample image')
	plt.show()
	plt.imshow(img_new,'gray')
	plt.title('Sythesized image for %s with windowsize %d' %(filename,win_size))
	plt.savefig('Inpainting_%s_size_%d.png' %(filename,win_size))
	plt.show()



def GrowImage(sample_img, img2bfilled, filled_status, win_size, sample_winsize):
	MaxErrThreshold = 0.3
	ct = 0
# 	pdb.set_trace()
	while not filled_status.all():
		print 'filled pixels is ', ct
		progress = 0
		row_idxs, col_idxs = GetUnfilledNeighbors(filled_status, win_size)
		for ridx, colidx in zip(row_idxs, col_idxs):
			template, validmask_tmp = GetNeighborhoodWindow(ridx, colidx, img2bfilled, filled_status, win_size)
			sample_imgrange, sam_filled_status = GetNeighborhoodWindow(ridx, colidx, img2bfilled, filled_status, sample_winsize)
			BMs_list,BMs_ssd, BMs_pixels= FindMatches(template, sample_imgrange, sam_filled_status, validmask_tmp, win_size, sample_winsize)
			BM_loc,BM_ssd,BM_value = RandomPick(BMs_list, BMs_ssd, BMs_pixels)
			# filled if BM_ssd is smaller than MaxErrThreshold
			if BM_ssd < MaxErrThreshold:
				img2bfilled[ridx, colidx] = BM_value
				progress = 1
				filled_status[ridx, colidx] = True
				ct+=1
		# pdb.set_trace()
		plt.imshow(img2bfilled, cmap='gray')
		plt.show()
		if progress == 0:
			MaxErrThreshold *= 1.1
			print 'increase Max error threshold to', MaxErrThreshold
	return img2bfilled



## GetUnfilledNeighbors
def GetUnfilledNeighbors(filled_status, win_size):
	subs_image = morphology.dilation(filled_status) - filled_status
	#     pdb.set_trace()
	[row_idxs, col_idxs] = np.where(subs_image != 0)
	# random permutation
	randidx = np.random.permutation(len(row_idxs))
	row_idxs = [row_idxs[idx] for idx in randidx]
	col_idxs = [col_idxs[idx] for idx in randidx]
	# sorted by decreasing number of filled neighbor pixels
	filledsum = scipy.ndimage.generic_filter((filled_status!=0).astype('double'), np.sum, win_size)
	#     pdb.set_trace()
	# filled sum for the boundary points in row_idxs,col_idxs
	filledsum_bs = [filledsum[x, y] for x, y in zip(row_idxs, col_idxs)]
	#     pdb.set_trace()
	sorted_idx = np.array(filledsum_bs).argsort()[::-1]
	res_row = [row_idxs[i] for i in sorted_idx]
	res_col = [col_idxs[i] for i in sorted_idx]
	return res_row, res_col

## GetNeighborhoodWindow
def GetNeighborhoodWindow(ridx, colidx, img2bfilled,filled_status, win_size):
	half_win_size = win_size / 2
	row_range = range(ridx - half_win_size, ridx + half_win_size + 1)
	col_range = range(colidx - half_win_size, colidx + half_win_size + 1)
	template = np.zeros((win_size, win_size))
	template_filled_status = np.ones((win_size, win_size)) == False
	# row,column range in the img2bfilled
	if row_range[0]>=0 and col_range[0]>=0 and row_range[-1] < img2bfilled.shape[0] and col_range[-1] < img2bfilled.shape[1]:
		template = img2bfilled[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
		template_filled_status = filled_status[row_range[0]:row_range[-1]+1,col_range[0]:col_range[-1]+1]
	else:
		minr_img2f = max(0, ridx - half_win_size)
		maxr_img2f = min(img2bfilled.shape[0], ridx + half_win_size + 1)
		minc_img2f = max(0, colidx - half_win_size)
		maxc_img2f = min(img2bfilled.shape[1], colidx + half_win_size + 1)
		template[minr_img2f-ridx+half_win_size : maxr_img2f-ridx+half_win_size,
		minc_img2f-colidx+half_win_size : maxc_img2f-colidx+half_win_size] = img2bfilled[minr_img2f : maxr_img2f, minc_img2f:maxc_img2f]
		template_filled_status[minr_img2f-ridx+half_win_size:maxr_img2f-ridx+half_win_size,
		minc_img2f-colidx+half_win_size : maxc_img2f-colidx+half_win_size] = filled_status[minr_img2f : maxr_img2f, minc_img2f:maxc_img2f]

	return template, template_filled_status

## Find Matches, return locations, corresponding ssd (in 1d), and corresponding pixel value
def FindMatches(template, sample_range, sam_filled_status, ValidMask_tmp, win_size, sample_winsize):
	Sigma = win_size / 6.4
	ErrThreshold = 0.1
	## define sample_img, location in terms of (ridx,colidx)
	patches_list = sfe.image.extract_patches_2d(sample_range, (win_size,win_size))# array, shape = (n_patches, patch_heidth)
	validmask_list = sfe.image.extract_patches_2d(sam_filled_status, (win_size,win_size))
	patches_nonzero = patches_list[np.sum(validmask_list,(1,2))==win_size*win_size]
	GaussMask = gkern(win_size, Sigma)
	mask_raw = ValidMask_tmp * GaussMask
	mask_normalized = mask_raw/mask_raw.sum()
	dist_filter = (patches_nonzero - template)**2*mask_normalized
	SSD = np.asarray([d.sum() for d in dist_filter])
	thr = np.nanmin(SSD) * (1 + ErrThreshold)
	res_loc_1d, = np.where(SSD<=thr) # location in 1d
	res_ssd_1d = SSD[res_loc_1d]  # ssd in those locations
	# pdb.set_trace()
	res_pixelvalues = patches_nonzero[res_loc_1d,win_size/2,win_size/2]
	return res_loc_1d,res_ssd_1d,res_pixelvalues

def gkern(size, sigma=3.0, center=None):
	""" Make a square gaussian kernel.

    size is the length of a side of the square
    sigma is standard deviation
    can be thought of as an effective radius.
    """

	x = np.arange(0, size, 1, float)
	y = x[:, np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]
	raw_res = 1/(2*np.pi*sigma**2)*np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2*sigma ** 2))
	return raw_res/raw_res.sum()

def RandomPick(BMs_list,BMs_ssd,BMs_pixel):
	## return the location,ssd and pixel value of the best match randomly from the candidates lists
	rand_idx = np.random.randint(0, len(BMs_list))
	return BMs_list[rand_idx],BMs_ssd[rand_idx],BMs_pixel[rand_idx]

## 2.2.2 image inpainting
filenames2 = ['test_im1','test_im2']
winsizes = [5,9,11]
for fn2 in filenames2:
    for ws in winsizes:
        inpainting(fn2, '.bmp', ws)