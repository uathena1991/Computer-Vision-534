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
from sklearn import feature_extraction as sfe
plt.interactive(False)

sys.path.append("/Users/xiaolihe/Documents/Computer-Vision-534/hw2")


# In[2]:


# In[ ]:

## growimage 
def synthesize(filename, win_size, shape_newimage, shape_seed=(15, 15)):
	# read sample image
	img_sample0 = io.imread(os.getcwd() + '/Assignment-II-images/' + filename)
	img_sample = img_as_float(img_sample0)
	# initialize img2bfilled with a random sample from sample image (size: shape_seed)
	randrow = int(np.random.rand() * (img_sample.shape[0] - shape_seed[0]))
	randcol = int(np.random.rand() * (img_sample.shape[1] - shape_seed[1]))
	seed_img = img_sample[randrow: randrow + shape_seed[0], randcol: randcol + shape_seed[1]]
	# (left-up corner is the center of img2bfilled)
	#     range_in_img2bfilled = (shape_newimage[0]/2 : (shape_newimage[0]/2 + shape_seed[0]),
	#                 shape_newimage[1]/2 : (shape_newimage[1]/2 + shape_seed[1]))
	img2bfilled = np.zeros(shape_newimage)
	img2bfilled[shape_newimage[0] / 2: (shape_newimage[0] / 2 + shape_seed[0]),
	shape_newimage[1] / 2: (shape_newimage[1] / 2 + shape_seed[1])] = seed_img
	filled_status = np.ones(shape_newimage) == False
	filled_status[shape_newimage[0] / 2: (shape_newimage[0] / 2 + shape_seed[0]),
	shape_newimage[1] / 2: (shape_newimage[1] / 2 + shape_seed[1])] = True

	plt.imshow(img2bfilled,cmap = 'gray')
	plt.title('Initial patch')
	plt.show()

	# Synthesis
	img_new = GrowImage(img_sample, img2bfilled, filled_status, win_size)

	# show image
	plt.imshow(img_sample,'gray')
	plt.title('sample image')
	plt.show()
	plt.imshow(img_new,'gray')
	plt.title('Sythesized image')
	plt.show()


def GrowImage(sample_img, img2bfilled, filled_status, win_size):
	MaxErrThreshold = 0.3
	ct = 0
	while not filled_status.all():
		print 'filled pixels is ', ct
		progress = 0
		row_idxs, col_idxs = GetUnfilledNeighbors(filled_status, win_size)

		for ridx, colidx in zip(row_idxs, col_idxs):
			template, validmask = GetNeighborhoodWindow(ridx, colidx, img2bfilled, filled_status, win_size)
			BMs_list,BMs_ssd = FindMatches(template, sample_img, validmask, win_size)
			BM_loc,BM_ssd = RandomPick(BMs_list, BMs_ssd)
			# find its 2d location index
			samp_img_r,samp_img_c = sample_img.shape
			BM_row, BM_col = BM_loc/(samp_img_c - win_size + 1),BM_loc%(samp_img_c - win_size + 1)
			# filled if BM_ssd is smaller than MaxErrThreshold
			if BM_ssd < MaxErrThreshold:
				img2bfilled[ridx, colidx] = sample_img[BM_row,BM_col]
				progress = 1
				filled_status[ridx, colidx] = True
				ct+=1
		# pdb.set_trace()
		# plt.imshow(img2bfilled, cmap='gray')
		# plt.show()
		if progress == 0:
			MaxErrThreshold *= 1.1
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
		template[minr_img2f-ridx+half_win_size:maxr_img2f-ridx+half_win_size,
		minc_img2f-colidx+half_win_size:maxc_img2f-colidx+half_win_size] = img2bfilled[minr_img2f:maxr_img2f, minc_img2f:maxc_img2f]
		template_filled_status[minr_img2f-ridx+half_win_size:maxr_img2f-ridx+half_win_size,
		minc_img2f-colidx+half_win_size:maxc_img2f-colidx+half_win_size] = filled_status[minr_img2f:maxr_img2f, minc_img2f:maxc_img2f]
	# for r in range(win_size):
	# 	for c in range(win_size):
	# 		if row_range[r] in range(img2bfilled.shape[0]) and col_range[c] in range(img2bfilled.shape[1]):
	# 			template[r, c] = img2bfilled[row_range[r], col_range[c]]
	# 			template_filled_status[r, c] = filled_status[row_range[r], col_range[c]]
	pdb.set_trace()
	return template, template_filled_status


## Find Matches, return locations, corresponding ssd (in 1d)
def FindMatches(template, sample_img, ValidMask, win_size):
	Sigma = win_size / 6.4
	ErrThreshold = 0.1
	GaussMask = gkern(win_size, Sigma)
	mask_raw = np.multiply(ValidMask, GaussMask)
	mask_normalized = mask_raw / mask_raw.sum()
	patches_list = sfe.image.extract_patches_2d(sample_img, (win_size,win_size))# array, shape = (n_patches, patch_heidth)
	dist_filter = (patches_list - template)**2*mask_normalized
	SSD = np.asarray([d.sum() for d in dist_filter])
	thr = SSD.min() * (1 + ErrThreshold)
	res_loc_1d, = np.where(SSD<=thr) # location in 1d
	res_ssd_1d = SSD[res_loc_1d]  # ssd in those locations
	return res_loc_1d,res_ssd_1d

	#
	# for i in range(win_size / 2, sample_r - win_size / 2 - 1):  # for each patch in sample_img
	# 	for j in range(win_size / 2, sample_c - win_size / 2 - 1):
	# 		#             pdb.set_trace()
	# 		curr_patch = sample_img[i - win_size / 2:i + win_size / 2+1, j - win_size / 2:j + win_size / 2 +1]
	# 		dist = (template - curr_patch) ** 2
	# 		SSD[i, j] += np.multiply(dist, mask_normalized).sum()
	# 	#     pdb.set_trace()
	# for i in range(SSD.shape[0]):
	# 	for j in range(SSD.shape[1]):
	# 		if SSD[i, j] <= thr:
	# 			PixelList.append([(i, j), SSD[i, j]])
	# return PixelList


def gkern(size, fwhm=3.0, center=None):
	""" Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

	x = np.arange(0, size, 1, float)
	y = x[:, np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]

	return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


def RandomPick(BMs_list,BMs_ssd):
	## return the location of the best match randomly from the candidates lists
	rand_idx = np.random.randint(0, len(BMs_list))
	return BMs_list[rand_idx],BMs_ssd[rand_idx]


synthesize('T1.gif', 5, [100, 100])
