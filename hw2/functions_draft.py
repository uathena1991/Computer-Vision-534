import numpy as np
import scipy
from skimage import morphology
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/Users/xiaolihe/Documents/Computer-Vision-534/hw2")
import pdb

def main(filename,winsize,shape_newimage,shape_seed = (3,3)):
    # read sample image
    img_sample = io.imread(os.getcwd()+'/Documents/Computer-Vision-534/hw2/Assignment-II-images/'+filename)
    plt.imshow(img_sample);
    plt.show()
#     subs_image = morphology.dilation(img_sample) - img_sample
#     plt.imshow(subs_image);
#     find(subs_image)

    # initialize img2bfilled with a random sample from sample image (size: shape_seed)
    randrow = round(rand()*(img_sample.shape[0]-shape_seed[0]))
    randcol = round(rand()*(img_sample.shape[1]-shape_seed[1]))
    seed_img = img_sample[randrow : randrow + shape_seed[0], randcol : randcol + shape_seed[1]]
    #(left-up corner is the center of img2bfilled)
    range_in_img2bfilled = (np.array(shape_newimage[0])/2 : np.array(shape_newimage[0])/2 + shape_seed[0],
                np.array(shape_newimage[1])/2 : np.array(shape_newimage[1])/2 + shape_seed[1])
    img2bfilled = np.zeros(shape_newimage)
    img2bfilled[range_in_img2bfilled] = seed_img
    filled_status = np.ones(shape_newimage) == False
    filled_status[range_in_img2bfilled] = True

    # Synthesis
    img_new = GrowImage(img_sample,img2bfilled,winsize)


def GrowImage(SampleImage, Image2bfilled, filled_status, WindowSize):
    while not filled_status.all():
        progress = 0
        row_idxs,col_idxs = GetUnfilledNeighbors(Image2bfilled, WindowSize)
        for ridx,colidx in zip(row_idxs,col_idxs):
            Template = GetNeighborhoodWindow(ridx,colidx,Image2bfilled,WindowSize)
            BestMatches = FindMatches(Template, SampleImage)
            BestMatch = RandomPick(BestMatches)
            if (BestMatch.error < MaxErrThreshold):
                Pixel.value = BestMatch.value
                progress = 1
                [filled[r,c] = 1 for r,c in zip(row_idxs,col_idxs)]
        if progress == 0:
            MaxErrThreshold = MaxErrThreshold * 1.1
    return Image

## GetUnfilledNeighbors
def GetUnfilledNeighbors(Image2bfilled, winsize):
    subs_image = morphology.dilation(Image2bfilled) - Image2bfilled
    [row_idxs, col_idxs] = np.where(subs_image!=0)
    # random permutation
    randidx = np.random.permutation(len(row_idxs))
    row_idxs = [row_idxs[idx] for idx in randidx]
    col_idxs = [col_idxs[idx] for idx in randidx]
    # sorted by decreasing number of filled neighbor pixels
    filledsum  = scipy.ndimage.generic_filter(Image2bfilled, np.sum, winsize)
    pdb.set_trace()
    filledsum_bs = [filledsum[x,y] for x,y in zip(row_idxs,col_idxs)]
    pdb.set_trace()
    sorted_idx = np.array(filledsum_bs).argsort()[::-1]
    res_row = [row_idxs[i] for i in sorted_idx]
    res_col = [col_idxs[i] for i in sorted_idx]
    return res_row, res_col

## GetNeighborhoodWindow
def GetNeighborhoodWindow(ridx,colidx,Image2bfilled,winsize):
    half_winsize = (winsize-1)/2
    row_range = range(ridx - half_winsize,ridx + half_winsize + 1)
    col_range = range(colidx - half_winsize,colidx + half_winsize + 1)
    template = np.zeros((winsize,winsize))
    for r in range(winsize):
        for c in range(winsize):
            if row_range[r] in range(Image2bfilled.shape[0]) and col_range[c] in range(Image2bfilled.shape[1]):
                template[r,c] = Image2bfilled[row_range[r],col_range[c]]
    return template


## Find Matches
def FindMatches(Template,SampleImage,filled_status,winsize):
    Sigma = winsize/6.4
    ErrThreshold = 0.1
    MaxErrThreshold = 0.3

    ValidMask = filled_status
    GaussMask = gaussian_kernal(winsize,Sigma)
    TotWeight = sum i,j GaussiMask(i,j)*ValidMask[i,j]
    for i,j do
    for ii,jj do
       dist = (Template(ii,jj)-SampleImage(i-ii,j-jj))^2
       SSD(i,j) = SSD(i,j) + dist*ValidMask(ii,jj)*GaussMask(ii,jj)
    end
    SSD(i,j) = SSD(i,j) / TotWeight
  end
  PixelList = all pixels (i,j) where SSD(i,j) <= min(SSD)*(1+ErrThreshold)
  return PixelList
end

def gaussian_kernel(size,sigma):
    size = int(size)
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2/float(2*sigma**2)+y**2/float(2*sigma**2)))
    return g / g.sum()

