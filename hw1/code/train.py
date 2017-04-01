import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, filters, morphology
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pickle

def feature_database(char,thr_binary,threshold_size,disp_idx,improve_idx):
    Features = []
    class_label = []
    img = io.imread(os.getcwd()+'/Documents/Computer-Vision-534/hw1/H1-16images/'+char+'.bmp')
    hist = exposure.histogram(img)
    
    if improve_idx:        
        ### equated contrast
        # img = exposure.equalize_adapthist(img)
        
        # smoothing
        # img = filters.gaussian(img, sigma = 1)
        # img = filters.laplace(img)
        img = filters.median(img,selem = morphology.disk(1))
                
        
        # automate threshold
        # thr_binary = filters.threshold_isodata(img)
        # thr_binary = filters.threshold_li(img)
        thr_binary = filters.threshold_otsu(img)
        # thr_binary = filters.threshold_yen(img)
        # print thr_binary
        # thr_binary = 200
        img_binary = (img < thr_binary).astype(np.double)
        # img_binary = np.logical_not(filters.threshold_adaptive(img,151,method='gaussian')).astype(np.double)
        
        ### dilation and erosion
        img_binary = morphology.binary_closing(img_binary).astype(np.double)  
        img_binary = morphology.skeletonize(img_binary).astype(np.double)
        img_binary = morphology.binary_dilation(img_binary).astype(np.double)    
    else:
        img_binary = (img < thr_binary).astype(np.double)
    
    img_label = label(img_binary, neighbors = 8, background=0)     ### labeling
    regions = regionprops(img_label,cache = True) 
    ### bounding boxes and Store features in Features
    if disp_idx: 
        io.imshow(img)
        plt.title('Original Image')
        io.show()
        plt.bar(hist[1], hist[0])
        plt.title('Histogram') 
        plt.show()
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
        io.imshow(img_label) 
        plt.title('Labeled Image') 
        io.show()
    io.imshow(img_binary)
    ax = plt.gca()
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc >= threshold_size and maxr - minr >= threshold_size:
            roi = img_binary[minr:maxr, minc:maxc] 
            m = moments(roi)
            cr = m[0, 1] / m[0, 0]
            cc = m[1, 0] / m[0, 0]
            mu = moments_central(roi, cr, cc) 
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            Features.append(hu)
            class_label.append(char)
            # if disp_idx:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    print 'Number of labeling components: %d' %len(Features)
    # if disp_idx:    
    ax.set_title('Bounding Boxes of %s' %char)
    # save it to files
    plt.savefig('training_images_%s_improve%d.png' %(char,improve_idx))
    io.show()
    return Features,class_label

def cal_knn(class_labels,num_k,x):
    res = [class_labels[x[i]] for i in range(1,num_k)]
    return max(set(res),key = res.count)    