import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure,filters
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pickle

def Recognition(filename,features_train, features_mean,features_std,class_labels,thr_binary,threshold_size,disp_idx):
    features_test = []
    ## read image and binarization
    img = io.imread(os.getcwd()+'/Documents/Computer-Vision-534/hw1/H1-16images/' + filename + '.bmp')
    hist = exposure.histogram(img)
    
    # thr_binary = filters.threshold_isodata(img)
    # thr_binary = filters.threshold_li(img)
    # thr_binary = filters.threshold_otsu(img)
    # thr_binary = filters.threshold_yen(img)
    img_binary = (img < thr_binary).astype(np.double)
    # print thr_binary
    
    # img_binary = np.logical_not(filters.threshold_adaptive(img,151,method='gaussian')).astype(np.double)
    # io.imshow(img_binary)
    # io.show()
    ## Extracting Characters and Their Features
    img_label = label(img_binary, background=0)     ### labeling
    ### bounding boxes and Store features in Features
    if disp_idx: 
        io.imshow(img)
        io.show()
        plt.title('Original Image')
        plt.bar(hist[1], hist[0])
        plt.title('Histogram') 
        plt.show()
        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()
        io.imshow(img_label) 
        plt.title('Labeled Image') 
        io.show()
        print 'Number of labeling components: %d' %np.amax(img_label)
        io.imshow(img_binary)
        ax = plt.gca()
    regions = regionprops(img_label)
    regions_return = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc >= threshold_size and maxr - minr >= threshold_size:
            regions_return.append([minr, minc, maxr, maxc])
            roi = img_binary[minr:maxr, minc:maxc] 
            m = moments(roi)
            cr = m[0, 1] / m[0, 0]
            cc = m[1, 0] / m[0, 0]
            mu = moments_central(roi, cr, cc) 
            nu = moments_normalized(mu)
            hu = moments_hu(nu)
            # normalization
            normed_hu = (hu - features_mean)/features_std
            features_test.append(normed_hu)            
            if disp_idx:
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    if disp_idx:
        ax.set_title('Bounding Boxes')
        io.show()
    ## calculating a distance matrix
    D = cdist(features_test,features_train)
    # print np.shape(features_test),np.shape(features_train)
    if disp_idx:
        io.imshow(D,aspect=10,vmin = D.mean()-3*D.std(),vmax = D.mean()+3*D.std()) 
        plt.title('Distance Matrix') 
        io.show()
    D_index = np.argsort(D, axis=1)
    Ypred = [class_labels[x[0]] for x in D_index]
    # print np.shape(D)
    return Ypred, features_test,regions_return