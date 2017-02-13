import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pickle

def feature_database(char,thr_binary,threshold_size,disp_idx):
    Features=[]
    img = io.imread(os.getcwd()+'/Documents/Computer-Vision-534/hw1/H1-16images/'+char+'.bmp')
    hist = exposure.histogram(img)
    img_binary = (img < thr_binary).astype(np.double)
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
            Features.append({char:hu})
            if disp_idx:
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    if disp_idx:
        ax.set_title('Bounding Boxes')
        io.show()
    return Features





    # ### Building Character Features Database for Recognition
    # 

    # In[33]:

    # Creating a File to Process Each Image