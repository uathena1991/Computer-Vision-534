import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, filters, morphology
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
import pickle


def cal_knn(class_labels,num_k,x):
    res = [class_labels[x[i]] for i in range(num_k)]
    return max(set(res),key = res.count)    

def Recognition(filename,features_train, features_mean,features_std,class_labels,thr_binary,threshold_size,disp_idx,improve_idx,num_k = 10):
    features_test = []
    ## read image and binarization
    img = io.imread(os.getcwd()+'/Documents/Computer-Vision-534/hw1/H1-16images/' + filename + '.bmp')
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
        # thr_binary = 200
        img_binary = (img < thr_binary).astype(np.double)
        # img_binary = np.logical_not(filters.threshold_adaptive(img,151,method='gaussian')).astype(np.double)
        
        ### dilation and erosion
        img_binary = morphology.binary_closing(img_binary).astype(np.double)   
        img_binary = morphology.skeletonize(img_binary).astype(np.double)
        img_binary = morphology.binary_dilation(img_binary).astype(np.double)    
        
    else:
        img_binary = (img < thr_binary).astype(np.double)
            
    
    ## Extracting Characters and Their Features
    img_label = label(img_binary, neighbors = 8, background=0)     ### labeling
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
    io.imshow(img_binary)
    ax = plt.gca()
    regions = regionprops(img_label,cache = True)
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
            # if disp_idx:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
    print 'Number of labeling components: %d' %len(features_test)
    # if disp_idx:
    ax.set_title('Bounding Boxes')
    plt.savefig('%s_Bounding_Boxes_improve%d' %(filename,improve_idx))
    io.show()
    ## calculating a distance matrix
    D = cdist(features_test,features_train)
    # print np.shape(features_test),np.shape(features_train)
    # if disp_idx:
    io.imshow(D,aspect='auto', vmin = D.mean()-3*D.std(),vmax = D.mean()+3*D.std()) 
    plt.title('Test data: Distance Matrix') 
    plt.savefig('%s_Distance_Matrix_improve%d' %(filename,improve_idx))
    io.show()
    D_index = np.argsort(D, axis=1)
    if improve_idx:
        # Ypred = [train.class_labels[x[0]] for x in D_index]
        Ypred = [cal_knn(class_labels,num_k,x) for x in D_index]
        # train.cal_knn(class_labels,num_k,x)
    else:
        Ypred = [class_labels[x[0]] for x in D_index]
    # print np.shape(D)
    return Ypred, features_test,regions_return



def evaluate_ORC(gtruth,regions,Ypred_test):
    import matplotlib.patches as patches
    # load groundtruth
    pkl_file = open(os.getcwd() + '/Documents/Computer-Vision-534/hw1/' + gtruth + '.pkl', 'rb')
    mydict = pickle.load(pkl_file) 
    pkl_file.close()
    classes = mydict['classes']  # N*1
    locations = mydict['locations'] # N*2
    # print classes,locations
    # function: in polygon
    def inpolygon(polygons,p):
        for i in range(len(polygons)):
            minr, minc, maxr, maxc = polygons[i]
            if (minr-p[1])*(maxr-p[1]) <= 0 and (minc-p[0])*(maxc-p[0]) <= 0:
                return i
        return -1
    # cal recognition rate
    correct_ct = 0
    ## visualize
    fig = plt.figure(figsize = [10,10])
    ax = fig.add_subplot(111)
    for i in range(len(locations)):
        rc_gt = locations[i]
        idx = inpolygon(regions,rc_gt)
        ## visualize
        ax.plot(rc_gt[0],rc_gt[1],'k.')
        ax.annotate(classes[i],xy = tuple(locations[i]+np.array([10,0])))
        if idx >= 0:
            ## visualize
            ax.add_patch(patches.Rectangle((regions[idx][1],regions[idx][0]),regions[idx][3]-regions[idx][1],
                                           regions[idx][2]-regions[idx][0],fill=False))
            if Ypred_test[idx]  == classes[i]:
                ax.plot(rc_gt[0],rc_gt[1],'r.')
                ax.annotate(Ypred_test[idx],xy = tuple(locations[i]+np.array([30,0])),color = 'red')
                correct_ct += 1
            else:
                ax.annotate(Ypred_test[idx],xy = tuple(locations[i]+np.array([30,0])),color = 'black')
    plt.gca().invert_yaxis()
    ax.set_title('Prediction vs. ground truth')
    # save it to files
    plt.savefig('%s_Prediction_vs_ground_truth_improve.png' %gtruth)
    plt.show()
    return float(correct_ct)/len(locations)
    