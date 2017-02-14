import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
from matplotlib import path
import pickle
import os
import sys
sys.path.append("/Users/xiaolihe/Documents/Computer-Vision-534/hw1")
import train
import test_xh
reload(train)
reload(test_xh)


# ### Calculate recognition accuracy and visualize it.
def evaluate_ORC(gtruth,regions,Ypred_test):
    import matplotlib.patches as patches
    # load groundtruth
    pkl_file = open(os.getcwd()+'/Documents/Computer-Vision-534/hw1/'+gtruth+'.pkl', 'rb')
    mydict = pickle.load(pkl_file) 
    pkl_file.close()
    classes = mydict['classes']  # N*1
    locations = mydict['locations'] # N*2
#     print classes,locations
    # function: in polygon
    def inpolygon(polygons,p):
        for i in range(len(polygons)):
            minr, minc, maxr, maxc = polygons[i]
            if (minr-p[1])*(maxr-p[1]) <= 0 and (minc-p[0])*(maxc-p[0]) <= 0:
#             if (minr-p[0])*(maxr-p[0]) <= 0 and (minc-p[1])*(maxc-p[1]) <= 0:
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
        ax.plot(rc_gt[0],rc_gt[1],'b.')
        ax.annotate(classes[i],xy = tuple(locations[i] + np.array([10,0])))
        if idx > 0:
            ## visualize
            ax.add_patch(patches.Rectangle((regions[idx][1], regions[idx][0]), regions[idx][3] - regions[idx][1],
                                           regions[idx][2] - regions[idx][0], fill=False))
            ax.annotate(Ypred_test[idx],xy = tuple(locations[i] + np.array([30,0])))
            if Ypred_test[idx]  == classes[i]:
                ax.plot(rc_gt[0],rc_gt[1],'r.')
                correct_ct += 1
    plt.gca().invert_yaxis()
    plt.show()
    return float(correct_ct)/len(locations)
    
    
def main(test_image_lists, test_gt_lists, display_idx = 1, threshold_binary = 200, threshold_size = 10):
    ### Training 
    characters = ['a','d','f','h','k','m','n','o','p','q','r','s','u','w','x','z']
    Features = []
    class_labels = []
    for c in characters:
        tmp_feature,tmp_class_label = train.feature_database(c,threshold_binary,threshold_size,display_idx)
        [Features.append(tf) for tf in tmp_feature]    
        [class_labels.append(tf) for tf in tmp_class_label]
    # Normalization
    mean_features =  np.asarray(np.mean(Features,0))
    std_features = np.asarray(np.std(Features,0))
    # print mean_features
    # print std_features
    normed_features = [(x-mean_features)/std_features for x in Features]
    ## check:
    # np.std(normed_features,0)
    # print np.shape(normed_features),np.shape(features)

    # ### Recognition on Training Data
    D = cdist(normed_features, normed_features)
    io.imshow(D,vmin = D.mean()-3*D.std(),vmax = D.mean()+3*D.std()) 
    plt.title('Distance Matrix') 
    io.show()
    D_index = np.argsort(D, axis=1)
    Ypred_train = [class_labels[x[1]] for x in D_index]
    ## confusion matrix
    confM = confusion_matrix(class_labels,Ypred_train)
    io.imshow(confM) 
    plt.title('Confusion Matrix') 
    io.show()

    # ### Testing (Recognition)
    Ypred_test = []
    features_test = []
    regions = []
    accuracy = []
    for test_image,test_gt in zip(test_image_lists,test_gt_lists):
        # print test_image,test_gt
        tmp_pred,tmp_features,tmp_regions = test_xh.Recognition(test_image, normed_features, mean_features, 
                                                               std_features, class_labels, threshold_binary, threshold_size,
                                                               display_idx)
        Ypred_test.append(tmp_pred)
        features_test.append(tmp_features)
        regions.append(tmp_regions)
        # ### Evaluate performance ( recognition rate)
        accuracy.append(evaluate_ORC(test_gt,regions[-1],Ypred_test[-1]))
    
    return accuracy,Ypred_test




