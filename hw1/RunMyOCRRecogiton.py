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

def RunMyOCRRecognition(test_image, test_image_gt, display_idx = 1, improve_idx = 1 ):
    """
    params: 
        test_image: file name of test_image
        test_image_gt: true labels of the test image
        display_idx: whether to display all figures
        improve_idx: whether to use all the improvements
    return:
        Ypred_test1: predicted labels
        features_test1:
        regions1: 
        accuracy: accuracy of predictions
    """
    characters = ['a','d','f','h','k','m','n','o','p','q','r','s','u','w','x','z']
    Features = []
    class_labels = []
    threshold_binary = 200 ## before any improvement
    threshold_size = 10
    num_k = 15 ## for knn
    # display_idx = 1
    # improve_idx = 1
    for c in characters:
        tmp_feature,tmp_class_label = train.feature_database(c,threshold_binary,threshold_size,display_idx,improve_idx)
        [Features.append(tf) for tf in tmp_feature]    
        [class_labels.append(tf) for tf in tmp_class_label]


    # ### Normalization
    # get features (exclude labels)
    mean_features =  np.asarray(np.mean(Features,0))
    std_features = np.asarray(np.std(Features,0))
    # print mean_features
    # print std_features
    normed_features = [(x-mean_features)/std_features for x in Features]


    ### Recognition on Training Data

    D = cdist(normed_features, normed_features)
    io.imshow(D,vmin = D.mean()-3*D.std(),vmax = D.mean()+3*D.std()) 
    plt.title('Distance Matrix(training set)') 
    io.show()
    D_index = np.argsort(D, axis=1)
    improve_idx = 1
    if improve_idx == 1:
        Ypred_train = [train.cal_knn(class_labels,num_k,x) for x in D_index]
    else:
        Ypred_train = [class_labels[x[1]] for x in D_index]
    ## confusion matrix
    confM = confusion_matrix(class_labels,Ypred_train)
    io.imshow(confM) 
    plt.title('Confusion Matrix (training set)') 
    io.show()


    # ### Testing (Recognition)
    Ypred_test1,features_test1,regions1 = test_xh.Recognition(test_image,normed_features, mean_features,std_features,class_labels,
                                                 threshold_binary,threshold_size,display_idx,improve_idx,num_k)
    for i in range(0,len(Ypred_test1),8):
        print Ypred_test1[i:i+8]


    # ### Evaluate performance ( recognition rate)
    accuracy = test_xh.evaluate_ORC(test_image_gt,regions1,Ypred_test1)
    return Ypred_test1, features_test1, regions1, accuracy



