import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.misc import face
from scipy.linalg import norm
from skimage import measure
import networkx as nx
from sklearn import cluster
import threading
import Queue
import time
from multiprocessing import cpu_count
import cv2
import numpy as np
import scipy
from skimage import morphology, io, img_as_float, color
import matplotlib.pyplot as plt
import os
import sys
import pdb
import time
from sklearn import feature_extraction as sfe

import matplotlib.pyplot as plt
import numpy as np
filename = 'test_im3.jpg'
img_sample0 = io.imread(os.getcwd() + '/Assignment-II-images/' + filename)
img_sample0 = color.rgb2gray(img_sample0) # rgb to gray level
img_sample = img_as_float(img_sample0) # convert image to 0-1 values

img = img_sample
# plt.imshow(img)
# plt.axis("off")
# plt.show()
boxes = []
counter = 0
while True:
    def on_mouse(event, x, y, flags, params):
        global boxes, img, counter

        if event == cv2.EVENT_LBUTTONDOWN:
            #print 'Start Mouse Position: '+str(x)+', '+str(y)
            boxes.append((x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            #print 'End Mouse Position: '+str(x)+', '+str(y)
            ebox = (x, y)
            boxes.append((x, y))
            cv2.rectangle(img, boxes[0], boxes[1], [0, 0, 0], thickness=-1)
            boxes = []
            counter += 1

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', on_mouse, 0)
    cv2.imshow('Image',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cv2.imwrite("../resources/created_.jpg", img)
        cv2.imwrite("result_jerusalem.jpg", img)
        break