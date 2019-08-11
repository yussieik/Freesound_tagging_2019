# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:53:47 2019

@author: YUSS
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


problematic = {}
img = cv2.imread('mfccs/000b6cfb.png',cv2.IMREAD_GRAYSCALE)

def crop_im(loadurl, saveurl):
   for root, dirs, files in os.walk(loadurl):
       for file in files:
           img = cv2.imread(os.path.join(root,file),cv2.IMREAD_GRAYSCALE)
           if(np.all(img[120,:] == 255)):
               img = img[121:166,54:390]
               if(np.all(img[22,4:-4] == 255)):
                   problematic[file] = img
               else:    
                   cv2.imwrite(os.path.join(saveurl,file), img)
           else:
               continue
           
crop_im('train_mfccs/', 'train_final/')           

plt.imshow(n_img)
plt.show()
