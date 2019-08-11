# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:47:49 2019

@author: YUSS
"""

"""Build IRCNN"""


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import  Convolution2D, MaxPooling2D, Input, concatenate, add
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import metrics
from sklearn.metrics import label_ranking_average_precision_score as lrap
from keras.optimizers import Adam
import numpy as np
import cv2
import pandas as pd
import os
from keras import losses
import tensorflow as tf

"""Build IRCNN"""


def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=5, nbFilters=16, filtersize = 3):
	model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):

    def RCL_block(l_settings, l, pool=True, increase_dim=True):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2

        else:
            out_num_filters = input_num_filters

		   
        conv1 = Convolution2D(out_num_filters, 1, 1, border_mode='same', init = 'he_normal')
        stack1 = conv1(l)   	
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)

        

        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        stack4 = conv2(stack3)
#        stack5 = add([stack1, stack4])
        stack5 = BatchNormalization()(stack4)
        stack6 = PReLU()(stack5)

    	

        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack7 = conv3(stack6)
#        stack9 = add([stack1, stack8])
        stack8 = BatchNormalization()(stack7)
        stack9 = PReLU()(stack8)    

        

        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack10 = conv4(stack9)
#        stack13 = add([stack1, stack12])
        stack11 = BatchNormalization()(stack10)
        stack12 = PReLU()(stack11)    

        
#        if pool:
        stack13 = MaxPooling2D((2, 2), border_mode='same')(stack12) 
        stack14 = Dropout(0.1)(stack13)
#        else:
#            stack15 = Dropout(0.1)(stack15)
            
        return stack14


    #Build Network
    input_img = Input(shape=(nbChannels, shape1, shape2))
    conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    l = conv_l(input_img)

    
    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=True)
        else:
            l = RCL_block(conv_l, l, pool=True)


    out = Flatten()(l)

    l_out = Dense(240, activation = 'relu', init = 'he_normal')(out)
    l_out = Dense(160, activation = 'relu', init = 'he_normal')(l_out)
    l_out = Dense(nbClasses, activation = 'sigmoid')(l_out)
    
    model = Model(input = input_img, output = l_out)
    model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['acc'])  
    
    model.summary()
    return model


my_model = makeModel(1, 45, 336, 80)

    




img = cv2.imread('crop/000b6cfb.png',cv2.IMREAD_GRAYSCALE).reshape(1,45,336,1)

one_hot = pd.read_csv('./one_hot_classes.csv')
one_hot.set_index(['Unnamed: 0'], inplace = True)

problematic = ['f76181c4','77b925c2','6a1f682a','c7db12aa','7752cc8a']
imgs = []
labels = []

for root, dirs, files in os.walk('train_final/'):
    for file in files:
#        if(file[:-4] in problematic):
#           continue
#        else:
        img = cv2.imread(os.path.join(root,file),cv2.IMREAD_GRAYSCALE)
        imgs.append(img)
        labels.append(one_hot.loc[file[:-3]+'wav', :])



x = (np.array(imgs))/255.0
y = np.array(labels)

x = x.reshape(24760, 1, 45, 336)

np.save('./X_inputs',x)
np.save('./Y_labels',y)

x = np.load('./X_inputs.npy')
y = np.load('./Y_labels.npy')

my_model.save_weights('./my_weights_l_0.0770.hdf5')  


filepath="./weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
my_model.fit(x, y, epochs=3000,batch_size = 32, validation_split = 0.2, callbacks=callbacks_list)


prediction = my_model.predict(x[100:200])
n_prediction = my_model.predict(x[100:200])

my_model.load_weights("./weights.best.hdf5")












