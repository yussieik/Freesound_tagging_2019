# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:47:49 2019

@author: YUSS
"""

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Bidirectional, CuDNNLSTM, ELU, GlobalAveragePooling2D
from keras.layers import  Conv2D, MaxPooling2D, concatenate, add, GlobalMaxPooling2D, GaussianNoise, Concatenate
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras import metrics
from keras.regularizers import l2, l1
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score as lrap
from keras.optimizers import Adam, Nadam
import numpy as np
import cv2
import pandas as pd
import keras
from keras import losses
import tensorflow as tf
#from tensorflow.keras.layers import Attention
import AttentionLSTM as Attention


"""Load onehot encoded dataset"""
one_hot = pd.read_csv('./one_hot_classes.csv')
one_hot.set_index(['Unnamed: 0'], inplace = True)
"""Separate only the one labeled for sport"""
one_label = one_hot.loc[one_hot.sum(axis=1) == 1]
#sums = one_label.sum(axis=1)
"""Load the mfccs"""
mfccs = np.load('./mfccs.npy')
mfccs = mfccs[()]
"""Prepare X and labels"""
def data_for_net(data,labels,model='cnn'):
    x = []
    y = []
    for key in labels.index.tolist():
        if(key in data.keys()):
            x.append(data[key])
            y.append(labels.loc[key].values)
    x = np.array(x)
    y = np.array(y)
    if model.lower()=='cnn':
        x_n = x.reshape(x.shape[0], x.shape[1], x.shape[2],1)
    elif model.lower()=='rnn':
        x_n = x.reshape(x.shape[0], x.shape[1], x.shape[2])
    return x_n,y

x_n,y=data_for_net(mfccs,one_hot, model='cnn')
"""
x = []
y_1 = []
for key in one_hot.index.tolist():
    if(key in mfccs.keys()):
        x.append(mfccs[key])
        y_1.append(one_hot.loc[key].values)
x = np.array(x)
y_1 = np.array(y_1)
x_1 = x.reshape(x.shape[0], x.shape[1], x.shape[2],1)
"""
"""Normalize"""
def normalize(x):
    _min, _max = float('inf'), -float('inf')
    _min = min(np.amin(x), _min)
    _max = max(np.amax(x), _max)
    X = (x - _min) / (_max - _min)
    return X

"""Label smoothing"""
def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y
y = smooth_labels(y.astype('float64'), .1)

"""Train test split"""
train_x, test_x, train_y, test_y = train_test_split(x_n, y, test_size = 0.15, random_state = 42)

"""Check distributions"""
dist_train = train_y.sum(axis = 0)/train_y.shape[0] * 100
dist_test = test_y.sum(axis = 0)/test_y.shape[0] * 100
#train_x = normalize(train_x)
#test_x = normalize(test_x)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""Build RCNN"""
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

        conv1 = Conv2D(out_num_filters, (filtersize, filtersize), border_mode='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))
        stack1 = conv1(l)
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)

        conv2 = Conv2D(out_num_filters, (filtersize, filtersize), strides=(1, 1), border_mode='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))
        stack4 = conv2(stack3)
        stack5 = add([stack1, stack4])
        batch_norm1 = BatchNormalization()(stack5)
        stack6 = PReLU()(batch_norm1)

        conv3 = Conv2D(out_num_filters, (filtersize, filtersize), strides=(1, 1), border_mode='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))
        stack7 = conv3(stack6)
        stack7 = Dropout(0.2)(stack7)

        stack8 = add([stack1, stack7])
        stack9 = BatchNormalization()(stack8)
        stack10 = PReLU()(stack9)

        conv4 = Conv2D(out_num_filters, (filtersize, filtersize), strides=(1, 1), border_mode='same', kernel_initializer = keras.initializers.glorot_normal(seed=None))
        stack11 = conv4(stack10)
#        stack12 = add([stack1, stack11])
        stack13 = BatchNormalization()(stack11)
        stack14 = PReLU()(stack13)


#        if pool:
        stack15 = MaxPooling2D((2, 2), border_mode='same')(stack14)
#        stack14 = Dropout(0.2)(stack13)
#        else:
#            stack15 = Dropout(0.1)(stack15)

        return stack15

    #Build Network
    input_img = Input(shape=(shape1, shape2, nbChannels))
    conv_l = Conv2D(nbFilters, (filtersize, filtersize), strides=(1, 1), border_mode='same', activation='relu',  kernel_initializer = keras.initializers.glorot_normal(seed=None))
    l = conv_l(input_img)
#    glob_pool = GlobalAveragePooling2D()(l)

    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=True)
        else:
            l = RCL_block(conv_l, l, pool=True)

    out = Flatten()(l)

    l_out = Dense(240, activation = 'relu', kernel_initializer = keras.initializers.glorot_normal(seed=None))(out)
    l_out = Dense(160, activation = 'relu', kernel_initializer = keras.initializers.glorot_normal(seed=None))(l_out)
    l_out = Dense(nbClasses, activation = 'sigmoid', kernel_initializer = keras.initializers.glorot_normal(seed=None))(l_out)

    model = Model(input = input_img, output = l_out)
    model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['acc'])

    model.summary()
    return model
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filepath="./weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
my_model = makeModel(1, 13, 80, 80)
my_model.fit(train_x, train_y, epochs=50, batch_size = 32, validation_split = 0.2, callbacks=callbacks_list)

#my_model.load_weights("./weights.best.hdf5")
prediction = my_model.predict(train_x)
preds = np.argmax(prediction, axis = 1)
tr_preds = np.argmax(train_y, axis = 1)
s = sum(preds == tr_preds)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def CNN(shape1, shape2, shape3, nbClasses):
    input_shape = (shape1, shape2, shape3)
#    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    n_classes = nbClasses
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    model.summary()
    return model

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = CNN(13, 80, 1, 80)
filepath="./weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(train_x, train_y, epochs=50, batch_size = 32, validation_split = 0.2, callbacks=callbacks_list)


prediction = my_model.predict(train_x)
preds = np.argmax(prediction, axis = 1)
tr_preds = np.argmax(train_y, axis = 1)
s = sum(preds == tr_preds)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=1, nbFilters=8, filtersize = 3):
	model = BuildCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):

    def RCL_block(l_settings, l, pool=True, increase_dim=True):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters


        conv1 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal', activation = 'relu')(l)
        drop1 = Dropout(0.2)(conv1)
#        batch_norm1 = BatchNormalization()(conv1)
#        max_pool1 = MaxPooling2D((2, 2), border_mode='same')(conv1)

        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal', activation = 'relu')(drop1)
        max_pool2 = MaxPooling2D((2, 2), border_mode='same')(conv2)

#
        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal', activation = 'relu')(max_pool2)
        max_pool3 = MaxPooling2D((2, 2), border_mode='same')(conv3)

        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal', activation = 'relu')(max_pool3)

        return conv4

    #Build Network
    input_img = Input(shape=(shape1, shape2, nbChannels))
    conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    l = conv_l(input_img)

    for n in range(nbRCL):
        if n % 2 == 0:
            l = RCL_block(conv_l, l, pool=True)
        else:
            l = RCL_block(conv_l, l, pool=True)

    out = Flatten()(l)
    l_out = Dense(180, activation = 'relu', init = 'he_normal')(out)
#    drop1 = Dropout(0.1)(l_out)
    l_out = Dense(nbClasses, activation = 'sigmoid')(l_out)

    model = Model(input = input_img, output = l_out)
    model.compile(optimizer = Adam(lr = 1e-2), loss = 'binary_crossentropy', metrics = ['mse'])

    model.summary()
    return model


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
filepath="./weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
my_model = makeModel(1, 13, 80, 80)
my_model.fit(train_x, train_y, epochs=50, batch_size = 32, validation_split = 0.2, callbacks=callbacks_list)


prediction = my_model.predict(test_x)
preds = np.argmax(prediction, axis = 1)
tr_preds = np.argmax(test_y, axis = 1)
s = sum(preds == tr_preds)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def BuildBidLSTM(shape1, shape2, nbClasses):

    input_shape = (shape1, shape2)
    optimizer = Adam(0.005, beta_1=0.1, beta_2=0.001, amsgrad=True)
    n_classes = nbClasses

    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True), input_shape=input_shape))
#    model.add(Attention(13))
    model.add(Dropout(0.2))
    model.add(Dense(400))
    model.add(ELU())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    model.summary()
    return model


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
my_bidlstm = BuildBidLSTM(13, 80, 80)

filepath="./weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
my_bidlstm.fit(train_x, train_y, epochs = 100, batch_size = 1024, validation_split = 0.2, callbacks=callbacks_list)

prediction = my_bidlstm.predict(test_x)
preds = np.argmax(prediction, axis = 1)
tr_preds = np.argmax(test_y, axis = 1)
s = sum(preds == tr_preds)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""







my_model.save_weights('./my_weights_l_0.0770.hdf5')
my_model.load_weights("./weights_sigmoid/weights.best.hdf5")




def calculate_overall_lwlrap_sklearn(truth, scores):
    """Calculate the overall lwlrap using sklearn.metrics.lrap."""
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = lrap(
        truth[nonzero_weight_sample_indices, :] > 0,
        scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap



#img = cv2.imread('crop/000b6cfb.png',cv2.IMREAD_GRAYSCALE).reshape(1,45,336,1)


#problematic = ['f76181c4','77b925c2','6a1f682a','c7db12aa','7752cc8a']
#imgs = []
#labels = []

#for root, dirs, files in os.walk('train_final/'):
#    for file in files:
##        if(file[:-4] in problematic):
##           continue
##        else:
#        img = cv2.imread(os.path.join(root,file),cv2.IMREAD_GRAYSCALE)
#        imgs.append(img)
#        labels.append(one_hot.loc[file[:-3]+'wav', :])


x = (np.array(imgs))/255.0
y = np.array(labels)
x = x.reshape(24760, 1, 45, 336)

np.save('./X_inputs',x)
np.save('./Y_labels',y)

mfccs = np.load('./mfccs.npy')
mfccs = mfccs[()]
y = np.load('./Y_labels.npy')




_min, _max = float('inf'), -float('inf')
_min = min(np.amin(x_n), _min)
_max = max(np.amax(x_n), _max)

X = (x_n - _min) / (_max - _min)
















