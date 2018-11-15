#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
def CNN_model(agg_num,n_freq):

    input_shape = (agg_num,n_freq,1)       
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    
    model.add(Convolution2D(8, (1, 1), activation='relu'))
    model.add(Convolution2D(16, (2, 2), activation="relu"))
    model.add(Convolution2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(8, activation='sigmoid'))
    	
    return model