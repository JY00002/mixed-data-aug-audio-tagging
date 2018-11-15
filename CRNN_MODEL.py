#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras import backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D, Convolution1D,MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.core import   Lambda,Reshape
from keras.layers import  GRU, TimeDistributed, Bidirectional
from keras.layers.merge import Multiply



def CRNN_ATT(n_freq, n_time, weights= None, input_tensor=None, include_top=True):

    def outfunc(vects):
        cla, att = vects    # (N, n_time, n_out), (N, n_time, n_out)
        att = K.clip(att, 1e-7, 1.)
        out = K.sum(cla * att, axis=1) / K.sum(att, axis=1)     # (N, n_out)
        return out
    input_shape = (n_time, n_freq, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    channel_axis = 3
    freq_axis = 1
    num_classes = 8
    x = BatchNormalization(axis=freq_axis, name='bn_0_freq')(melgram_input)
    # Conv block 1
    x = Convolution2D(8, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool1')(x)
    x = Dropout(0.1, name='dropout1')(x)
    
    # Conv block 2
    x = Convolution2D(16, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool2')(x)
    x = Dropout(0.1, name='dropout2')(x)
    
    # Conv block 3
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool3')(x)
    x = Dropout(0.1, name='dropout3')(x)
    
    # Conv block 4
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool4')(x)
    x = Dropout(0.1, name='dropout4')(x)
    
    # Conv block 5
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool5')(x)
    x = Dropout(0.1, name='dropout5')(x)
    
    # Conv block 6
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv6')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn6')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool6')(x)
    x = Dropout(0.1, name='dropout6')(x)
    
    # Conv block 7
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv7')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn7')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(1, 2), name='pool7')(x)
    x = Dropout(0.1, name='dropout7')(x)
    
    # Gated BGRU
    x = Reshape((n_time, 64))(x)
    rnnout = Bidirectional(GRU(64, activation='linear', return_sequences=True))(x)
    rnnout_gate = Bidirectional(GRU(64, activation='sigmoid', return_sequences=True))(x)
    x1 = Multiply()([rnnout, rnnout_gate])
    x1 = Dropout(0.1)(x1)
    
    # Attention
    cla = TimeDistributed(Dense(num_classes, activation='sigmoid'), name='localization_layer')(x1)
    att = TimeDistributed(Dense(num_classes, activation='softmax'))(x1)
    out = Lambda(outfunc, output_shape=(num_classes,))([cla, att])
    # Create model
    model = Model(melgram_input, out)
    return model    
