#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from tensorflow.python.lib.io import file_io
import pandas as pd
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.utils import shuffle

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.preprocessing import image

#from sklearn.metrics import log_loss
#from sklearn.model_selection import StratifiedKFold


SEED=np.random.randint(1337)
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)


class BalanceDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        n_each = batch_size // n_labs   
        
        index_list = []
        for i1 in range(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in range(n_labs):
            np.random.shuffle(index_list[i1])
        
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            for i1 in range(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_each, len_list[i1])]
                batch_x.append(x[batch_idx])
                batch_y.append(y[batch_idx])
                pointer_list[i1] += n_each
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

class RatioDataGenerator(object):
    def __init__(self, batch_size, type, te_max_iter=100, verbose=1):
        assert type in ['train', 'test']
        self._batch_size_ = batch_size
        self._type_ = type
        self._te_max_iter_ = te_max_iter
        self._verbose_ = verbose
            
    def _get_lb_list(self, n_samples_list):
        lb_list = []
        for idx in range(len(n_samples_list)):
            n_samples = n_samples_list[idx]
            if n_samples < 1000:
                lb_list += [idx]
            elif n_samples < 2000:
                lb_list += [idx] * 2
            elif n_samples < 3000:
                lb_list += [idx] * 3
            elif n_samples < 4000:
                lb_list += [idx] * 4
            else:
                lb_list += [idx] * 5
        return lb_list
        
    def generate(self, xs, ys):
        batch_size = self._batch_size_
        x = xs[0]
        y = ys[0]
        (n_samples, n_labs) = y.shape
        
        n_samples_list = np.sum(y, axis=0)
        lb_list = self._get_lb_list(n_samples_list)
        
        if self._verbose_ == 1:
            print("n_samples_list: %s" % (n_samples_list,))
            print("lb_list: %s" % (lb_list,))
            print("len(lb_list): %d" % len(lb_list))
        
        index_list = []
        for i1 in range(n_labs):
            index_list.append(np.where(y[:, i1] == 1)[0])
            
        for i1 in range(n_labs):
            np.random.shuffle(index_list[i1])
        
        queue = []
        pointer_list = [0] * n_labs
        len_list = [len(e) for e in index_list]
        iter = 0
        while True:
            if (self._type_) == 'test' and (iter == self._te_max_iter_):
                break
            iter += 1
            batch_x = []
            batch_y = []
            
            while len(queue) < batch_size:
                random.shuffle(lb_list)
                queue += lb_list
                
            batch_idx = queue[0 : batch_size]
            queue[0 : batch_size] = []
            
            n_per_class_list = [batch_idx.count(idx) for idx in range(n_labs)]
            
            for i1 in range(n_labs):
                if pointer_list[i1] >= len_list[i1]:
                    pointer_list[i1] = 0
                    np.random.shuffle(index_list[i1])
                
                per_class_batch_idx = index_list[i1][pointer_list[i1] : min(pointer_list[i1] + n_per_class_list[i1], len_list[i1])]
                batch_x.append(x[per_class_batch_idx])
                batch_y.append(y[per_class_batch_idx])
                pointer_list[i1] += n_per_class_list[i1]
            batch_x = np.concatenate(batch_x, axis=0)
            batch_y = np.concatenate(batch_y, axis=0)
            yield batch_x, batch_y

			
			
#########
##SAMPLE PAIRING			

def train_generator(X_train, Y_train, transform = None):
    """Generator that returns transformed batches
        args: X_train: input images, batched (batch_size, W, H, C)
              Y_train: input labels
              transform: transformation to apply from the below dictionary
    """

#    transforms = {'translate discrete': iaa.Affine(translate_px={"x": iap.Choice([9, 0, -9]), "y": iap.Choice([9, 0, -9])}),
#                  'translate random': iaa.Affine(translate_px={"x": (-12, 12), "y": (-12, 12)}), 
#                  'rotate': iaa.Affine(rotate=(-45, 45)),
#                  'vertical flip': iaa.Flipud(0.5),
#                  'horizontal flip':iaa.Fliplr(0.5),
#                  'scale': iaa.Affine(scale=(0.95, 1.05)),
#                  'blur': iaa.GaussianBlur(sigma = 3.0),
#                  'combined': iaa.Sequential([ 
#                                        iaa.Affine(translate_px={"x": iap.Choice([9, 0, -9]), "y": iap.Choice([9, 0, -9])}),
#                                        iaa.Affine(scale=(0.95, 1.05))
#                                        ])}
    
    if transform:
        if transform in ['original']:
            """no further transformation"""
            pass
#        else:
#            seq = transforms[transform]
#            X_train = seq.augment_images(X_train)
            
    iter_ = image.ImageDataGenerator() 
    batch = iter_.flow(X_train, Y_train, batch_size = 32, seed = 1337) 
    while True:
        yield batch.next()

def merge_bands(band1, band2):
    b1 = np.array(band1).astype(np.float32)
    b2 = np.array(band2).astype(np.float32)
    return [np.stack([b1, b2], -1).reshape(75, 75, 2)]

def print_and_save_history(h):
    history_val_loss = []
    history_val_acc =  []
    hloss = zip(h.history['loss'], h.history['val_loss'], h.history['acc'], h.history['val_acc'])
    for idx, loss in enumerate(hloss):
        print('Step {} loss: {}, val_loss: {}'.format(idx, loss[0], loss[1]))
    history_val_loss.append(h.history['val_loss'])
    history_val_acc.append(h.history['val_acc'])
    return history_val_loss, history_val_acc
    
#for i in history:
#    val_loss, val_acc = print_and_save_history(i)

def load(train_file):
    with file_io.FileIO(train_file, mode='r') as stream:
        df = pd.read_json(stream).set_index('id')
        if 'bands' not in df.columns:
            df['bands'] = df.apply(lambda row: merge_bands(row['band_1'], row['band_2']), axis=1)
            df = df.drop(['band_1', 'band_2'], axis=1)
        return df

def get_callbacks(filepath, patience=10):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    #csv_logger = CSVLogger('C:/Users/Frank/Desktop/kaggle/statoil/statoil_keras_files/' + filepath + 'log.csv', append=True, separator=';')
    return [es] # , csv_logger

def SamplePairing(X_train, Y_train, N):
    """Randomly add two samples and concatenate them to X_train, Y_train
        args: X_train: original dataset 
            N number of samples per class to add
        
    """
    p_1 = np.random.rand()
    p = 1
    if p_1 > p:
        return X_train, Y_train
    else:
        high = X_train.shape[0]
        for i in range(N):
            """expand positive examples by N"""
            random1, random2 = np.random.randint(low = 0, high = high, size=2)
            new_sample = np.expand_dims((X_train[random1, :, :, :] + X_train[ random2, :, :, :])/2, axis = 0)
            X_train = np.concatenate((X_train, new_sample), axis = 0)
            Y_train = np.concatenate((Y_train, (Y_train[random1],)), axis = 0)
    
        X_train, Y_train = shuffle(X_train, Y_train, random_state = 0)
        # shuffle
        return X_train, Y_train


#########
##MIXUP

	
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        
#__init__函数是在创建一个类的实例，作用是初始化某个类的一个实例。
        
    def __call__(self):
        while True:                                              
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                
                yield X, y
                
   
    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
#        _, h, w, c = self.X_train.shape
        if self.alpha == 0:
            l = np.zeros(self.batch_size)
        else:
            l = np.random.beta(self.alpha, self.alpha, self.batch_size)       
#        print(l.shape)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)
        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
class MixupGenerator_prelab():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        
    def __call__(self):
        while True:                                              
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                
                yield X, y             
  
    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
#        l = np.ones(self.batch_size)
        l_y = np.ones(self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l_y.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

###sample pairing
class SPGenerator():
    def __init__(self, X_train, y_train, batch_size=32, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        
    def __call__(self):
        while True:                                              
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                
                yield X, y             
  
    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        l = 0.5*np.ones(self.batch_size)
        l_y = np.ones(self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l_y.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y

#########
##get_random_eraser	
		
		
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=64):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def mixup(data, one_hot_labels, alpha=0.3, debug=False):
    np.random.seed(42)

    batch_size = len(data)
    weights = np.random.beta(alpha, alpha, batch_size)
    index = np.random.permutation(batch_size)
    #shuffle 返回 None，这点尤其要注意，也就是说没有返回值，而 permutation 则返回打乱后的 array。
    x1, x2 = data, data[index]
    x = np.array([x1[i] * weights [i] + x2[i] * (1 - weights[i]) for i in range(len(weights))])
    y1 = np.array(one_hot_labels).astype(np.float)
    y2 = np.array(np.array(one_hot_labels)[index]).astype(np.float)
    y = np.array([y1[i] * weights[i] + y2[i] * (1 - weights[i]) for i in range(len(weights))])
    if debug:
        print('Mixup weights', weights)
    return x, y		

class extrapolationGenerator():
    def __init__(self, X_train, y_train, batch_size=32,  alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.alpha = alpha
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        
#__init__函数是在创建一个类的实例，作用是初始化某个类的一个实例。
        
    def __call__(self):
        while True:                                              
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))
            
            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)
                
                yield X, y
                

    
    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

        

    def __data_generation(self, batch_ids):
#        _, h, w, c = self.X_train.shape
        l = 0.5*np.ones(self.batch_size)
#        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        l_y = np.zeros(self.batch_size)
#        print(l.shape)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l_y.reshape(self.batch_size, 1)
#        y_l = l.reshape(self.batch_size, 1)
        
        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X2 * (1 + X_l) - X_l * X1 

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y2 * (1 + y_l) - y_l * y1 )
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y2 * (1 + y_l) - y_l * y1

        return X, y
