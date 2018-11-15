#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import pickle
import os
import wavio
import librosa
import config as cfg
import csv
from sklearn import metrics
import numpy


def compute_eer(result_filename, label, label_assignments):         
    results = []
    #with open(result_filename, 'rt') as f:
    with open(result_filename, 'r') as f:
        for row in csv.reader(f, delimiter=','):        #对于f中的每一列
            if len(row[1]) != 1 or not row[1].isalpha():    #Python isalpha() 方法检测字符串是否只由字母组成。
                raise ValueError('The label identfier "' + row[1] + '" in row ' + str(row) + ' is not valid.')
            if row[1] == label:
                results.append((row[0], row[1], float(row[2]))) #分别为文件名，标签及得分
                
    if len(numpy.unique([r[0] for r in results])) != len(results):
        raise ValueError('File ' + result_filename + ' contains duplicate score assignments.')
    if len(set([r[0] for r in results]).symmetric_difference(set(label_assignments.keys()))) != 0:
        raise ValueError('One-to-one mapping between files listed in ' + result_filename + ' and ground truth assignments for label ' + label + ' not satisfied.')
    
    y_true = numpy.array([label_assignments[row[0]] for row in results])    #键（文件名）
    y_score = numpy.array([row[2] for row in results])  #得分“Score”表示每个测试样本属于正样本的概率
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true,y_score,drop_intermediate=True)
    eps = 1E-6
    Points = [(0,0)]+list(zip(fpr, tpr))
    for i, point in enumerate(Points):
        if point[0]+eps >= 1-point[1]:  
            break
    P1 = Points[i-1]; P2 = Points[i]
    if abs(P2[0]-P1[0]) < eps:
        EER = P1[0]        
    else:        
        m = (P2[1]-P1[1]) / (P2[0]-P1[0])
        o = P1[1] - m * P1[0]
        EER = (1-o) / (1+m)        
    return EER
	
	

def mat_2d_to_3d(X, agg_num, hop):
    # pad to at least one block至少填充一个数据块
    len_X, n_in = X.shape     #.shape查看矩阵或者数组的维数len_X为x的行数
    if (len_X < agg_num):
        X = np.concatenate((X, np.zeros((agg_num-len_X, n_in))))
        
    # agg 2d to 3d
    len_X = len(X)
    i1 = 0
    X3d = []
    while (i1+agg_num <= len_X):
        X3d.append(X[i1:i1+agg_num])
        i1 += hop
    return np.array(X3d)

def tp_fn_fp_tn(p_y_pred, y_gt, thres):
    y_pred = np.zeros_like(p_y_pred)       
    #Return an array of zeros with the same shape and type as a given array.
    y_pred[ np.where(p_y_pred>thres) ] = 1.
    tp = np.sum(y_pred + y_gt > 1.5)
    fn = np.sum(y_gt - y_pred > 0.5)
    fp = np.sum(y_pred - y_gt > 0.5)
    tn = np.sum(y_pred + y_gt < 0.5)
    return tp, fn, fp, tn

def prec_recall_fvalue(p_y_pred, y_gt, thres):
    eps = 1e-10
    (tp, fn, fp, tn) = tp_fn_fp_tn(p_y_pred, y_gt, thres)
    prec = tp / max(float(tp + fp), eps)    
    recall = tp / max(float(tp + fn), eps)  
    fvalue = 2 * (prec * recall) / max(float(prec + recall), eps)
    return prec, recall, fvalue

### readwav
def readwav(path):
    Struct = wavio.read(path)
    #读取一个WAV文件并返回一个保存采样率，采样宽度（以字节为单位）和包含数据的numpy数组的对象。
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)#np.power(a,b)求a的b次方
#    print(Struct.sampwidth)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    #wav_fd，fe_fd分别为波形和特征文件路径
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    #listdir返回指定路径下的文件和文件夹列表；endswith如果字符串含有指定的后缀返回True，否则返回False。
    names = sorted(names)
    for na in names:
        print(na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        print('wav shape:', wav.shape)
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
        assert fs==cfg.fs#assert在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=cfg.win, noverlap=cfg.win/2, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=128, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )#矩阵乘法X*melW.T
        X = np.log(X + 1e-8)
        X = X.astype(np.float32)
        print(X.shape)
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        #pickle.dump(对象, 文件，[使用协议])将对象转换为一种可以传输或存储的格式  

        
##time shifting   改变音频起点slightly shift the starting point of the audio, then pad it to original length.     
def GetMel_Time_Shifting( wav_fd, fe_fd, n_delete ):
    #wav_fd，fe_fd分别为波形和特征文件路径
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    #listdir返回指定路径下的文件和文件夹列表；endswith如果字符串含有指定的后缀返回True，否则返回False。
    names = sorted(names)
    for na in names:
        print(na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
        assert fs==cfg.fs#assert在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）
        
        p_1 = np.random.rand()
        p = 0.5
        if p_1 > p:
            start_ = int(np.random.uniform(-4800,4800))
            if start_ >= 0:
                wav_ts = np.r_[wav[start_:], np.random.uniform(-0.001,0.001, start_)]
            else:
                wav_ts = np.r_[np.random.uniform(-0.001,0.001, -start_), wav[:start_]]
        else:
            wav_ts = wav   
            
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav_ts, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=64, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )#矩阵乘法X*melW.T
        X = np.log(X + 1e-8)
        X = X.astype(np.float32)
        
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        #pickle.dump(对象, 文件，[使用协议])将对象转换为一种可以传输或存储的格式   
        
##Time Stretching  音速     
def GetMel_Time_Stretch( wav_fd, fe_fd, n_delete ):
    #wav_fd，fe_fd分别为波形和特征文件路径
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    #listdir返回指定路径下的文件和文件夹列表；endswith如果字符串含有指定的后缀返回True，否则返回False。
    names = sorted(names)
    for na in names:
        print(na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
        assert fs==cfg.fs#assert在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）
        
        speed_rate = np.random.uniform(0.7,1.3)
        p_1 = np.random.rand()
        p = 0.5
        if p_1 > p:
            wav_st = librosa.effects.time_stretch(wav, speed_rate) 
        else:
            wav_st = wav
#        wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
#        print('speed rate: %.3f' % speed_rate, '(lower is faster)')
#        if len(wav_speed_tune) < 16000:
#            pad_len = 16000 - len(wav_speed_tune)
#            wav_st = np.r_[np.random.uniform(-0.001,0.001,int(pad_len/2)),
#                                   wav_speed_tune,
#                                   np.random.uniform(-0.001,0.001,int(np.ceil(pad_len/2)))]
#        else: 
#            cut_len = len(wav_speed_tune) - 16000
#            wav_st = wav_speed_tune[int(cut_len/2):int(cut_len/2)+16000]
            
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav_st, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=64, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )#矩阵乘法X*melW.T
        X = np.log(X + 1e-8)
        X = X.astype(np.float32)
        
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        #pickle.dump(对象, 文件，[使用协议])将对象转换为一种可以传输或存储的格式   
##pitch_shift 音高
def GetMel_Pitch_Shift( wav_fd, fe_fd, n_delete ):
    #wav_fd，fe_fd分别为波形和特征文件路径
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    #listdir返回指定路径下的文件和文件夹列表；endswith如果字符串含有指定的后缀返回True，否则返回False。
    names = sorted(names)
    for na in names:
        print(na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
        assert fs==cfg.fs#assert在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）
        
        n_steps = np.random.uniform(-6,6)
        p_1 = np.random.rand()
        p = 0.5
        if p_1 > p:
            wav_ps = librosa.effects.pitch_shift(wav, fs, n_steps=n_steps)
        else:
            wav_ps = wav
            
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav_ps, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=64, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )#矩阵乘法X*melW.T
        X = np.log(X + 1e-8)
        X = X.astype(np.float32)
        
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        #pickle.dump(对象, 文件，[使用协议])将对象转换为一种可以传输或存储的格式  
##bg      
def GetMel_bg( wav_fd, fe_fd, n_delete ):
    #wav_fd，fe_fd分别为波形和特征文件路径
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    #listdir返回指定路径下的文件和文件夹列表；endswith如果字符串含有指定的后缀返回True，否则返回False。
    names = sorted(names)
    for na in names:
        print(na)
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        if ( wav.ndim==2 ): 
            wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
        assert fs==cfg.fs#assert在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）
        p_1 = np.random.rand()
        p = 0.5
        if p_1 > p:
            bg_files = os.listdir('E:/projects/date/_background_noise_/')
            bg_files.remove('README.md')
            chosen_bg_file = bg_files[np.random.randint(6)]
            bg, sr = librosa.load('E:/projects/date/_background_noise_/'+chosen_bg_file, sr=None)
            
            start_ = np.random.randint(bg.shape[0]-64000)
            bg_slice = bg[start_ : start_+64000]
            wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)
        else:
            wav_with_bg = wav
           
        ham_win = np.hamming(cfg.win)
        [f, t, X] = signal.spectral.spectrogram( wav_with_bg, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' ) 
        X = X.T
        
        # define global melW, avoid init melW every time, to speed up. 
        if globals().get('melW') is None:
            global melW
            melW = librosa.filters.mel( fs, n_fft=cfg.win, n_mels=64, fmin=0., fmax=8000 )
            melW /= np.max(melW, axis=-1)[:,None]
            
        X = np.dot( X, melW.T )#矩阵乘法X*melW.T
        X = np.log(X + 1e-8)
        X = X.astype(np.float32)
        
        out_path = fe_fd + '/' + na[0:-10] + '.f'
        pickle.dump( X, open(out_path, 'wb'), protocol=pickle.HIGHEST_PROTOCOL )
        #pickle.dump(对象, 文件，[使用协议])将对象转换为一种可以传输或存储的格式 


### format label
# get tags
def GetTags( info_path ):
    with open( info_path, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)#list() 方法用于将元组转换为列表。
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)标签分类
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y

### if set fold=None means use all data as training data
def GetAllData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'r') as f:
        reader = csv.reader(f)
        lis = list(reader)
   #获取交叉验证的数据列表     
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
    tr_na_list, te_na_list = [], []
        
    # read one line，第二列为文件名，第三列为当前文件块
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = pickle.load(open(fe_path, 'rb'))
        
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ] * len( X3d )
            te_na_list.append( na )
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ] * len( X3d )
            tr_na_list.append( na )

    if fold is None:
        return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ), tr_na_list
    else:
        return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ), tr_na_list, \
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist ), te_na_list
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)#os.makedirs() 方法用于递归创建目录 #os.path.exists(fd)如果fd存在，返回True；如果fd不存在，返回False。 