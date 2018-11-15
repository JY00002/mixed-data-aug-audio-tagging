# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 14:51:38 2018

@author: pdl_jy
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import librosa
import matplotlib.pyplot as plt
import os
import cv2
import IPython.display as ipd
from Unity import readwav
EPS = 1e-8
from scipy import signal
def get_spectrogram(wav):
    D = librosa.stft(wav, n_fft=480, hop_length=160,
                     win_length=480, window='hamming')
    spect, phase = librosa.magphase(D)
    return spect
path = 'E:/projects/date/chime_home/chunks/CR_lounge_200110_1601.s0_chunk0.16kHz.wav'
path1 = 'E:/projects/date/chime_home/chunks/CR_lounge_200110_1601.s0_chunk1.16kHz.wav'
wav, fs = readwav( path )
wav1, fs1 = readwav( path1 )
plt.plot(wav, '-', );
plt.show()
plt.plot(wav1, '-', );
plt.show()
def scale(wav):
    if ( wav.ndim==2 ): 
        wav = np.mean( wav, axis=-1 )#axis通常有3个值：-1,0，1分别表示：默认，列，行,mean表示求平均。压缩列
    return wav

wav = scale(wav)
wav1 = scale(wav1)

def get_logmel(wav):
    ham_win = np.hamming(1024)
    [f, t, X] = signal.spectral.spectrogram( wav, window=ham_win, nperseg=1024, noverlap=512, detrend=False, return_onesided=True, mode='magnitude' ) 
    X = X.T
    melW = librosa.filters.mel( fs, n_fft=1024, n_mels=128, fmin=0., fmax=8000 )
    melW /= np.max(melW, axis=-1)[:,None]
    X = np.dot( X, melW.T )#矩阵乘法X*melW.T
    X = np.log(X + 1e-8)
    log_mel = X.astype(np.float32)
    return log_mel


log_mel = get_logmel(wav)
log_mel1 = get_logmel(wav1)
print('log_mel shape:', log_mel.shape)
plt.imshow(log_mel, aspect='auto', origin='lower')
#plt.title('log_mel of origin audio')
plt.show()
plt.imshow(log_mel1,aspect='auto', origin='lower')
plt.show()

alpha = 1.5
l = np.random.beta(alpha, alpha)
log_mel_mixup = l*log_mel + (1-l)*log_mel1
log_mel_samp=0.5*(log_mel+log_mel1)

plt.imshow(log_mel_mixup,aspect='auto', origin='lower')
plt.show()
plt.imshow(log_mel_samp,aspect='auto', origin='lower')
plt.show()
#de_mel= log_mel_samp-log_mel_mixup
#plt.imshow(de_mel,aspect='auto', origin='lower')
#plt.show()