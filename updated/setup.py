# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:57:39 2019

@author: YUSS
"""

from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import multiprocessing
import time
import csv
import os
import wave, struct, math, random
import utils

prep = utils.Preprocess()

"""Upload and prepare dataset for usage"""
df = pd.read_csv(r'./one_hot_classes.csv')
df.set_index(df['Unnamed: 0'], inplace = True)
df.drop(columns = ['Unnamed: 0'], inplace = True)

"""Upload WAV file, extract rate and signal"""
for f in df.index:
    try:
        rate, signal = wavfile.read('audio/'+f)
        df.at[f, 'length'] = signal.shape[0]/rate
    except:
        df.drop(index = f, inplace = True)

df.to_csv('./whole_audio_l.csv')


"""Upload complete dataset"""
df = pd.read_csv(r'./whole_audio_l.csv')
df.set_index(df['Unnamed: 0'], inplace = True)
df.drop(columns = ['Unnamed: 0'], inplace = True)

"""Sort by audio length, show statistics"""
lengths = df['length']
lengths.hist()
df.sort_values(['length'], inplace = True)


"""Upload signals, ffts"""
fft = prep.upload('./fft/', 0, 25, 15)
signals = prep.upload('./raw/',0,25,15)


wavs = []
for route,dirs,files in os.walk('.\audio'):
    wavs = files

"""Calculate signals, ffts"""
for i in range(0,len(files),200):
    rate, signal = wavfile.read('audio/'+wavs[i])
    signals[wavs[i]] = (rate, signal)
    fft[wavs[i]] = prep.calc_fft(signal, rate)

#for filename in os.listdir('./raw'):
#     a = np.load('./raw/'+filename)
#     a = a[()]
#     raw = {**raw, **a}

"""Plots"""
prep.plot_sig_fft(signals, ffts, 5)
prep.plot_mfccs(mfccs)

prep.plot_save_mffcs()
prep.plot_fft(fft)
plot_fbank(fbank)
#signals[wav_file_nop] = signal
#mask = envelope(signal, rate, 60)
#signal = signal[mask]
#fft[wav_file_nop] = calc_fft(signal, rate)
#prep.plot_fft(fft)
#bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#fbank[wav_file_nop] = bank
#mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
#mfccs[wav_file_nop] = mel


"""
Load wav_files.
Clear the wav_files noise by using the envelope(using mean window amplitude).
If no signal left - we return the original signal, else the signal is returned after
envelop's mask.
If the signal's length is less than 1 second - concatenate/duplicate it until it reaches
1 second length.
Return mfccs of the processed signals.
"""

for route,dirs,files in os.walk('./audio'):
    counter=0
    for c in files:
        wav_file = c
        rate, signal = wavfile.read('audio/'+wav_file)
        signals[c] = signal
        test_lengths[c] = signal.shape[0]/rate
        while test_lengths[c] < 1.0:
            signals[c] = np.hstack((signals[c], signals[c]))
            test_lengths[c] = signals[c].shape[0]/rate
#        if(len(signals[c]) != 0):
#            fft[c] = calc_fft(signal, rate)
#            bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#            fbank[c] = bank
        bin_l = test_lengths[c] / 40
#        fft[c] = calc_fft(signal, rate)
        mel = mfcc(signals[c], rate, winlen = bin_l, winstep = bin_l/2, numcep = 13, nfft = math.ceil(44100 * bin_l)).T
        mfccs[c] = mel
        counter+=1
        if(counter%200 == 0):
            time.sleep(5)






"""Getting workable dataframe of wav files, length based"""


"""second_plus = df[df['length'] >= 1.0]"""
#

#for c in second_plus.index:
#    wav_file = c
#    rate, signal = wavfile.read('audio/'+wav_file)
#    signals[c] = signal
#    fft[c] = calc_fft(signal, rate)
#    bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#    fbank[c] = bank
#    mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
#    mfccs[c] = mel


"""
def calc_mfccs_signals():
    batchsize = 200
    for i in range(0, len(second_plus.index), batchsize):

        for c in second_plus.index[i:batchsize + i]:
            if(c not in mfccs):
                wav_file = c
                rate, signal = wavfile.read('audio/'+wav_file)
#        signals[c] = signal
#        fft[c] = calc_fft(signal, rate)
#        bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#        fbank[c] = bank
                mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
                mfccs[c] = mel

        if(i == 23900):
            batchsize = 131
        elif(i==24000):
            continue

        for c in second_plus.index[i:batchsize + i]:
            if(c not in mfccs):
                wav_file = c
                rate, signal = wavfile.read('audio/'+wav_file)
                mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
                mfccs[c] = mel

        time.sleep(10)
"""


#"""Calculate mfcc signals"""
#calc_mfccs_signals()
#plot_save_mffcs(mfccs)


"""flattened mfccs(DF.from_dict accepts 2-d[not matrices])"""
flatt_mfccs = {}
for k in mfccs:
    flatt_mfccs[k] = mfccs[k].flatten()


"""Save mfccs to .csv"""
mfccs_df = pd.DataFrame.from_dict(flatt_mfccs, orient = 'index')
mfccs_df.to_csv('./mfccs.csv')
np.save('./mfccs', mfccs)


"""Load mfccs.csv"""
mfccs_df = pd.read_csv('./mfccs.csv')
#a = np.load('./mfccs_randomidx.npy')
#mfccs = a[()]


n_signals = {}
#if len(os.listdir('clean')) == 0:
signal, rate = librosa.load(r'audio/00c7ff40.wav')
n_signals['0b0bd019.wav'] = signal
plot_signals(n_signals)
mask = envelope(signal, rate, 0)
signal = signal[mask]
signals['0b0bd019.wav'] = signal
plot_signals(signals)
wavfile.write(filename = 'clean/0af9f0b4.wav', rate =rate , data=signal)


for m in mfccs:
    empty = np.zeros((13,80))
    if mfccs[m].shape == (13, 79):
        empty[:,:-1] = mfccs[m]
        mfccs[m] = empty


for m in mfccs:
    mfccs[m] = mfccs[m].reshape()
