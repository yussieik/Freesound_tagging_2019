# -*- coding: utf-8 -*-
"""
Created on Sat May 18 19:46:25 2019

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

"""Various signal properities"""
signals = {}
fft = {}
fbank = {}
mfccs = {}
test_lengths = {}

"""Sort by audio length, show statistics"""
lengths = df['length']
lengths.hist()
df.sort_values(['length'], inplace = True)


"""Calculate fft"""
def calc_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n, d = 1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)


"""Use envelope to reduce noise"""
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center = True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask




#wav_file = '0a871061.wav'
#rate, signal = wavfile.read('audio/'+wav_file)
#signals[wav_file] = signal
#fft[wav_file] = calc_fft(signal, rate)
#bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#fbank[wav_file] = bank
#mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
#mfccs[wav_file] = mel
#tt = rand_indx(signal, rate)
#plot_fft(fft)
#plot_mfcc(mfccs)




wavs = []
for route,dirs,files in os.walk(r'G:\KAGGLE_COMPTS\Freesound Audio Tagging 2019\audio'):
    wavs = files

signals = {}
raw = {}
fft = {}
ffts = {}
np.save('./raw_data', signals)
np.save('./fft', fft)
counter = 0
for i in range(0,len(wavs)):
    rate, signal = wavfile.read('audio/'+wavs[i])
    signals[wavs[i]] = (rate, signal)
    fft[wavs[i]] = (calc_fft(signal, rate))
    counter += 1
    if(counter % 2000 == 0):
        raw = np.load('./raw_data.npy')
        ffts = np.load('./fft.npy')
        raw = raw[()]
        ffts = ffts[()]
        raw = {**raw, **signals}
        ffts = {**ffts, **fft}
        np.save('./raw_data', raw)
        np.save('./fft', ffts)
        raw = {}
        ffts = {}
        signal = {}
        fft = {}
    elif(counter == len(wavs)):
        raw = np.load('raw_data.npy')
        ffts = np.load('fft.npy')
        raw = raw[()]
        ffts = ffts[()]
        raw = {**raw, **signals}
        ffts = {**ffts, **fft}
        np.save('raw_data', raw)
        np.save('fft', ffts)
        raw = {}
        ffts = {}
        signal = {}
        fft = {}
#raw = np.load('raw_data.npy')
#raw = raw[()]
#time.sleep(10)
#a = np.load('./mfccs_randomidx.npy')
#mfccs = a[()]
#np.save('./mfccs', mfccs)

x={'new':1,'old':2,'used':3}
y={'blakc':0,'white':255}
z={**x,**y}




for i in range(0,len(files),200):
    rate, signal = wavfile.read('audio/'+wavs[i])
    signals[wavs[i]] = (rate, signal)
    fft[wavs[i]] = calc_fft(signal, rate)

plot_signals(signals)
#signals[wav_file_nop] = signal
#mask = envelope(signal, rate, 60)
#signal = signal[mask]

#fft[wav_file_nop] = calc_fft(signal, rate)
#plot_fft(fft)
#plot_signals(signals)
#bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#fbank[wav_file_nop] = bank
#mel = mfcc(signal[:rate], rate, numcep = 13, nfft = 1103).T
#mfccs[wav_file_nop] = mel



"""Thought experiment: get random piece out of the signal"""
def rand_indx(signal,rate):
    if len(signal)==rate:
        return signal
    else:
        indx=np.random.randint(len(signal)-rate-1)
        return signal[indx:indx+rate-1]

"""
Load wav_files.
Clear the wav_files noise by using the envelope(using mean window amplitude).
If no signal left - we return the original signal, else the signal is returned after
envelop's mask.
If the signal's length is less than 1 second - concatenate/duplicate it until it reaches
1 second length.
Return mfccs of the processed signals.
"""
for route,dirs,files in os.walk(r'G:\KAGGLE_COMPTS\Freesound Audio Tagging 2019\audio'):
    counter=0
    for c in files:
        wav_file = c
        rate, signal = wavfile.read('audio/'+wav_file)
        mask = envelope(signal, rate, 5)
        if(len(signal[mask]) == 0):
            signals[c] = signal
        else:
            signal = signal[mask]
        signals[c] = signal
        test_lengths[c] = signal.shape[0]/rate
        while test_lengths[c] < 1.0:
            signals[c] = np.hstack((signals[c], signals[c]))
            test_lengths[c] = signals[c].shape[0]/rate
#        if(len(signals[c]) != 0):
#            fft[c] = calc_fft(signal, rate)
#            bank = logfbank(signal[:rate], rate, nfilt = 26, nfft = 1103).T
#            fbank[c] = bank

        mel = mfcc(rand_indx(signals[c], rate), rate, numcep = 13, nfft = 1103).T
        mfccs[c] = mel
        counter+=1
        if(counter%200 == 0):
            time.sleep(10)
#
#N = 15.0* rate
#Y_k = np.fft.fft(signal)[0:int(N/2)]/N # FFT function from numpy
#Y_k[1:] = 2*Y_k[1:] # need to take the single-sided spectrum only
#Pxx = np.abs(Y_k) # be sure to get rid of imaginary part
#
#for i in range(0, 119, 10):


keys = fft.keys()

t_sig = signals['000ccb97.wav']
t_len = test_lengths['000ccb97.wav']

t_sig = np.hstack((t_sig, t_sig))
l = t_sig.shape[0]/44100

"""plot audio signal"""
def plot_signals(signals):
    keys_list=list(signals.keys())
    for j in range(0,119,10):
        my_keys=keys_list[j:j+10]
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=False, figsize=(20,5))
        fig.suptitle('Time Series', size=16)
        i = 0
        for x in range(2):
            for y in range(5):
                data = signals[my_keys[i]]
                Y, freq = data,np.arange(data.shape[0])
                axes[x,y].set_title(my_keys[i])
                axes[x,y].plot(freq, Y)
                axes[x,y].get_xaxis().set_visible(False)
                axes[x,y].get_yaxis().set_visible(True)
                i += 1

def plot_sig_fft(signals, ffts, threshold, file_name):
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False,
                             sharey=False, figsize=(20,5))
    fig.suptitle(file_name)
    amp_0 = signals[1]
    time = np.arange(amp_0.shape[0])
    axes[0,0].set_title('raw data')
    axes[0,0].plot(time,amp_0)
    axes[0,0].get_xaxis().set_visible(True)
    axes[0,0].get_yaxis().set_visible(True)

    freq = ffts[1]
    amp = ffts[0]
    axes[0,1].set_title('fft')
    axes[0,1].plot(freq,amp)
    axes[0,1].get_xaxis().set_visible(True)
    axes[0,1].get_yaxis().set_visible(True)

    mask = envelope(amp_0, signals[0], threshold)
    n_signal = amp_0[mask]
    print(len(n_signal))
    if(len(n_signal) > 0):
        n_amp, n_freq = calc_fft(n_signal, 44100)
        time_n = np.arange(len(n_signal))
        axes[1,0].plot(time_n, n_signal,'r')
        axes[1,0].get_xaxis().set_visible(True)
        axes[1,0].get_yaxis().set_visible(True)

        axes[1,1].plot(n_freq, n_amp,'r')
        axes[1,1].get_xaxis().set_visible(True)
        axes[1,1].get_yaxis().set_visible(True)

    fig.savefig('pics/'+file_name[:-3]+'.png')

#for sig, fft in signals, fft:
    plot_sig_fft(signals[wavs[0]], fft[wavs[0]], 500, wavs[0])

for wav in wavs:
    if wav in signals.keys():
        plot_sig_fft(signals[wav], fft[wav], 5, wav)



def loc_problematic(fft, filename):
    if(np.mean(fft[0]) == 0):
        return filename

for key, value in fft.items:




plot_signals(signals)
plot_fft(fft)


"""plot fft(fast fourier transform)"""
def plot_fft(fft):
    keys_list=list(fft.keys())
    for j in range(0,119,10):
        my_keys=keys_list[j:j+10]
        fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True,
                                sharey=False, figsize=(20,5))
        fig.suptitle('Fourier Transforms', size=16)
        i = 0
        for x in range(2):
            for y in range(5):
                data = list(fft[my_keys[i]])
                Y, freq = data[0], data[1]
                axes[x,y].set_title(my_keys[i])
                axes[x,y].plot(freq, Y)
#               axes[x,y].get_xaxis().set_visible(False)
#               axes[x,y].get_yaxis().set_visible(False)
                i += 1


def plot_fbank(fbank):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=False, figsize=(20,5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(fbank.keys())[i])
            axes[x,y].imshow(list(fbank.values())[i],
                    cmap='hot', interpolation='nearest')
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1


"""plot audio mfcc(Mel-frequency cepstral coefficients)"""
def plot_mfccs(mfccs):
    fig, axes = plt.subplots(nrows=2, ncols=5, sharex=False,
                             sharey=False, figsize=(20,5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for x in range(2):
        for y in range(5):
            axes[x,y].set_title(list(mfccs.keys())[i])
            axes[x,y].imshow(list(mfccs.values())[i],
                    cmap='hot', interpolation='nearest')
#            axes[x,y].get_xaxis().set_visible(False)
#            axes[x,y].get_yaxis().set_visible(False)
            i += 1

plot_mfccs(mfccs)



"""Save audio mfcc for further usage"""
def plot_save_mffcs():
    counter=0
    for filename,mfcc in mfccs.items():
        fig=plt.figure()
#        plt.title(filename)
        plt.imshow(mfcc,cmap='hot')
        ax=plt.gca()
#        plt.show()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.savefig('train_mfccs/'+filename[:-3]+'png')
#        if (counter%100==0):
#            time.sleep(10)
        counter+=1

plot_save_mffcs()
plot_signals(signals)
#plt.show()
plot_fft(fft)
#plt.show()
plot_fbank(fbank)
#plt.show()
#plot_mfccs(mfccs)
#plt.show()



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
#np.save('./mfccs', mfccs)

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


