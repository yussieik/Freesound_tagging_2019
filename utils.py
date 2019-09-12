# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:11:24 2019

@author: YUSS
"""
from scipy.io import wavfile
from python_speech_features import logfbank
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import librosa
#import multiprocessing
#import time
#import csv
import os
#import wave, struct, math, random

#class Config:
#    def __init__(self, mode = 'conv', nfilt=26, nfeat=13, nfft=512, rate=16000):
#        self.mode = mode
#        self.nfilt = nfilt
#        self.nfeat = nfeat
#        self.nfft = nfft
#        self.rate = rate
#        self.step = int(rate/10)
#
#def build_rand_feat():
#    X = []
#    y = []
#    _min, _max = float('inf'), -float('inf')
#    for _ in tqdm(range(n_samples)):
#        rand_class = np.random.choice(class_dist.index, p=prob_dist)
#        file = np.random.choice(df[df.label==rand_class].index)
#        rate, wav = wavfile.read('clean/'+file)
#        label = df.at[file, 'label']
#        rand_index = np.random.randint(0, wav.shape[0]-config.step)
#        sample = wav[rand_index:rand_index + config.step]
#        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
#        _min = min(np.amin(X_sample), _min)
#        _max = max(np.amax(X_sample), _max)
#        X.append(X_sample if config.mode == 'conv' else X_sample.T)
#        y.append(classes.index(label))
#    X, y = np.array(X), np.array(y)
#    X = (X - _min) / (_max - _min)
#    if config.mode == 'conv':
#        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
#    elif config.mode == 'time':
#        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
#    y = to_categorical(y, num_classes = 1)

"""
Create function for deviding n overlaping bins = sample_signal / n. Such as a running kernel(A function) on the signal.
That function will provide mffc for each bin. A concatentaion will put all the mffcs into one piece, that we'll feed to the net.
"""
class Preprocess():
    def __init__(self, wavfiles):
        self.wavfiles = wavfiles
        print("Preprocess from utils is ready for use...")

    def upload(self, directory, start, end, step):
        for_upload = {}
        for filename in os.listdir(directory)[start:end:step]:
            a = np.load(directory+filename)
            a = a[()]
            for_upload = {**for_upload, **a}
        return for_upload

    """Calculate fft"""
    def calc_fft(self, y, rate):
        n = len(y)
        freq = np.fft.rfftfreq(n, d = 1/rate)
        Y = abs(np.fft.rfft(y)/n)
        return (Y,freq)

    def download_fft_signals(self):
        signals = {}
        raw = {}
        fft = {}
        ffts = {}
        counter = 0
        for i in range(0,len(self.wavs)):
            rate, signal = wavfile.read(self.wavfiles)
            signals[self.wavfiles[i]] = (rate, signal)
            fft[self.wavfiles[i]] = (self.calc_fft(signal, rate))
            counter += 1
            if(counter % 1000 == 0):
                raw = signals
                ffts = fft
                np.save('raw/raw_data_'+str(counter), raw)
                np.save('fft/fft_'+str(counter), ffts)
                raw = {}
                ffts = {}
                signals = {}
                fft = {}
            elif(counter == len(self.wavs)):
                raw = signals
                ffts = fft
                np.save('raw_data_'+str(counter), raw)
                np.save('fft_'+str(counter), ffts)
                raw = {}
                ffts = {}
                signals = {}
                fft = {}

    def loc_problematic(self, fft, filename):
        if(np.mean(fft[0]) == 0):
            return filename

    """Thought experiment: get random piece out of the signal"""
    def rand_indx(self, signal, rate):
        if len(signal)==rate:
            return signal
        else:
            indx=np.random.randint(len(signal)-rate-1)
            return signal[indx:indx+rate-1]

    """Use envelope to reduce noise"""
    def envelope(self, y, rate, threshold):
        mask = []
        y = pd.Series(y).apply(np.abs)
        y_mean = y.rolling(window=int(rate/10), min_periods=1, center = True).mean()
        for mean in y_mean:
            if mean > threshold:
                mask.append(True)
            else:
                mask.append(False)
        return mask

    def plot_signals(self, signals):
        keys_list=list(signals.keys())
        for j in range(0, len(signals), 2):
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

    def plot_sig_fft(self, threshold, file_name, signals, ffts):
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

        mask = self.envelope(amp_0, signals[0], threshold)
        n_signal = amp_0[mask]

        if(len(n_signal) > 0):
            n_amp, n_freq = self.calc_fft(n_signal, 44100)
            time_n = np.arange(len(n_signal))
            axes[1,0].plot(time_n, n_signal,'r')
            axes[1,0].get_xaxis().set_visible(True)
            axes[1,0].get_yaxis().set_visible(True)

            axes[1,1].plot(n_freq, n_amp,'r')
            axes[1,1].get_xaxis().set_visible(True)
            axes[1,1].get_yaxis().set_visible(True)

        fig.savefig('pics/'+file_name[:-3]+'.png')

    """plot fft(fast fourier transform)"""
    def plot_fft(self, fft):
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

                    i += 1


    def plot_fbank(self, fbank):
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
    def plot_mfccs(self, mfccs):
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

    """Save audio mfcc for further usage"""
    def plot_save_mffcs(self, mfccs):
        counter=0
        for filename,m in mfccs.items():
            fig=plt.figure()
    #        plt.title(filename)
            plt.imshow(m,cmap='hot')
            ax=plt.gca()
    #        plt.show()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.savefig('./train_mfccs/'+filename[:-4]+'.png')
    #        if (counter%100==0):
    #            time.sleep(10)
            counter+=1

