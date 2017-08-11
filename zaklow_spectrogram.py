#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs

def build_spectrogram(samplerate, samples, binsize):
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    return ims, freq

def plotstft_samples(samplerate, samples, binsize, plotpath, colormap):
    ims, freq = build_spectrogram(samplerate, samples, binsize)
    
    timebins, freqbins = np.shape(ims)

    print ims.shape
    print '# of time bins = ', timebins
    print '# of frequency bins = ', freqbins
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    plt.colorbar()

    plt.xlabel("time (s)")
    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    plt.axvline(x=3000)

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()
    
def trim_first_seconds(secondsToTrim, sampleRate, samples):
    startSample = sampleRate*secondsToTrim
    return samples[startSample:samples.shape[0]:1]

def take_first_seconds(secondsToTake, sampleRate, samples):
    endSample = sampleRate*secondsToTake
    return samples[0:endSample]

""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    startS = 4991000
    endS   = 5009000

    # some cutoff at 30
    # cutoff init at 50

    # over at 80

    #samples = take_first_seconds(80, samplerate, samples)
    #samples = trim_first_seconds(65, samplerate, samples)
    #samples = samples #samples[startS:endS:1]
    plotstft_samples(samplerate, samples, binsize, plotpath, colormap)

#plotstft("my_audio_file.wav")
#plotstft("./CFP_KEY_2/iPhone6sAudio.wav", 2**10)
plotstft("./angles/iPhone6sAudio.wav", 2**10)

binSize = 2**10

anglesSampleRate, anglesSamples = wav.read("./angles/iPhone6sAudio.wav")

anglesSamples = anglesSamples[0:100000:1]

angleSpectrogram, angleFreqs = build_spectrogram(anglesSampleRate, anglesSamples, binSize)


print 'Angles spectrogram shape = ', angleSpectrogram.shape

squareSampleRate, squareSamples = wav.read("./CFP_KEY_2/iPhone6sAudio.wav")

squareSamples = anglesSamples[0:100000:1]

squareSpectrogram, squareFreqs = build_spectrogram(squareSampleRate, squareSamples, binSize)

print 'Square spectrogram shape = ', squareSpectrogram.shape

assert(len(angleFreqs) == len(squareFreqs))

for i in range(0, len(angleFreqs)):
    assert(angleFreqs[i] == squareFreqs[i])

for i in range(0, angleSpectrogram.shape[0]):
    angleSpec = angleSpectrogram[i]
    print 'angleSpec shape =', angleSpec.shape

    for j in range(0, squareSpectrogram.shape[0]):
        squareSpec = squareSpectrogram[j]

        cor = signal.correlate(angleSpec, squareSpec)

        print 'Correlation norm =', np.linalg.norm(cor)

        # plt.plot(cor, color='B')
        # plt.xlabel('Frequency (Khz)')
        # plt.ylabel('Correlation')
        # plt.show()
