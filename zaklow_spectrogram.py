#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal

from spectrogram_utils import take_first_seconds
from spectrogram_utils import trim_first_seconds
from spectrogram_utils import build_spectrogram
from spectrogram_utils import plot_spectrogram

binSize = 2**10
#plotstft("./angles_45/iPhone6sAudio.wav", binSize)
#plotstft("./CFP_KEY_2/iPhone6sAudio.wav", 2**10)
#plotstft("./angles/iPhone6sAudio.wav", 2**10)


anglesSampleRate, anglesSamples = wav.read("./angles_45/iPhone6sAudio.wav")
#wav.read("./angles/iPhone6sAudio.wav")

# Trims to about the actual program
#anglesSamples = take_first_seconds(1200, anglesSampleRate, anglesSamples)
#anglesSamples = trim_first_seconds(150, anglesSampleRate, anglesSamples)

anglesSamples = take_first_seconds(100, anglesSampleRate, anglesSamples)
anglesSamples = trim_first_seconds(33, anglesSampleRate, anglesSamples)

#plotstft_samples(anglesSampleRate, anglesSamples, binSize)

angleSpectrogram, angleFreqs = build_spectrogram(anglesSampleRate, anglesSamples, binSize)

# def test_move_split_points(numSegments, angleSpectrogram):
#     segmentSize = angleSpectrogram.shape[0] / numSegments
#     print 'Segment size = ', segmentSize
#     segmentStart = 0
#     segments = []
#     for i in range(0, numSegments):
#         segments.append(segmentStart) #(angleSpectrogram[segmentStart:(segmentStart + segmentSize):1])
#         segmentStart += segmentSize
#     return segments

# angleLines = test_move_split_points(360, angleSpectrogram)

angleLines = [250, 400,
              1060, 1200,
              1860, 2015,
              2675, 2825,
              3475, 3640,
              4285, 4450,
              5100, 5250,
              5900, 6050]

plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, angleLines)

print 'Angles spectrogram shape = ', angleSpectrogram.shape

squareSampleRate, squareSamples = wav.read("./Manual_square/iPhone6sAudio.wav")

squareSamples = squareSamples #[0:100000:1]

# squareSamples = take_first_seconds(76, squareSampleRate, squareSamples)
# squareSamples = trim_first_seconds(68, squareSampleRate, squareSamples)

squareSamples = take_first_seconds(74, squareSampleRate, squareSamples)
squareSamples = trim_first_seconds(69, squareSampleRate, squareSamples)


#plotstft_samples(squareSampleRate, squareSamples, binSize)

squareSpectrogram, squareFreqs = build_spectrogram(squareSampleRate, squareSamples, binSize)

plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, [0, 150])

print 'Square spectrogram shape = ', squareSpectrogram.shape

assert(len(angleFreqs) == len(squareFreqs))

for i in range(0, len(angleFreqs)):
    assert(angleFreqs[i] == squareFreqs[i])

singleSample = squareSpectrogram[0]
print 'singleSample shape =', singleSample.shape

singleSample90 = squareSpectrogram[150] # + (squareSpectrogram.shape[0] / 4)]
print 'singleSample90 shape =', singleSample90.shape

correlations = []
#for i in range(0, angleSpectrogram.shape[0]):

def break_spectrogram_into_segments(numSegments, angleSpectrogram):
    segmentSize = angleSpectrogram.shape[0] / numSegments
    print 'Segment size = ', segmentSize
    segmentStart = 0
    segments = []
    for i in range(0, numSegments):
        segments.append(angleSpectrogram[segmentStart:(segmentStart + segmentSize):1])
        segmentStart += segmentSize
    return segments
    
#moves = break_spectrogram_into_segments(360, angleSpectrogram)
#assert(len(moves) == 360)

for i in range(0, len(angleLines), 2):
    moveNum = (angleLines[i + 1] + angleLines[i]) / 2


    angleSpec = angleSpectrogram[moveNum]

    plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, [moveNum])
    # Pick a representative of the given move angle
    #angleSpec = testMove[testMove.shape[0] / 2]

    corVec = signal.correlate(angleSpec, singleSample)
    cor = np.amax(corVec) #np.linalg.norm(corVec)
    print 'Max Correlation with move angle', i, ' = ', cor

    corVec = signal.correlate(angleSpec, singleSample90)
    cor = np.amax(corVec) #np.linalg.norm(corVec)
    print 'Max Correlation of 90 degree move with angle', i, ' = ', cor
    
    #correlations.append(np.linalg.norm(cor))
    
# for i in range(0, 1000):
#     angleSpec = angleSpectrogram[i]
#     #print 'angleSpec shape =', angleSpec.shape
#     cor = signal.correlate(angleSpec, singleSample)
#     correlations.append(np.linalg.norm(cor))

# correlations.sort()
# correlations = reversed(correlations)

for c in correlations:
    print c

    #print 'Correlation norm = ', np.linalg.norm(cor)

def spec_cmp():
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
