from sklearn import svm

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal

from spectrogram_utils import take_first_seconds
from spectrogram_utils import trim_first_seconds
from spectrogram_utils import build_spectrogram
from spectrogram_utils import plot_spectrogram

X = [[0, 1], [1, 1]]

y = [0, 1]

clf = svm.SVC()

clf.fit(X, y)

pred = clf.predict([[2., 2.], [0, 0]])

for p in pred:
    print p

# Proper numpy based training data with spectrogram


## Build the full spectrogram for all input data
binSize = 2**10
#plotstft("./angles/iPhone6sAudio.wav", 2**10)

anglesSampleRate, anglesSamples = wav.read("./angles_45/iPhone6sAudio.wav")

anglesSamples = take_first_seconds(100, anglesSampleRate, anglesSamples)
anglesSamples = trim_first_seconds(33, anglesSampleRate, anglesSamples)

angleSpectrogram, angleFreqs = build_spectrogram(anglesSampleRate, anglesSamples, binSize)

print 'Spectrogram shape =', angleSpectrogram.shape

## Clip away the high band frequencies with no real activity
angleSpectrogram = angleSpectrogram[:, 0:325]

print 'Spectrogram shape =', angleSpectrogram.shape



## Approximate ranges for each train / test movement
angleLines = [(250, 400),
              (1060, 1200),
              (1860, 2015),
              (2675, 2825),
              (3475, 3640),
              (4285, 4450),
              (5100, 5250),
              (5900, 6050)]

train = angleLines[0:6]
test = angleLines[6:7]

print '# of training samples = ', len(train)
print '# of testing samples  = ', len(test)

## Plot the spectrogram to view consistency
plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, [])
