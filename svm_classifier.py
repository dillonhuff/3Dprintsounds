from sklearn import svm
from sklearn.naive_bayes import GaussianNB

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
test = angleLines[6:8]

print '# of training sample groups = ', len(train)
print '# of testing sample groups  = ', len(test)

## Plot the spectrogram to view consistency
#plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, [])

# Assemble training and test arrays

## Clip the training and test ranges to be sure they are correct
def clip_ranges(ranges, clip_value):
    rs = []
    for r in ranges:
        rs.append((r[0] + clip_value, r[1] - clip_value))

    return rs

train = clip_ranges(train, 10)
test = clip_ranges(test, 10)

# print 'Train ranges'
# for ls in train:
#     print ls

# print 'Test ranges'
# for ls in test:
#     print ls

## Put the training data into matrices
def build_labels(train_ranges, positive_ranges):
    labels = []
    for i in range(0, len(train_ranges)):
        samples_in_range = train_ranges[i][1] - train_ranges[i][0]
        print 'Samples in range =', samples_in_range
        
        if i in positive_ranges:
            for j in range(0, samples_in_range):
                labels.append(90)
        else:
            for j in range(0, samples_in_range):
                labels.append(45)
            

    return labels

def take_row_ranges(train_ranges, array_2d):
    wanted = []

    for i in range(0, len(train_ranges)):
        samples_in_range = train_ranges[i][1] - train_ranges[i][0]
        print 'Samples in range =', samples_in_range
        
        for j in range(train_ranges[i][0], train_ranges[i][1]):
            wanted.append(j)

    print wanted
    #return array_2d[np.logical_or.reduce([array_2d[:,1] == x for x in wanted])]
    return array_2d[np.array(wanted)]
    
def build_training_data(train_ranges, positive_ranges, spec):
    train_labels = build_labels(train_ranges, positive_ranges)

    print '# of training labels = ', len(train_labels)

    train_vectors = take_row_ranges(train_ranges, spec)

    print '# of training vectors = ', train_vectors.shape[0]

    assert(train_vectors.shape[0] == len(train_labels))

    return train_vectors, train_labels

ninety_deg_ranges = [0, 2, 4]
X, y = build_training_data(train, ninety_deg_ranges, angleSpectrogram)


# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                      hidden_layer_sizes=(500, 200), random_state=1)
# #clf = svm.SVC()

# clf.fit(X, y)

# print 'Score = ', clf.score(X, y)

gnb = GaussianNB()
gnbF = gnb.fit(X, y)
y_pred = gnbF.predict(X)
print("Number of mislabeled points in training set out of a total %d points : %d"
      % (X.shape[0],(y != y_pred).sum()))

# Build test data
Z, z = build_training_data(test, [0], angleSpectrogram)

y_pred = gnbF.predict(Z)
print("Number of mislabeled points in test set out of a total %d points : %d"
      % (Z.shape[0],(z != y_pred).sum()))

# Build test data from different file
Z, z = build_training_data(test, [0], angleSpectrogram)

y_pred = gnbF.predict(Z)
print("Number of mislabeled points in test set out of a total %d points : %d"
      % (Z.shape[0],(z != y_pred).sum()))

binSize = 2**10
squareSampleRate, squareSamples = wav.read("./Manual_square/iPhone6sAudio.wav")

squareSamples = take_first_seconds(77, squareSampleRate, squareSamples)
squareSamples = trim_first_seconds(68, squareSampleRate, squareSamples)

squareSpectrogram, squareFreqs = build_spectrogram(squareSampleRate, squareSamples, binSize)

## Clip away the high band frequencies with no real activity
squareSpectrogram = squareSpectrogram[:, 0:325]

square_lines = [(60, 185),
                (190, 310),
                (320, 445),
                (455, 575)]
plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, [])

Sq, sq = build_training_data(square_lines, [0, 1, 2, 3], squareSpectrogram)

y_pred = gnbF.predict(Sq)
print("Number of mislabeled points in test set out of a total %d points : %d"
      % (Sq.shape[0],(sq != y_pred).sum()))

for pt in y_pred:
    print pt

#print 'Score for test data = ', clf.score(Z, z)

# test_lines = []
# for r in test:
#     test_lines.append(r[0])
#     test_lines.append(r[1])

# plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, test_lines)

# # Unforgiveable sin of data analysis
# pred = clf.predict(Z)

# #for p in pred:
# for i in range(0, len(pred)):
#     p = pred[i]
#     print 'Predicted = ', p
#     print 'Actual = ', z[i]
