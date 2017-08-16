from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as metrics

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal
import scipy


from spectrogram_utils import plotstft
from spectrogram_utils import take_first_seconds
from spectrogram_utils import trim_first_seconds
from spectrogram_utils import build_spectrogram
from spectrogram_utils import plot_spectrogram

print 'Scipy version =' , scipy.__version__

# Proper numpy based training data with spectrogram


## Set frequency cutoff
freq_cutoff = 325

## Build the full spectrogram for all input data
binSize = 2**10
#plotstft("./angles/iPhone6sAudio.wav", 2**10)
#plotstft("./angles_10/iPhone6sAudio.wav", 2**10)

ang10SampleRate, ang10Samples = wav.read("./angles_10/iPhone6sAudio.wav")

total_samples = ang10Samples.shape[0]

print 'Total samples =', total_samples

#ang10Samples = take_first_seconds(80, ang10SampleRate, ang10Samples)
#ang10Samples = trim_first_seconds(25, ang10SampleRate, ang10Samples)
#f, t, Sxx = signal.spectrogram(ang10Samples, ang10SampleRate)

ang10Spectrogram, ang10Freqs = build_spectrogram(ang10SampleRate, ang10Samples, binSize)

total_time = 360
num_samples = total_time * (ang10SampleRate / binSize)

#plot_spectrogram(ang10Spectrogram, ang10Freqs, ang10Samples, ang10SampleRate, binSize, ang10Lines)
## Print out spectrogram time increments
timebins, freqbins = np.shape(ang10Spectrogram)

xlocs = np.float32(np.linspace(0, timebins-1, 5))
print '### xlocs'
for x in xlocs:
    print x

def sample_to_time(sample_num, total_num_samples, num_times, binSize, sample_rate):
    return ((sample_num*total_num_samples/num_times)+(0.5*binSize))/sample_rate

print 'Sample 8200 is at time ', sample_to_time(8200, len(ang10Samples), timebins, binSize, ang10SampleRate)

plot_times = ((xlocs*len(ang10Samples)/timebins)+(0.5*binSize))/ang10SampleRate

print '### Plot times'
for l in plot_times:
    print l


sys.exit()

ang10Spectrogram = ang10Spectrogram[:, 0:freq_cutoff]

prog_start = 3210
## 1800 mm / min -> 30 mm / sec
feedrate = 1800.0 / 60.0
move_distance = 40.0
move_time = move_distance / feedrate
fast_move_time = move_distance / (2*feedrate)

#move_samples = move_time * ang10SampleRate

print 'Spectrogram shape =', ang10Spectrogram.shape

num_samples = ang10Spectrogram.shape[0]
exec_time = num_samples / binSize

print 'Exec time = ', exec_time

spec_samples_per_second = 2*ang10SampleRate / binSize

print 'Spec samples per second =', spec_samples_per_second

#move_spec_samples = 2*((move_time * ang10SampleRate) / binSize)

move_spec_samples = spec_samples_per_second*move_time
fast_move_spec_samples = spec_samples_per_second*fast_move_time

wait_spec_samples = 3*spec_samples_per_second

#print 'Samples per move     = ', move_samples
print 'Spectrogram per move = ', move_spec_samples

ang10Lines = [prog_start, prog_start + move_spec_samples,
              prog_start + move_spec_samples + wait_spec_samples] #2*wait_spec_samples]# + fast_move_spec_samples]
plot_spectrogram(ang10Spectrogram, ang10Freqs, ang10Samples, ang10SampleRate, binSize, ang10Lines)

sys.exit()

anglesSampleRate, anglesSamples = wav.read("./angles_45/iPhone6sAudio.wav")

anglesSamples = take_first_seconds(100, anglesSampleRate, anglesSamples)
anglesSamples = trim_first_seconds(33, anglesSampleRate, anglesSamples)

angleSpectrogram, angleFreqs = build_spectrogram(anglesSampleRate, anglesSamples, binSize)

print 'Spectrogram shape =', angleSpectrogram.shape

## Clip away the high band frequencies with no real activity
angleSpectrogram = angleSpectrogram[:, 0:freq_cutoff]

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
plot_spectrogram(angleSpectrogram, angleFreqs, anglesSamples, anglesSampleRate, binSize, [])

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
        #print 'Samples in range =', samples_in_range
        
        if i in positive_ranges:
            for j in range(0, samples_in_range):
                labels.append(1)
        else:
            for j in range(0, samples_in_range):
                labels.append(0)
            

    return labels

def wanted_data(train_ranges):
    wanted = []

    for i in range(0, len(train_ranges)):
        samples_in_range = train_ranges[i][1] - train_ranges[i][0]
        
        for j in range(train_ranges[i][0], train_ranges[i][1]):
            wanted.append(j)

    return wanted

def take_row_ranges(train_ranges, array_2d):
    wanted = wanted_data(train_ranges)

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

# Create prediction test function
def predict_and_score(all_square_lines, positive_range_inds, squareSpectrogram):
    Sq, sq = build_training_data(all_square_lines,
                                 positive_range_inds,
                                 squareSpectrogram)

    y_pred = gnbF.predict(Sq)
    print("Number of mislabeled points in undivided test set out of a total %d points : %d"
          % (Sq.shape[0],(sq != y_pred).sum()))

    print 'Precision score =', metrics.precision_score(sq, y_pred, [45, 90])
    print 'Recall score =', metrics.recall_score(sq, y_pred, [45, 90])

print '--- Score on training data set ----'
predict_and_score(train, ninety_deg_ranges, angleSpectrogram)
# y_pred = gnbF.predict(X)
# print("Number of mislabeled points in training set out of a total %d points : %d"
#       % (X.shape[0],(y != y_pred).sum()))

# Build test data
predict_and_score(test, [0], angleSpectrogram)
# Z, z = build_training_data(test, [0], angleSpectrogram)

# y_pred = gnbF.predict(Z)
# print("Number of mislabeled points in test set out of a total %d points : %d"
#       % (Z.shape[0],(z != y_pred).sum()))

# Build test data from different file
# Z, z = build_training_data(test, [0], angleSpectrogram)

# y_pred = gnbF.predict(Z)
# print("Number of mislabeled points in test set out of a total %d points : %d"
#       % (Z.shape[0],(z != y_pred).sum()))

squareSampleRate, squareSamples = wav.read("./Manual_square/iPhone6sAudio.wav")

squareSamples = take_first_seconds(77, squareSampleRate, squareSamples)
squareSamples = trim_first_seconds(68, squareSampleRate, squareSamples)

squareSpectrogram, squareFreqs = build_spectrogram(squareSampleRate, squareSamples, binSize)

## Clip away the high band frequencies with no real activity
squareSpectrogram = squareSpectrogram[:, 0:freq_cutoff]

# Test on transitional data
square_lines = [(60, 185),
                (190, 310),
                (320, 445),
                (455, 575)]
#plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, [])

print '---- Score on clipped square cutting dataset ----'
predict_and_score(square_lines, [0, 1, 2, 3], squareSpectrogram)

Sq, sq = build_training_data(square_lines, [0, 1, 2, 3], squareSpectrogram)
y_pred = gnbF.predict(Sq)
# print("Number of mislabeled points in test set out of a total %d points : %d"
#       % (Sq.shape[0],(sq != y_pred).sum()))

    
# Test on entire dataset

all_square_lines = [(60, 575)]

print '---- Score on entire square printing dataset ----'
predict_and_score(all_square_lines, [0], squareSpectrogram)

inds = wanted_data(all_square_lines)

wrong_labels = []
for i in range(0, len(sq)):
    if sq[i] != y_pred[i]:
        wrong_labels.append(inds[i])

plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, wrong_labels)

# for pt in y_pred:
#     print pt

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
