from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from scipy import signal
import scipy

from classifier_utils import take_row_ranges
from classifier_utils import build_training_data
from classifier_utils import clip_ranges
from classifier_utils import predict_and_score

from spectrogram_utils import plotstft
from spectrogram_utils import take_first_seconds
from spectrogram_utils import trim_first_seconds
from spectrogram_utils import build_spectrogram
from spectrogram_utils import plot_spectrogram
from spectrogram_utils import sample_to_time
from spectrogram_utils import time_to_sample
from spectrogram_utils import sample_to_time_no_add
from spectrogram_utils import time_to_sample_no_add

print 'Scipy version =' , scipy.__version__

# Proper numpy based training data with spectrogram


## Set frequency cutoff
freq_cutoff = 325

## Build the full spectrogram for all input data
binSize = 2**10

ang10SampleRate, ang10Samples = wav.read("./angles_10/iPhone6sAudio.wav")

total_samples = ang10Samples.shape[0]

print 'Total samples =', total_samples

ang10Spectrogram, ang10Freqs = build_spectrogram(ang10SampleRate, ang10Samples, binSize)

total_time = 360
num_samples = total_time * (ang10SampleRate / binSize)

## Print out spectrogram time increments
timebins, freqbins = np.shape(ang10Spectrogram)

xlocs = np.float32(np.linspace(0, timebins-1, 5))
print '### xlocs'
for x in xlocs:
    print x

time_for_8200 = sample_to_time(8200, len(ang10Samples), timebins, binSize, ang10SampleRate)
print 'Sample 8200 is at time ', time_for_8200

print 'Sample 8200 is at time to sample', time_to_sample(time_for_8200, len(ang10Samples), timebins, binSize, ang10SampleRate)

ang10Spectrogram = ang10Spectrogram[:, 0:freq_cutoff]

prog_start = 3200
## 1800 mm / min -> 30 mm / sec
feedrate = 1800.0 / 60.0
move_distance = 40.0

move_time = move_distance / feedrate
fast_move_time = move_distance / (2*feedrate)
wait_time = 3
down_time = 0.31

print 'Spectrogram shape =', ang10Spectrogram.shape

num_samples = ang10Spectrogram.shape[0]
exec_time = num_samples / binSize

print 'Exec time = ', exec_time

spec_samples_per_second = 2*ang10SampleRate / binSize

print 'Spec samples per second =', spec_samples_per_second

move_spec_samples = time_to_sample_no_add(move_time, len(ang10Samples), timebins, ang10SampleRate)
fast_move_spec_samples = time_to_sample_no_add(fast_move_time, len(ang10Samples), timebins, ang10SampleRate)

wait_spec_samples = time_to_sample_no_add(3, len(ang10Samples), timebins, ang10SampleRate)

print 'Spectrogram per move = ', move_spec_samples

forty_second_line = time_to_sample(40, len(ang10Samples), timebins, binSize, ang10SampleRate)

sixty_second_line = forty_second_line + time_to_sample_no_add(20, len(ang10Samples), timebins, ang10SampleRate)

move_locs = [0]
last_move_start = 0
for i in range(0, 35):
    print i
    move_end = last_move_start + move_time
    move_locs.append(move_end)

    last_move_start = move_end + 2*wait_time + 2*down_time + fast_move_time

    move_locs.append(last_move_start)

move_locs.append(last_move_start + move_time)

print '# of move_locs =', len(move_locs)
ang10Lines = []
current_line = prog_start

for move_time in move_locs:
    ang10Lines.append(prog_start + time_to_sample(move_time, len(ang10Samples), timebins, binSize, ang10SampleRate))

plot_spectrogram(ang10Spectrogram, ang10Freqs, ang10Samples, ang10SampleRate, binSize, ang10Lines)

move_groups = []
for i in range(0, len(ang10Lines) - 1, 2):
    start = int(round(ang10Lines[i]))
    end = int(round(ang10Lines[i + 1]))

    #move_groups.append((int(round(ang10Lines[i])), int(round(ang10Lines[i + 1]))))
    move_groups.append((start, end))
    print 'Move group', i, 'has size', end - start

print '---- Move groups:'
print move_groups

assert(len(move_groups) == 36)

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
labels = labels + labels + labels + labels

print labels

clipped_data = clip_ranges(move_groups, 10)
train = clipped_data[0:35]
test = clipped_data[35:36]

# assert(len(train) == )
# assert(len(test) == 9)

assert(len(test) + len(train) == 36)

X = take_row_ranges(train, ang10Spectrogram)

y = []
for i in range(0, len(train)):
    lab = labels[i]
    y = y + ([lab] * (train[i][1] - train[i][0]))

y_test = []
for j in range(0, len(test)):
    i = j + len(train);
    lab = labels[i]

    print 'j = ', j, 'label =', lab

    y_test = y_test + ([lab] * (test[j][1] - test[j][0]))
    

print 'X shape = ', X.shape
print 'y length = ', len(y)
print 'y_test length = ', len(y_test)

assert(len(y) == X.shape[0])

gnb = GaussianNB()
gnbF = gnb.fit(X, y)

## Evaluate on training data data
y_pred = gnbF.predict(X)
print("Number of mislabeled points in training set out of a total %d points : %d"
          % (X.shape[0],(y_pred != y).sum()))

## Evaluate on test data
X_test = take_row_ranges(test, ang10Spectrogram)
y_pred = gnbF.predict(X_test)
print("Number of mislabeled points in undivided test set out of a total %d points : %d"
          % (X_test.shape[0],(y_pred != y_test).sum()))

sys.exit()

ninety_deg_ranges = [0, 2, 4]
X, y = build_training_data(train, ninety_deg_ranges, angleSpectrogram)

gnb = GaussianNB()
gnbF = gnb.fit(X, y)

print '--- Score on training data set ----'
predict_and_score(train, ninety_deg_ranges, angleSpectrogram)

# Build test data
predict_and_score(test, [0], angleSpectrogram)

# squareSampleRate, squareSamples = wav.read("./Manual_square/iPhone6sAudio.wav")

# squareSamples = take_first_seconds(77, squareSampleRate, squareSamples)
# squareSamples = trim_first_seconds(68, squareSampleRate, squareSamples)

# squareSpectrogram, squareFreqs = build_spectrogram(squareSampleRate, squareSamples, binSize)

# ## Clip away the high band frequencies with no real activity
# squareSpectrogram = squareSpectrogram[:, 0:freq_cutoff]

# # Test on transitional data
# square_lines = [(60, 185),
#                 (190, 310),
#                 (320, 445),
#                 (455, 575)]
# #plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, [])

# print '---- Score on clipped square cutting dataset ----'
# predict_and_score(square_lines, [0, 1, 2, 3], squareSpectrogram)

# Sq, sq = build_training_data(square_lines, [0, 1, 2, 3], squareSpectrogram)
# y_pred = gnbF.predict(Sq)
# # print("Number of mislabeled points in test set out of a total %d points : %d"
# #       % (Sq.shape[0],(sq != y_pred).sum()))

    
# # Test on entire dataset

# all_square_lines = [(60, 575)]

# print '---- Score on entire square printing dataset ----'
# predict_and_score(all_square_lines, [0], squareSpectrogram)

# inds = wanted_data(all_square_lines)

# wrong_labels = []
# for i in range(0, len(sq)):
#     if sq[i] != y_pred[i]:
#         wrong_labels.append(inds[i])

# plot_spectrogram(squareSpectrogram, squareFreqs, squareSamples, squareSampleRate, binSize, wrong_labels)
