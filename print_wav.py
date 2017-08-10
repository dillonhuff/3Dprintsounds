import numpy as np
import scipy.io.wavfile

rate, data = scipy.io.wavfile.read('CFP_KEY_2/iPhone6sAudio.wav')

sin_data = np.sin(data)

print sin_data
