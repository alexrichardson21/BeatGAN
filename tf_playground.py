import tensorflow as tf
from pydub import AudioSegment
import numpy as np
import sys
# in: (20, 10, 2) out: (20, 18)
# out_dim = (fft_length = 2 * (inner - 1))
# inner = out / 2 + 1
def iFFT(x):
    real, imag = tf.split(x, 2, axis=-1)
    x = tf.complex(real, imag) 
    x = tf.squeeze(x, axis=[-1])
    x = tf.spectral.irfft(x)
    return x

# in: (20, 10) out: (20, 6, 2)
# out_dim = (fft_length / 2 + 1, 2)


def FFT(x):
    x = tf.spectral.rfft(x)
    extended_bin = x[..., None]
    return tf.concat([tf.real(extended_bin), tf.imag(extended_bin)], axis=-1)


tf.compat.v1.enable_eager_execution()
song = AudioSegment.from_file("datasets/alex_sc/120bpm/slices/ai security_1_slice0.wav", format='wav')
data = np.reshape(
    np.array(song.get_array_of_samples()), (210, 1, 420))
db = max([data.max(), abs(data.min())])
# data = data / db
for s in data:
    f = FFT(s)
    x = f.numpy()
    # x = tf.print(f, output_stream=sys.stderr, summarize=-1)
    fi = iFFT(f)
    y = fi.numpy()
    # print(fi)
    # print(f)
