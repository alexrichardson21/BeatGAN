import tensorflow as tf

# in: (20, 10, 2) out: (20, 18)
# out_dim = (fft_length = 2 * (inner - 1))
# inner = out / 2 + 1
def iFFT(x):
    real, imag = tf.split(x, 2, axis=-1)
    x = tf.complex(real, imag)
    x = tf.squeeze(x, axis=[-1, -2])
    x = tf.spectral.irfft(x, )
    return x

# in: (20, 10) out: (20, 6, 2)
# out_dim = (fft_length / 2 + 1, 2)
def FFT(x):
    x = tf.squeeze(x, axis=[-1,-2])
    x = tf.spectral.rfft(x)
    extended_bin = x[..., None]
    x = tf.concat([tf.real(extended_bin), tf.imag(extended_bin)], axis=-1)
    return x

FFT(tf.random.uniform((20,10,1, 2)))
