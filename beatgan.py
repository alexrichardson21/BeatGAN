from __future__ import division, print_function

from keras_layer_normalization import LayerNormalization
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob
import math

import keras.backend as K
import tensorflow as tf

from pydub import AudioSegment, silence

import argparse
import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib.pyplot as plt
import numpy as np
import aubio
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape, Bidirectional, TimeDistributed, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.recurrent import LSTM
# from keras.layers import ConvRecurrent2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from audio_god import AudioGod

# from keras_contrib.layers.normalization.instancenormalization import \
#     InstanceNormalization

# Shape: (slices, channels, samples_per_slice)

class BeatGAN():
    def __init__(self, tempo, rnn_size, cnn_size):
        self.bpm = tempo
        
        self.channels = 1
        self.bit_rate = 128 * 1000
        self.bit_depth = 16
        self.sample_rate = 44100 #self.bit_rate // self.bit_depth
        
        # self.samples_per_beat = int(60 / self.bpm * self.sample_rate)
        # self.samples_per_bar = self.samples_per_beat * 4
        self.samples_per_bar = self.sample_rate * 60 // self.bpm * 4

        
        self.ngf = 8
        self.ndf = 4
        self.noise = 5
        
        self.k = (8, 2)
        self.s = (2, 1)

        
        # shape = None
        # slice_combos = []
        # for i in range(1, int(self.samples_per_bar**0.5)+1):
        #     if self.samples_per_bar % i == 0:
        #         slice_combos += [(i, self.samples_per_bar // i)]
        # slice_combos += [(b, a) for a, b in slice_combos]

        # while shape not in slice_combos:
        #     [print(combo) for combo in slice_combos]
        #     print('Pick same combo:')
        #     ns = input('<num_slices>\n')
        #     sps = input('<samples_per_slice>\n')
        #     shape = (int(ns), int(sps))

        self.slices = rnn_size#int(ns)
        self.samples_per_slice = cnn_size#int(sps)
        self.shape = (self.slices, self.channels, self.samples_per_slice)
        
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the generator
        self.generator = self.build_new_gen()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Build and compile the discriminator
        self.discriminator = self.build_new_dis()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


        # The generator takes noise as input and generated imgs
        z = Input((self.slices, self.noise))
        song = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The same takes generated images as input and determines sameity
        same = self.discriminator(song)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines sameity
        self.combined = Model(z, same)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def build_new_gen(self):
        
        # INIT
        ##################
        nconvs = 7
        nlstms = 3
        m = [2**x for x in range(nlstms+nconvs)]
        m.reverse()
        
        conv_mults = m[nlstms:]
        lstm_mults = m[:nlstms]

        cnn_input_dim = self.samples_per_slice // 2 + 1
        zero_pad_i = []
        
        # determine conv input size and necessary zero padding
        for i in range(nconvs):
            if not (cnn_input_dim / 2).is_integer():
                cnn_input_dim -= 1
                zero_pad_i += [nconvs-i-1]
            cnn_input_dim //= 2
        
               
        # define CNN model
        ########################
        print("GENERATOR CNN")
        cnn = Sequential()

        cnn_input_size = cnn_input_dim*self.channels*lstm_mults[-1]*self.ngf
        
        cnn.add(
            Reshape((self.channels, cnn_input_dim, lstm_mults[-1] * self.ngf), input_shape=(cnn_input_size,)))
        
        # n Convolutions
        for i, mult in enumerate(conv_mults):
            cnn.add(Conv2DTranspose(
                filters=self.ngf*mult,
                kernel_size=(1, 8),
                strides=(1, 2),
                padding='same',
            ))
            if i in zero_pad_i:
                cnn.add(ZeroPadding2D(padding=((0, 0), (1, 0))))
            cnn.add(BatchNormalization(momentum=.8))
            cnn.add(Activation("relu"))

        # Final Convolution
        cnn.add(Conv2DTranspose(
                filters=2,
                kernel_size=(1, 4),
                strides=1,
                padding='same',
                activation='sigmoid',
                ))
        
        # in (None, 1, 211, 2)
        # out (None, 1, 420)
        def iFFT(x):
            real, imag = tf.split(x, 2, axis=-1)
            x = tf.complex(real, imag) 
            x = tf.squeeze(x, axis=[-1])
            x = tf.spectral.irfft(x)
            
            return x
        # cnn.summary()
        cnn.add(Lambda(iFFT))
        cnn.add(Activation('tanh'))
        
        cnn.summary()

        # define ConvLSTM model
        #########################
        print("GENERATOR LSTM")
        convlstm = Sequential()
        
        convlstm.add(
            LSTM(
                units=cnn_input_size*4, 
                return_sequences=True,
                input_shape=(self.slices, self.noise)))
        convlstm.add(LayerNormalization())

        convlstm.add(
            LSTM(
                units=cnn_input_size*2,
                return_sequences=True,))
        convlstm.add(LayerNormalization())

        convlstm.add(
            LSTM(
                units=cnn_input_size,
                return_sequences=True,))
        convlstm.add(LayerNormalization())
        
        
        convlstm.add(TimeDistributed(cnn))
        
        convlstm.summary()

        noise = Input(shape=(self.slices, self.noise))
        song = convlstm(noise)

        return Model(noise, song)

    def build_new_dis(self):
        
        # INIT
        ##################
        nconvs = 8
        nlstms = 2
        m = [2**x for x in range(nlstms+nconvs)]
        
        conv_mults = m[:nconvs]
        lstm_mults = m[nconvs:]
        
        # define CNN model
        ########################
        print("DISCRIMINATOR CNN")
        
        cnn = Sequential()

        # in: (None, 1, 420)  out: (None, 1, 211, 2)
        def FFT(x):
            x = tf.spectral.rfft(x)
            extended_bin = x[..., None]
            return tf.concat([tf.real(extended_bin), tf.imag(extended_bin)], axis=-1)
        
        cnn.add(
            Lambda(FFT, input_shape=(self.channels, self.samples_per_slice))
        )
        
        for mult in conv_mults:
            cnn.add(
                Conv2D(
                    filters=self.ndf*mult,
                    kernel_size=(1,6),
                    strides=(1,2),
                    padding='same',
                )
            )
            cnn.add(BatchNormalization(momentum=0.8))
            cnn.add(LeakyReLU(alpha=0.2))
        
        cnn.add(Flatten())
        # cnn.add(Dense(self.ndf*conv_mults[-1]))
        cnn.add(Dropout(.2))
        
        cnn.summary()

        
        # define ConvLSTM model
        #########################
        # input shape: (slices, samples per slice, channels)
        print("DISCRIMINATOR LSTM")
        
        convlstm = Sequential()
        convlstm.add(TimeDistributed(cnn, input_shape=self.shape))
        
        for mult in [2,4]:
            convlstm.add(LSTM(
                units=512*mult, 
                return_sequences=True,
                recurrent_dropout=0.1))
            convlstm.add(LayerNormalization())
        convlstm.add(LSTM(units=1, activation="sigmoid"))
        
        convlstm.summary()

        
        four_bars = Input(shape=self.shape)
        sameity = convlstm(four_bars)

        return Model(four_bars, sameity)

    def train(self, training_dir, epochs, batch_size=16, save_interval=10, preprocess=False):

        # ---------------------
        #  Preprocessing
        # ---------------------

        # Load from training_dir and normalize dataset
        god = AudioGod(training_dir, self.bpm, 1, self.shape)

        if preprocess:
            god.preprocess()
        X_train = god.load_slices()

        # -1 to 1
        db = max([X_train.max(), abs(X_train.min())])
        X_train = X_train / db

        start_time = datetime.datetime.now()

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            songs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.slices,self.noise))

            # Generate a half batch of new images
            gen_songs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                songs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                gen_songs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(
                0, 1, (batch_size, self.slices, self.noise))

            # The generator wants the discriminator to label the generated samples as same (ones)
            same_y = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, same_y)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f] time: %s" %
                  (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_beats(epoch, training_dir)

        # Save generator
        self.generator.save('art_dc_gan_generator.h5')

    def save_beats(self, epoch, dataset):
        NUM_BEATS = 10
        
        noise = np.random.normal(0, 1, (NUM_BEATS, self.slices, self.noise))
        
        gen_beats = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_beats = (gen_beats - .5) * 2

        if not os.path.exists("samples/%s_%dbpm" %
                              (dataset, self.bpm)):
            os.mkdir("samples/%s_%dbpm" %
                     (dataset, self.bpm))
        if not os.path.exists("samples/%s_%dbpm/epoch%d" % 
                            (dataset, self.bpm, epoch)):
            os.mkdir("samples/%s_%dbpm/epoch%d" % 
                    (dataset, self.bpm, epoch))

        for i, beat in enumerate(gen_beats):
            beat = np.reshape(beat, (-1, self.channels))
            wavfile.write(
                'samples/%s_%dbpm/epoch%d/%d.wav' % 
                (dataset, self.bpm, epoch, i+1), self.sample_rate, beat)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Art Bitch')
    parser.add_argument('epochs', type=int,
                        help='number of epochs')
    parser.add_argument('training_dir', type=str,
                        help='filepath of training set (if wikiart url is given then filepath becomes the save dir)')
    parser.add_argument('tempo', type=int,
                        help='Tempo of song output')
    parser.add_argument('rnn size', type=int,
                        help='size of recurrent layer')
    parser.add_argument('cnn size', type=int,
                        help='size of convolutional layer')
    parser.add_argument('-b', '--batchsize',
                        default=16, type=int, help='size of batches per epoch')
    parser.add_argument('-s', '--saveinterval',
                        type=int, default=10, help='interval to save sample images')
    parser.add_argument('-p', '--preprocess',
                        type=bool, default=False, help='preprocess songs')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_command_line_args()
    bg = BeatGAN(args['tempo'],
                 args['rnn size'],
                 args['cnn size'])
    bg.train(training_dir=args['training_dir'],
             epochs=args['epochs'],
             batch_size=args['batchsize'],
             save_interval=args['saveinterval'],
             preprocess=args['preprocess'])
