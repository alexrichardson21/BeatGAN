from __future__ import division, print_function

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import glob

import keras.backend as K



import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import aubio
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.layers import Activation, Dense, Dropout, Flatten, Input, Reshape, Bidirectional, TimeDistributed, BatchNormalization, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D, Conv2DTranspose
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from audio_god import AudioGod
# from keras_contrib.layers.normalization.instancenormalization import \
#     InstanceNormalization


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', input_shape=None):
    if input_shape:
        x = Lambda(lambda x: K.expand_dims(x, axis=2), input_shape=input_shape)(input_tensor)
    else:
        x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(
        kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

class BeatGAN():
    def __init__(self):
        self.channels = 2
        self.bpm = 120
        self.beats = 16
        self.sample_rate = 44100
        self.samples_per_beat = int(60 / self.bpm * self.sample_rate)
        self.wav_shape = (self.samples_per_beat * self.beats, self.channels)
        
        self.ngf = 32
        self.ndf = 32
        self.noise = 10
        
        self.k = (12, 1)
        self.s = (2, 1)

        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])


        # The generator takes noise as input and generated imgs
        z = Input((self.noise,))
        song = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(song)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def build_generator(self):
        # input shape: (noise,)

        # for convolution kernel
        conv_init = RandomNormal(0, 0.02)
        # for batch normalization
        gamma_init = RandomNormal(1., 0.02)

        # define CNN model
        cnn = Sequential()
        
        cnn.add(Reshape((45,1,10), input_shape=(45,10)))
        cnn.add(Conv2DTranspose(
            filters=128,
            kernel_size=self.k,
            strides=self.s,
            padding='same',
            kernel_initializer=conv_init
        ))
        cnn.add(BatchNormalization(momentum=.8, gamma_initializer=gamma_init,))
        cnn.add(LeakyReLU(alpha=.2))

        cnn.add(Conv2DTranspose(
            filters=64,
            kernel_size=self.k,
            strides=self.s,
            padding='same',
            kernel_initializer=conv_init
        ))
        cnn.add(BatchNormalization(momentum=.8, gamma_initializer=gamma_init,))
        cnn.add(LeakyReLU(alpha=.2))

        cnn.add(Conv2DTranspose(
            filters=2,
            kernel_size=self.k,
            strides=1,
            padding='same',
            kernel_initializer=conv_init
        ))
        cnn.add(Activation('tanh'))

        cnn.add(Reshape((180, 2)))

        
        # define ConvLSTM model
        convlstm = Sequential()
        
        
        convlstm.add(
            LSTM(units=20*5, return_sequences=True, input_shape=(245, 5)))
        convlstm.add(LSTM(units=45*10, return_sequences=True))
        convlstm.add(Reshape((245, 45, 10)))
        convlstm.add(TimeDistributed(cnn))

        
        # define LSTM model
        lstm = Sequential()
        
        lstm.add(LSTM(units=120*5, return_sequences=True,
                      input_shape=(self.beats, 10)))
        lstm.add(LSTM(units=245*5, return_sequences=True))
        lstm.add(Reshape((self.beats, 245, 5)))
        lstm.add(TimeDistributed(convlstm))

        lstm.summary()
        convlstm.summary()
        cnn.summary()

        noise = Input(shape=(self.beats, 10))
        img = lstm(noise)

        return Model(noise, img)
    
    def build_discriminator(self):
        # input shape: (songs, bars, slices, samples per slice, channels)
        
        cnn = Sequential()
        
        # define CNN model
        # iterates over samples
        # input shape: (samples, channels)
        cnn.add(
            Conv1D(
                filters=self.ndf,
                kernel_size=12,
                strides=2,
                padding='same',
                input_shape=(210, 2),
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))
    
        cnn.add(
            Conv1D(
                filters=self.ndf*2,
                kernel_size=12,
                strides=2,
                padding='same',
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        cnn.add(
            Conv1D(
                filters=self.ndf*4,
                kernel_size=12,
                strides=1,
                padding='same',
                use_bias=False,
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Flatten())
        cnn.summary()

        # define ConvLSTM model
        # iterates over slices
        # input shape: (slices, samples per slice, channels)
        convlstm = Sequential()
        convlstm.add(TimeDistributed(cnn, input_shape=(245, 180, 2)))

        convlstm.add(LSTM(units=128, return_sequences=True))
        convlstm.add(LSTM(units=256, return_sequences=True))
        
        convlstm.add(Dense(1))
        convlstm.add(Flatten())

        convlstm.summary()

        # define LSTM model
        # iterates over bars
        # input shape: (bars, slices, samples per slice, channels)
        lstm = Sequential()
        lstm.add(TimeDistributed(convlstm, input_shape=(self.beats, 245, 180, 2)))

        lstm.add(LSTM(units=128, return_sequences=True))
        lstm.add(LSTM(units=256, return_sequences=True))

        lstm.add(Dense(1, activation='sigmoid'))
        lstm.add(Flatten())

        lstm.summary()


        four_bars = Input(shape=(self.beats, 245, 180, 2))
        validity = lstm(four_bars)

        return Model(four_bars, validity)

    def build_new_gen(self):
        
        # define CNN model
        cnn = Sequential()

        cnn.add()
        
        cnn.add(Reshape((45,1,10), input_shape=(45,10)))
        cnn.add(Conv2DTranspose(
            filters=128,
            kernel_size=self.k,
            strides=self.s,
            padding='same',
        ))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(LeakyReLU(alpha=.2))

        cnn.add(Conv2DTranspose(
            filters=64,
            kernel_size=self.k,
            strides=self.s,
            padding='same',
        ))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(LeakyReLU(alpha=.2))

        cnn.add(Conv2DTranspose(
            filters=2,
            kernel_size=self.k,
            strides=1,
            padding='same',
        ))
        cnn.add(Activation('tanh'))

        cnn.add(Reshape((180, 2)))

        cnn.add(Lambda(lambda x: np.fft.ifft(x)))

        # define ConvLSTM model
        convlstm = Sequential()

        convlstm.add(
            LSTM(units=20*5, return_sequences=True, input_shape=(245, 5)))
        convlstm.add(LSTM(units=45*10, return_sequences=True))
        convlstm.add(Reshape((245, 45, 10)))
        convlstm.add(TimeDistributed(cnn))

    def build_new_dis(self):
        # cnn
        cnn = Sequential()

        # define CNN model
        # iterates over samples
        # input shape: (samples, channels)
            

        cnn.add(Lambda(lambda x: np.fft.fft(x), input_shape=(180,2)))
        cnn.add(
            Conv1D(
                filters=self.ndf,
                kernel_size=12,
                strides=2,
                padding='same',
                input_shape=(180, 2),
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        cnn.add(
            Conv1D(
                filters=self.ndf*2,
                kernel_size=12,
                strides=2,
                padding='same',
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        cnn.add(
            Conv1D(
                filters=self.ndf*4,
                kernel_size=12,
                strides=1,
                padding='same',
            )
        )
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))
        cnn.add(Flatten())

        cnn.summary()

        # define ConvLSTM model
        # iterates over slices
        # input shape: (slices, samples per slice, channels)
        convlstm = Sequential()
        convlstm.add(TimeDistributed(cnn, input_shape=(245, 180, 2)))

        convlstm.add(LSTM(units=256, return_sequences=True))
        convlstm.add(LSTM(units=128, return_sequences=True))

        convlstm.add(Flatten())
        convlstm.add(Dense(1))
        

        convlstm.summary()

    def train(self, training_dir, epochs, batch_size=32, save_interval=100, transform=50):

        # ---------------------
        #  Preprocessing
        # ---------------------

        # Load from training_dir and normalize dataset
        god = AudioGod()

        # Load from x training_dir
        X_train = god.load_songs(
            self.train_shape,
            training_dir,
        )

        start_time = datetime.datetime.now()

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.noise))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(
                imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(
                gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.noise))

            # The generator wants the discriminator to label the generated samples as valid (ones)
            valid_y = np.ones((batch_size, 1))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f] time: %s" %
                  (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

        # Save generator
        self.generator.save('art_dc_gan_generator.h5')
if __name__ == '__main__':
    bg = BeatGAN()
