from __future__ import division, print_function

import argparse
import datetime
import glob
import math
import os
import random
import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.layers import (Activation, BatchNormalization,
                          CuDNNLSTM, Dense, Dropout, Flatten, Input, Lambda,
                          Reshape, TimeDistributed)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, ZeroPadding2D
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.optimizers import Adam
from scipy.io import wavfile

from iwgan import wasserstein_loss, gradient_penalty_loss, RandomWeightedAverage
from functools import partial

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Shape: (slices, channels, samples_per_slice)

class BeatGAN():
    def __init__(self):
        self.bpm = 120

        self.channels = 1
        self.bars = 1
        self.bit_rate = 128 * 1000
        self.bit_depth = 16
        self.sample_rate = 44100
        self.samples_per_bar = self.sample_rate * 60 // self.bpm * 4

        self.ngf = 32
        self.ndf = 32
        self.noise = 100

        self.wave_shape = (self.samples_per_bar, self.bars, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the generator
        self.gen = self.build_gen()
        # self.gen.compile(
        #     loss='binary_crossentropy', optimizer=optimizer)

        # Build and compile the discriminators
        self.dis = self.build_dis()
        # self.dis.compile(loss='binary_crossentropy',
        #                                 optimizer=optimizer,
        #                                 metrics=['accuracy'])

        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)

        # Wasserstein Loss
        # Compile Generator Model
        z = Input((self.noise,))
        song = self.gen(z)

        self.dis.trainable = False

        same = self.dis(song)

        self.gen_model = Model(inputs=[z],
                               outputs=[same])
        self.gen_model.compile(optimizer=optimizer,
                               loss=wasserstein_loss)

        # Compile Discriminator Model
        self.dis.trainable = True
        self.gen.trainable = False

        song = Input(shape=self.wave_shape)
        z = Input(shape=(self.noise,))
        gen_song = self.gen(z)
        valid_gen = self.dis(gen_song)
        valid_real = self.dis(song)

        ave_song = RandomWeightedAverage()([song, gen_song])
        valid_ave = self.dis(ave_song)

        self.gp_weight = 10
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=ave_song,
                                  gradient_penalty_weight=self.gp_weight)

        # If we don't concatenate the real and generated samples, however, we get three
        # outputs: One of the generated samples, one of the real samples, and one of the
        # averaged samples, all of size BATCH_SIZE. This works neatly!
        self.critic_model = Model(inputs=[song, z],
                                  outputs=[valid_real, valid_gen,
                                           valid_ave])
        # We use the Adam paramaters from Gulrajani et al. We use the Wasserstein loss for both
        # the real and generated samples, and the gradient penalty loss for the averaged samples
        self.critic_model.compile(optimizer=optimizer,
                                  loss=[wasserstein_loss,
                                        wasserstein_loss,
                                        partial_gp_loss])

        # Functions need names or Keras will throw an error
        partial_gp_loss.__name__ = 'gradient_penalty'

    def build_gen(self):

        k = (30, 1)
        s = (4, 1)

        model = Sequential()

        # Input
        model.add(Dense(20*self.ngf*64, input_shape=(self.noise,)))
        model.add(Reshape((20, 1, self.ngf*64)))

        # Conv 1
        model.add(
            Conv2DTranspose(
                filters=self.ndf*32,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        model.add(ZeroPadding2D(((2, 4), (0, 0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Conv 2
        model.add(
            Conv2DTranspose(
                filters=self.ndf*16,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(ZeroPadding2D(((1,0),(0,0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Conv 3
        model.add(
            Conv2DTranspose(
                filters=self.ndf*8,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        model.add(ZeroPadding2D(((1, 1), (0, 0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Conv 4
        model.add(
            Conv2DTranspose(
                filters=self.ndf*4,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(ZeroPadding2D(((1, 1), (0, 0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Conv 5
        model.add(
            Conv2DTranspose(
                filters=self.ndf*2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(ZeroPadding2D(((1, 1), (0, 0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Conv 6
        model.add(
            Conv2DTranspose(
                filters=self.ndf,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        model.add(ZeroPadding2D(((4, 4), (0, 0))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))

        # Output
        model.add(
            Conv2DTranspose(
                filters=self.channels,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
            ))
        model.add(Activation('tanh'))

        model.summary()

        # x = Input(shape=(self.noise,))
        # y = model(x)

        # return Model(x,y)
        return model

    def build_dis(self):

        k = (30, 1)
        s = (4, 1)

        model = Sequential()

        def PhaseShuffle(x, rad=2, pad_type='reflect'):
            # x = tf.keras.backend.squeeze(x, axis=-2)
            b, x_len, nch = x._keras_shape

            phase = tf.random_uniform(
                [], minval=-rad, maxval=rad + 1, dtype=tf.int32)
            pad_l = tf.maximum(phase, 0)
            pad_r = tf.maximum(-phase, 0)
            phase_start = pad_r
            x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

            x = x[:, phase_start:phase_start+x_len]
            x.set_shape([b, x_len, nch])
            # x = tf.keras.backend.expand_dims(x, axis=-2)

            return x

        # Conv 1
        model.add(
            Conv2D(
                filters=self.ndf,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
                input_shape=self.wave_shape
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Conv 2
        model.add(
            Conv2D(
                filters=self.ndf*2,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Conv 4
        model.add(
            Conv2D(
                filters=self.ndf*4,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Conv 5
        model.add(
            Conv2D(
                filters=self.ndf*8,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Conv 6
        model.add(
            Conv2D(
                filters=self.ndf*16,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Conv 7
        model.add(
            Conv2D(
                filters=self.ndf*32,
                kernel_size=k,
                strides=s,
                padding='same',
                use_bias=False,
            ))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(0.2))
        model.add(Lambda(lambda x: tf.squeeze(x, axis=[-2])))
        model.add(Lambda(PhaseShuffle))
        model.add(Lambda(lambda x: tf.expand_dims(x, axis=[-2])))

        # Output
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        # x = Input(shape=self.wave_shape)
        # y = model(x)

        # return Model(x, y)
        return model

    def train(self, training_dir, epochs, training_ratio=5, batch_size=16, save_interval=50):

        # ---------------------
        #  Preprocessing
        # ---------------------

        # Load from training_dir and normalize dataset
        # all_file_names = glob.glob(
        #     'datasets/%s/%dbpm/slices/*.wav' % (training_dir, self.bpm))
        all_file_names = glob.glob(
            'E:/datasets/%s/%dbpm/slices/*.wav' % (training_dir, self.bpm))
        d_losses = []
        g_losses = []

        start_time = datetime.datetime.now()

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_d_losses = []
            for _ in range(training_ratio):
                songs = np.zeros((half_batch,) + self.wave_shape)
                num_songs = 0
                while num_songs < half_batch:
                    filename = np.random.choice(all_file_names)
                    song = wavfile.read(filename)[1].reshape(self.wave_shape)
                    if not np.array_equal(song, np.zeros(self.wave_shape)):
                        songs[num_songs] = song
                        num_songs += 1

                # -1 to 1
                songs = songs / 32768.0

                noise = np.random.normal(0, 1, (half_batch, self.noise))

                positive_y = np.ones((half_batch, 1))
                negative_y = -positive_y
                dummy_y = np.zeros((half_batch, 1))

                batch_d_losses += [self.critic_model.train_on_batch([songs, noise],
                                                              [positive_y, negative_y, dummy_y])]

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(
                0, 1, (batch_size, self.noise))

            positive_y = np.ones((batch_size, 1))

            g_loss = self.gen_model.train_on_batch(noise, positive_y)

            elapsed_time = datetime.datetime.now() - start_time

            ave_d_loss = np.average([l[0] for l in batch_d_losses])

            # Plot the progress
            print("%d/%d [D loss: %f] [G loss: %f] time: %s" %
                  (epoch+1, epochs, ave_d_loss, g_loss, elapsed_time))

            d_losses.append(ave_d_loss)
            g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_beats(epoch, training_dir)

                # Save generator
                self.gen_model.save('E:/models/beat_gan_lstm_generator.h5')
                self.critic_model.save(
                    'E:/models/beat_gan_lstm_discriminator.h5')

        # Plot Loss Graph
        plt.plot(range(epochs), d_losses)
        plt.plot(range(epochs), g_losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('loss_graph.png')

    def save_beats(self, epoch, dataset):
        NUM_BEATS = 10

        noise = np.random.normal(0, 1, (NUM_BEATS, self.noise))
        gen_beats = self.gen.predict(noise)

        # # Rescale images 0 - 1
        # gen_beats = (gen_beats - .5) * 2
        gen_beats = gen_beats.astype(np.float32)

        if not os.path.exists("samples"):
            os.mkdir("samples")

        for i, beat in enumerate(gen_beats):
            beat = np.reshape(beat, (-1, self.channels))
            filename = '%s_%dbpm_epoch%d_%d.wav' % (
                dataset, self.bpm, epoch, i+1)
            try:
                wavfile.write('E:/samples/%s' %
                              filename, self.sample_rate, beat)
            except:
                print("x")


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Beats Bitch')
    parser.add_argument('epochs', type=int,
                        help='number of epochs')
    parser.add_argument('training_dir', type=str,
                        help='filepath of training set')
    parser.add_argument('-b', '--batchsize',
                        default=16, type=int, help='size of batches per epoch')
    parser.add_argument('-s', '--saveinterval',
                        type=int, default=1000, help='interval to save sample images')
    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_command_line_args()
    bg = BeatGAN()
    bg.train(training_dir=args['training_dir'],
             epochs=args['epochs'],
             batch_size=args['batchsize'],
             save_interval=args['saveinterval'])
