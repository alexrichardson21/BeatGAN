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
    def __init__(self, tempo, rnn_size, cnn_size):
        self.bpm = tempo
        
        self.channels = 1
        self.bit_rate = 128 * 1000
        self.bit_depth = 16
        self.sample_rate = 44100 #self.bit_rate // self.bit_depth
        
        self.samples_per_bar = self.sample_rate * 60 // self.bpm * 4

        
        self.ngf_cnn = 128
        self.ndf_cnn = 64
        self.ngf_lstm = 100
        self.ndf_lstm = 100
        self.noise = 100

        self.slices = rnn_size #int(ns)
        self.samples_per_slice = cnn_size #int(sps)
        self.lstm_shape = (self.slices, self.channels, self.samples_per_slice)
        self.cnn_shape = (self.channels, self.samples_per_slice)
        
        optimizer = Adam(0.0002, 0.5)
        
        # Build and compile the generator
        self.gen = self.build_gen()
        self.gen.compile(
            loss='binary_crossentropy', optimizer=optimizer)

        # Build and compile the discriminators        
        self.dis = self.build_dis()
        self.dis.compile(loss='binary_crossentropy',
                                        optimizer=optimizer,
                                        metrics=['accuracy'])

        optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)

        # Wasserstein Loss
        # Compile Generator Model
        z = Input((self.noise,))
        song = self.gen(z)
        
        self.lstm_discriminator.trainable = False
        
        same = self.dis(song)
        
        self.gen_model = Model(inputs=[z],
                                outputs=[same])
        self.gen_model.compile(optimizer=optimizer,
                                loss=wasserstein_loss)

        # Compile Discriminator Model
        self.dis.trainable = True
        self.gen.trainable = False

        song = Input(shape=self.lstm_shape)
        z = Input(shape=(self.noise,))
        gen_song = self.gen()
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
        self.critic_model = Model(inputs=[song,z],
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
    
    def build_gen_cnn(self):

        k = (1, 35)
                   
        # define CNN model
        ########################
        print("GENERATOR CNN")
        cnn = Sequential()
        
        cnn.add(
            Reshape((self.channels, 1, self.ngf_lstm), 
            input_shape=(self.ngf_lstm,)
        ))

        # Conv 1
        cnn.add(Conv2DTranspose(
                filters=self.ngf_cnn*16,
                kernel_size=k,
                strides=(1, 2),
                padding='same',
                use_bias=False,
                ))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(Activation("relu"))

        # Conv 2
        cnn.add(Conv2DTranspose(
                filters=self.ngf_cnn*8,
                kernel_size=k,
                strides=(1, 2),
                padding='same',
                use_bias=False,
                ))
        cnn.add(ZeroPadding2D(padding=((0, 0), (1, 0))))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(Activation("relu"))

        # Conv 3
        cnn.add(Conv2DTranspose(
                filters=self.ngf_cnn*4,
                kernel_size=k,
                strides=(1, 2),
                padding='same',
                use_bias=False,
                ))
        cnn.add(ZeroPadding2D(padding=((0, 0), (1, 2))))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(Activation("relu"))

        # Conv 4
        cnn.add(Conv2DTranspose(
                filters=self.ngf_cnn*2,
                kernel_size=k,
                strides=(1, 4),
                padding='same',
                use_bias=False,
                ))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(Activation("relu"))

        # Conv 5
        cnn.add(Conv2DTranspose(
                filters=self.ngf_cnn,
                kernel_size=k,
                strides=(1, 4),
                padding='same',
                use_bias=False,
                ))
        cnn.add(ZeroPadding2D(padding=((0, 0), (2, 1))))
        cnn.add(BatchNormalization(momentum=.8))
        cnn.add(Activation("relu"))

        # Final Conv
        cnn.add(Conv2DTranspose(
                filters=2,
                kernel_size=k,
                strides=1,
                padding='same',
                use_bias=False,
        ))
        # cnn.add(Activation("tanh"))
        
        # Inverse Fast Fourier Transformation Layer
        # in (None, 1, 211, 2)  out (None, 1, 420)
        def iFFT(x):
            real, imag = tf.split(x, 2, axis=-1)
            x = tf.complex(real, imag) 
            x = tf.squeeze(x, axis=[-1])
            x = tf.spectral.irfft(x)
            
            return x
        cnn.add(Lambda(iFFT))
        cnn.add(Activation('tanh'))
        
        cnn.summary()

        inp = Input(shape=(self.ngf_lstm,))
        valid = cnn(inp)

        return Model(inp, valid)

    def build_dis_cnn(self):
        
        # define CNN model
        ########################
        print("DISCRIMINATOR CNN")

        k = (1, 30)
        
        cnn = Sequential()

        # Fast Fourier Transformation Layer
        # in: (None, 1, 420)  out: (None, 1, 211, 2)
        def FFT(x):
            x = tf.spectral.rfft(x)
            extended_bin = x[..., None]
            return tf.concat([tf.real(extended_bin), tf.imag(extended_bin)], axis=-1)
        cnn.add(
            Lambda(FFT, input_shape=(self.channels, self.samples_per_slice))
        )
        
        # Conv 1
        cnn.add(Conv2D(
            filters=self.ndf_cnn,
            kernel_size=k,
            strides=(1, 4),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        # Conv 2
        cnn.add(Conv2D(
            filters=self.ndf_cnn*2,
            kernel_size=k,
            strides=(1, 4),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        # Conv 3
        cnn.add(Conv2D(
            filters=self.ndf_cnn*4,
            kernel_size=k,
            strides=(1, 2),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        # Conv 4
        cnn.add(Conv2D(
            filters=self.ndf_cnn*8,
            kernel_size=k,
            strides=(1, 2),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        # Conv 5
        cnn.add(Conv2D(
            filters=self.ndf_cnn*16,
            kernel_size=k,
            strides=(1, 2),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))

        # Conv 6
        cnn.add(Conv2D(
            filters=self.ndf_cnn*32,
            kernel_size=k,
            strides=(1, 2),
            padding='same',
            use_bias='False',
        ))
        cnn.add(BatchNormalization(momentum=0.8))
        cnn.add(LeakyReLU(alpha=0.2))
        
        cnn.add(Flatten())
        cnn.add(Dropout(.2))

        cnn.add(Dense(self.ndf_lstm, activation='tanh'))
        
        cnn.summary()

        inp = Input(shape=(self.channels, self.samples_per_slice))
        valid = cnn(inp)

        return Model(inp, valid)
      
    def build_gen(self, CuDNN=True):
        # define ConvLSTM model
        #########################
        print("GENERATOR LSTM")
        convlstm = Sequential()
        
        # Init Layer
        convlstm.add(
            Dense(self.slices*self.ngf_lstm*4, input_shape=(self.noise,)))
        convlstm.add(Activation('tanh'))
        convlstm.add(Reshape((self.slices, self.ngf_lstm*4)))

        # LSTM 1
        if CuDNN:
            convlstm.add(
                CuDNNLSTM(
                    units=self.ngf_lstm*3,
                    return_sequences=True,
                ))
        else:
            convlstm.add(
                LSTM(
                    units=self.ngf_lstm*3,
                    return_sequences=True,
                ))

        # LSTM 2
        if CuDNN:
            convlstm.add(
                CuDNNLSTM(
                    units=self.ngf_lstm*2,
                    return_sequences=True,
                ))
        else:
            convlstm.add(
                LSTM(
                    units=self.ngf_lstm*2,
                    return_sequences=True,
                ))

        # LSTM 2
        if CuDNN:
            convlstm.add(
                CuDNNLSTM(
                    units=self.ngf_lstm,
                    return_sequences=True,
                ))
        else:
            convlstm.add(
                LSTM(
                    units=self.ngf_lstm,
                    return_sequences=True,
                ))
        
        # Time Distrubute Thru CNN
        convlstm.add(TimeDistributed(self.cnn_generator))
        
        convlstm.summary()

        
        noise = Input(shape=(self.noise,))
        song = convlstm(noise)

        return Model(noise, song)

    def build_dis(self, CuDNN=True):
        # define ConvLSTM model
        #########################
        print("DISCRIMINATOR LSTM")
        
        convlstm = Sequential()

        # Time Distribute Thru CNN
        convlstm.add(TimeDistributed(self.cnn_discriminator, input_shape=self.lstm_shape))
        
        # LSTM 1
        if CuDNN:
            convlstm.add(CuDNNLSTM(
                units=self.ndf_lstm*2,
                return_sequences=True,
                # recurrent_dropout=0.1,
                # use_bias=False,
            ))
        else:
            convlstm.add(LSTM(
                units=self.ndf_lstm*2,
                return_sequences=True,
            ))
        convlstm.add(Dropout(.1))

        # LSTM 2
        if CuDNN:
            convlstm.add(CuDNNLSTM(
                units=self.ndf_lstm*3,
                return_sequences=True,
                # recurrent_dropout=0.1,
                # use_bias=False,
            ))
        else:
            convlstm.add(LSTM(
                units=self.ndf_lstm*3,
                return_sequences=True,
            ))
        convlstm.add(Dropout(.1))

        # LSTM 3
        if CuDNN:
            convlstm.add(CuDNNLSTM(
                units=self.ndf_lstm*4,
                return_sequences=True,
                # recurrent_dropout=0.1,
                # use_bias=False,
            ))
        else:
            convlstm.add(LSTM(
                units=self.ndf_lstm*4,
                return_sequences=True,
            ))

        convlstm.add(Flatten())
        convlstm.add(Dropout(.2))
        convlstm.add(Dense(1, activation="sigmoid"))
        
        convlstm.summary()
        

        four_bars = Input(shape=self.lstm_shape)
        sameity = convlstm(four_bars)

        return Model(four_bars, sameity)

    def lstm_train(self, training_dir, epochs, training_ratio=5, batch_size=16, save_interval=50):

        # ---------------------
        #  Preprocessing
        # ---------------------

        # Load from training_dir and normalize dataset
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
            for _ in range(training_ratio):
                batch_files = np.random.choice(all_file_names, half_batch)
                songs = np.zeros((half_batch,) + self.lstm_shape)
                num_songs = 0

                for filename in batch_files:
                    song = wavfile.read(filename)[1].reshape(self.lstm_shape)
                    songs[num_songs] = song
                    num_songs += 1

                # -1 to 1
                db = max([songs.max(), abs(songs.min())])
                songs = songs / 32767.0
                
                noise = np.random.normal(0, 1, (half_batch, self.noise))
                
                positive_y = np.ones((half_batch, 1))
                negative_y = -positive_y
                dummy_y = np.zeros((half_batch, 1))
                
                d_loss = self.critic_model.train_on_batch([songs, noise],
                                                            [positive_y, negative_y, dummy_y])

            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(
                0, 1, (batch_size, self.noise))

            positive_y = np.ones((batch_size, 1))
            
            g_loss = self.gen_model.train_on_batch(noise, positive_y)

            elapsed_time = datetime.datetime.now() - start_time

            # Plot the progress
            print("%d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f] time: %s" %
                  (epoch+1, epochs, d_loss[0], 100*d_loss[1], g_loss, elapsed_time))

            d_losses.append(d_loss[0])
            g_losses.append(g_loss)

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_beats(epoch, training_dir, 'lstm')

                # Save generator
                self.gen.save('E:/models/beat_gan_lstm_generator.h5')
                self.dis.save('E:/models/beat_gan_lstm_discriminator.h5')
                self.lstm_combined.save('E:/models/beat_gan_lstm_combined.h5')

        # Plot Loss Graph
        plt.plot(range(epochs), d_losses)
        plt.plot(range(epochs), g_losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig('lstm_loss_graph.png')

    def save_beats(self, epoch, dataset, network):
        NUM_BEATS = 10
        
        noise = np.random.normal(0, 1, (NUM_BEATS, self.noise))
        if network is 'cnn':
            gen_beats = self.cnn_generator.predict(noise)
        elif network is 'lstm':
            gen_beats = self.gen.predict(noise)
        else:
            raise Exception()

        # # Rescale images 0 - 1
        # gen_beats = (gen_beats - .5) * 2
        gen_beats = gen_beats.astype(np.float32)

        if not os.path.exists("samples"):
            os.mkdir("samples")

        for i, beat in enumerate(gen_beats):
            beat = np.reshape(beat, (-1, self.channels))
            filename = '%s_%s_%dbpm_epoch%d_%d.wav' % (network, dataset, self.bpm, epoch, i+1)
            try:
                wavfile.write('E:/samples/%s' % filename, self.sample_rate, beat)
            except:
                print("x")

def parse_command_line_args():
    parser = argparse.ArgumentParser(description='AI Generated Beats Bitch')
    parser.add_argument('epochs', type=int,
                        help='number of epochs')
    parser.add_argument('training_dir', type=str,
                        help='filepath of training set')
    parser.add_argument('-t', '--tempo', type=int, default=120,
                        help='Tempo of song output')
    parser.add_argument('-r', '--rnnsize', type=int, default=210,
                        help='size of recurrent layer')
    parser.add_argument('-c', '--cnnsize', type=int, default=420,
                        help='size of convolutional layer')
    parser.add_argument('-b', '--batchsize',
                        default=16, type=int, help='size of batches per epoch')
    parser.add_argument('-s', '--saveinterval',
                        type=int, default=1000, help='interval to save sample images')
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_command_line_args()
    bg = BeatGAN(args['tempo'],
                 args['rnnsize'],
                 args['cnnsize'])
    bg.lstm_train(training_dir=args['training_dir'],
                  epochs=args['lstm_epochs'],
                  batch_size=args['batchsize'],
                  save_interval=args['saveinterval'])
