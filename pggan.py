# example of progressive growing gan on celebrity faces dataset
from math import sqrt
import glob
import os
from numpy import load
from numpy import asarray
from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import AveragePooling2D
from keras.layers import LeakyReLU
from keras.layers import Layer
from keras.layers import Add
from keras.constraints import max_norm
from keras.initializers import RandomNormal
from keras import backend
from matplotlib import pyplot
from scipy.io import wavfile
import numpy as np

# Load from training_dir and normalize dataset
# all_file_names = glob.glob(
#     'datasets/%s/%dbpm/slices/*.wav' % (training_dir, self.bpm))
# all_file_names = glob.glob(
#     'E:/datasets/%s/%dbpm/slices/*.wav' % (training_dir, self.bpm))

class PixelNormalization(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs**2.0
        # calculate the mean pixel values
        mean_values = backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape

class MinibatchStdev(Layer):
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = backend.shape(inputs)
        output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)

class WeightedSum(Add):
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = backend.variable(alpha, name='ws_alpha')

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

def wasserstein_loss(y_true, y_pred):
    return backend.mean(y_true * y_pred)


class PGGAN():
    def __init__(self, external=False):
        self.training_dir = 'yung_gan'
        self.bpm = 120
        
        self.file_names = dict()
        for rate in self.sample_rates:
            self.file_names[rate] = glob.glob(
                'datasets/%s/%dbpm/slices/%d/*.wav' % (self.training_dir, self.bpm, rate))

        # number of growth phases
        self.sample_rates = [64, 256, 1024, 4096, 16384, 65536]
        self.n_blocks = len(self.sample_rates)
        # size of the latent space
        self.latent_dim = 100
        # define dis models
        self.d_models = self.define_discriminator()
        # define gen models
        self.g_models = self.define_generator()
        # define composite models
        self.gan_models = self.define_composite()
        
        self.n_batch = [16, 16, 16, 8, 4, 4]
        # 10 epochs == 500K images per training phase
        self.n_epochs = [5, 8, 8, 10, 10, 10]

    def add_discriminator_block(self, old_model, n_input_layers=3):
        # add a discriminator block

        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)

        # get shape of existing model
        in_shape = list(old_model.input.shape)
        # define new input shape as double the size
        input_shape = (in_shape[-3].value*4,
                    in_shape[-2].value, in_shape[-1].value)
        in_image = Input(shape=input_shape)

        # define new input processing layer
        d = Conv2D(128, (3, 1), padding='same', kernel_initializer=init,
                kernel_constraint=const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # define new block
        d = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = AveragePooling2D(pool_size=(4, 1))(d)
        block_new = d
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model1 = Model(in_image, d)
        # compile model
        model1.compile(loss=wasserstein_loss, optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))

        # downsample the new larger image
        downsample = AveragePooling2D(pool_size=(4, 1))(in_image)
        # connect old input processing to downsampled new input
        block_old = old_model.layers[1](downsample)
        block_old = old_model.layers[2](block_old)
        # fade in output of old model input layer with new input
        d = WeightedSum()([block_old, block_new])
        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(old_model.layers)):
            d = old_model.layers[i](d)
        # define straight-through model
        model2 = Model(in_image, d)
        # compile model
        model2.compile(loss=wasserstein_loss, optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        return [model1, model2]

    def define_discriminator(self, input_shape=(64, 1, 1)):
        # define the discriminator models for each image resolution

        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        model_list = list()
        # base model input
        in_image = Input(shape=input_shape)
        # conv 1x1
        d = Conv2D(128, (3, 1), padding='same', kernel_initializer=init,
                kernel_constraint=const)(in_image)
        d = LeakyReLU(alpha=0.2)(d)
        # conv 3x3 (output block)
        d = MinibatchStdev()(d)
        d = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # conv 4x4
        d = Conv2D(128, (12, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(d)
        d = LeakyReLU(alpha=0.2)(d)
        # dense output layer
        d = Flatten()(d)
        out_class = Dense(1)(d)
        # define model
        model = Model(in_image, out_class)
        # compile model
        model.compile(loss=wasserstein_loss, optimizer=Adam(
            lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
        # store model
        model_list.append([model, model])
        # create submodels
        for i in range(1, self.n_blocks):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = self.add_discriminator_block(old_model)
            # store model
            model_list.append(models)
        return model_list

    def add_generator_block(self, old_model):
        # add a generator block

        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        # get the end of the last block
        block_end = old_model.layers[-2].output
        # upsample, and define new block
        upsampling = UpSampling2D(size=(4, 1))(block_end)
        g = Conv2D(128, (9, 1), padding='same', kernel_initializer=init,
                kernel_constraint=const)(upsampling)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        g = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # add new output layer
        out_image = Conv2D(1, (3, 1), padding='same',
                        kernel_initializer=init, kernel_constraint=const)(g)
        # define model
        model1 = Model(old_model.input, out_image)
        # get the output layer from old model
        out_old = old_model.layers[-1]
        # connect the upsampling to the old output layer
        out_image2 = out_old(upsampling)
        # define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([out_image2, out_image])
        # define model
        model2 = Model(old_model.input, merged)
        return [model1, model2]

    def define_generator(self, in_dim=64):
        # define generator models

        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = max_norm(1.0)
        model_list = list()
        
        # base model latent input
        in_latent = Input(shape=(self.latent_dim,))
        
        # linear scale up to activation maps
        g = Dense(128 * in_dim, kernel_initializer=init,
                kernel_constraint=const)(in_latent)
        g = Reshape((in_dim, 1, 128))(g)
        # conv 4x4, input block
        g = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # conv 3x3
        g = Conv2D(128, (9, 1), padding='same',
                kernel_initializer=init, kernel_constraint=const)(g)
        g = PixelNormalization()(g)
        g = LeakyReLU(alpha=0.2)(g)
        # conv 1x1, output block
        out_image = Conv2D(1, (3, 1), padding='same',
                        kernel_initializer=init, kernel_constraint=const)(g)
        
        # define model
        model = Model(in_latent, out_image)
        # store model
        model_list.append([model, model])

        # create submodels
        for i in range(1, self.n_blocks):
            # get prior model without the fade-on
            old_model = model_list[i - 1][0]
            # create new model for next resolution
            models = self.add_generator_block(old_model)
            # store model
            model_list.append(models)
        return model_list

    def define_composite(self):
        # define composite models for training generators via discriminators

        model_list = list()
        # create composite models
        for i in range(len(self.d_models)):
            g_models, d_models = self.g_models[i], self.d_models[i]
            
            # straight-through model
            d_models[0].trainable = False
            model1 = Sequential()
            model1.add(g_models[0])
            model1.add(d_models[0])
            model1.compile(loss=wasserstein_loss, optimizer=Adam(
                lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            
            # fade-in model
            d_models[1].trainable = False
            model2 = Sequential()
            model2.add(g_models[1])
            model2.add(d_models[1])
            model2.compile(loss=wasserstein_loss, optimizer=Adam(
                lr=0.001, beta_1=0, beta_2=0.99, epsilon=10e-8))
            # store
            model_list.append([model1, model2])
        return model_list

    def generate_real_samples(self, n_samples, shape=(65536, 1, 1)):
        # select real samples
        X = np.zeros((n_samples,) + shape)
        
        num_songs = 0
        while num_songs < n_samples:
            filename = np.random.choice(self.file_names[shape[0]])
            song = wavfile.read(filename)[1].reshape(shape)
            # Check to make sure slice is not empty
            if not np.array_equal(song, np.zeros(shape)):
                X[num_songs] = song
                num_songs += 1

        # -1 to 1
        X = X / 32768.0
        
        # generate class labels
        y = np.ones((n_samples, 1))
        
        return X, y

    def generate_latent_points(self, n_samples):
        # generate points in latent space as input for the generator

        # generate points in the latent space
        x_input = randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(n_samples, self.latent_dim)
        return x_input

    def generate_fake_samples(self, generator, n_samples):
        # use the generator to generate n fake examples, with class labels

        # generate points in latent space
        x_input = self.generate_latent_points(n_samples)
        # predict outputs
        X = generator.predict(x_input)
        # create class labels
        y = -ones((n_samples, 1))
        return X, y

    def update_fadein(self, models, step, n_steps):
        # update the alpha value on each instance of WeightedSum

        # calculate current alpha (linear from 0 to 1)
        alpha = step / float(n_steps - 1)
        # update the alpha for each model
        for model in models:
            for layer in model.layers:
                if isinstance(layer, WeightedSum):
                    backend.set_value(layer.alpha, alpha)

    def train_epochs(self, g_model, d_model, gan_model, scale, n_epochs, n_batch, fadein=False):
        # train a generator and discriminator

        # calculate the number of batches per training epoch
        bat_per_epo = int(scale[0] / n_batch)
        # calculate the number of training iterations
        n_steps = bat_per_epo * n_epochs
        # calculate the size of half a batch of samples
        half_batch = int(n_batch / 2)
        # manually enumerate epochs
        for i in range(n_steps):
            # update alpha for all WeightedSum layers when fading in new blocks
            if fadein:
                self.update_fadein([g_model, d_model, gan_model], i, n_steps)
        
        # prepare real and fake samples
        X_fake, y_fake = self.generate_fake_samples(g_model, half_batch)
        X_real, y_real = self.generate_real_samples(half_batch, scale)

        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update the generator via the discriminator's error
        z_input = self.generate_latent_points(n_batch)
        y_real2 = ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(z_input, y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))

    def train(self):
        # train the generator and discriminator

        # fit the baseline model
        g_normal, d_normal, gan_normal = self.g_models[0][0], self.d_models[0][0], self.gan_models[0][0]

        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        shape = gen_shape[1:]
        print('Scaled Data', shape)

        # train normal or straight-through models
        self.train_epochs(
            g_normal,
            d_normal,
            gan_normal,
            shape,
            self.n_epochs[0],
            self.n_batch[0])
        self.summarize_performance('tuned', g_normal)

        # process each level of growth
        for i in range(1, len(self.g_models)):
            # retrieve models for this level of growth
            [g_normal, g_fadein] = self.g_models[i]
            [d_normal, d_fadein] = self.d_models[i]
            [gan_normal, gan_fadein] = self.gan_models[i]

            # scale dataset to appropriate size
            gen_shape = g_normal.output_shape[1:]
            print('Scaled Data', gen_shape)

            # train fade-in models for next level of growth
            self.train_epochs(
                g_fadein,
                d_fadein,
                gan_fadein,
                gen_shape,
                self.n_epochs[i],
                self.n_batch[i],
                True)
            self.summarize_performance('faded', g_fadein)

            # train normal or straight-through models
            self.train_epochs(g_normal, d_normal, gan_normal,
                        gen_shape, self.n_epochs[i], self.n_batch[i])
            self.summarize_performance('tuned', g_normal)

    def summarize_performance(self, status, g_model, n_samples=10):
        # generate samples and save as a plot and save the model
        # devise name
        gen_shape = g_model.output_shape
        name = '%03d-%s' % (gen_shape[1], status)
        
        # generate images
        X, _ = self.generate_fake_samples(g_model, n_samples)
        
        if not os.path.exists("samples"):
            os.mkdir("samples")

        for i, beat in enumerate(X):
            beat = np.reshape(beat, (-1, 1))
            filename1 = '%s_%d_%s_%d.wav' % (
                self.training_dir, gen_shape[1], status, i+1)
            try:
                wavfile.write('samples/%s' %
                              filename1, gen_shape[1]//2, beat)
            except:
                print("x")
        # save the generator model
        filename2 = 'model_%s.h5' % (name)
        g_model.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))

if __name__ == "__main__":
    gan = PGGAN()
    gan.train()

