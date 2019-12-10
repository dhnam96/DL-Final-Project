import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Conv2d, Flatten, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from preprocess import get_data
from imageio import imwrite
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='GAN')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--out-dir', type=str, default='./output',
                    help='Data where sampled output images will be written')
parser.add_argument('--img-width', type=int, default=256,
                    help='Width of images in pixels')
parser.add_argument('--img-height', type=int, default=256,
                    help='Height of images in pixels')
args = parser.parse_args()

def sample(m, logsigma):
    eps = tf.random.normal(tf.shape(m), .0, 1.0)
    return m + tf.math.exp(logsigma / 2) * eps

def kullback_leibler(m, logsigma):
    return -tf.reduce_sum(logsigma - tf.math.pow(m, 2) - tf.math.exp(logsigma) + 1)/2

class Encoder(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel, dense_out):
        super(Encoder, self).__init__()

        # Sequential Encoder Layers
        self.encoder_model = Sequential()
        self.encoder_model.add(Conv2d(filters = filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 2*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 4*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 8*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Flatten())

        # Intermediate Layers:
        self.mean = tf.keras.layers.Dense(dense_out)
        self.logsigma = tf.keras.layers.Dense(dense_out, activation="tanh")

    @tf.function
    def call(self, inputs):
        intermediate_output = self.encoder_model(inputs)
        mean = self.mean(intermediate_output)
        logsigma = self.logsigma(intermediate_output)
        encoder_output = sample(mean, logsigma)
        return mean, logsigma, encoder_output
    
    def loss_function(self, _):
        # TODO


class Decoder(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel, dense_out):
        super(Decoder, self).__init__()

        # Sequential Decoder Layers:
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(8*filter_size*args.img_width*args.img_height))
        self.decoder_model.add(Reshape((args.img_width, args.img_height, 8*filter_size)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 4*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 2*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = channel, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(Activation('tanh'))
    
    @tf.function
    def call(self, inputs):
        return self.decoder_model(inputs)
    
    def loss_function(self, )


class Discriminator(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel):
        super(Discriminator, self).__init__()

        # TODO


        # Define loss
        self.real_loss = tf.keras.losses.BinaryCrossentropy()
        self.fake_loss = tf.keras.losses.BinaryCrossentropy()
        self.tilde_loss = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        # TODO

        return middle_conv, output

    def loss_function(self, disc_real_output, disc_fake_output, disc_tilde_output):
        return self.real_loss(tf.ones_like(disc_real_output), disc_real_output) + \
            self.fake_loss(tf.zeros_like(disc_fake_output), disc_fake_output) + \
            self.tilde_loss(tf.zeros_like(disc_tilde_output), disc_tilde_output)



