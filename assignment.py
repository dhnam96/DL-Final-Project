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

class Generator_Model(tf.keras.Model):
    def __init__(self):
        super(Generator_Model, self).__init__()

        # Encoder Layers:
        # self.conv1 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.batch_norm1 = tf.keras.layers.BatchNormalization(epsilon = 1e-5)
        # self.leaky1 = tf.keras.layers.LeakyReLU(alpha = 0.2)
        # self.conv2 = tf.keras.layers.Conv2d(filters = 10, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.batch_norm2 = tf.keras.layers.BatchNormalization(epsilon = 1e-5)
        # self.leaky2 = tf.keras.layers.LeakyReLU(alpha = 0.2)
        # self.conv3 = tf.keras.layers.Conv2d(filters = 20, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.batch_norm3 = tf.keras.layers.BatchNormalization(epsilon = 1e-5)
        # self.leaky3 = tf.keras.layers.LeakyReLU(alpha = 0.2)
        # self.conv4 = tf.keras.layers.Conv2d(filters = 40, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.batch_norm4 = tf.keras.layers.BatchNormalization(epsilon = 1e-5)
        # self.leaky4 = tf.keras.layers.LeakyReLU(alpha = 0.2)
        # self.flatten = tf.keras.layers.Flatten()
        # self.mean = tf.keras.layers.Dense(512)
        # self.logsigma = tf.keras.layers.Dense(512, activation="tanh")

        # Hyperparameters
        self.filter_size = 5
        self.channel = 3
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)

        # Sequential Encoder Layers
        self.encoder_model = Sequential()
        self.encoder_model.add(Conv2d(filters = self.filter_size, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 2*self.filter_size, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 4*self.filter_size, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 8*self.filter_size, kernel_size = 32, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Flatten())

        # Sequential Decoder Layers:
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(8*self.filter_size*args.img_width*args.img_height))
        self.decoder_model.add(Reshape((args.img_width, args.img_height, 8*filter_size)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 4*self.filter_size, kernel_size = 5, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 2*self.filter_size, kernel_size = 5, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = self.filter_size, kernel_size = 5, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = self.channel, kernel_size = 5, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(Activation('tanh'))


    def encoder(batch_img):
        l1 = self.conv1(batch_img)
        l2 = self.batch_norm1(l1)
        l3 = self.leaky1(l2)
        l4 = self.conv2(l3)
        l5 = self.batch_norm2(l4)
        l6 = self.leaky2(l5)
        l7 = self.conv3(l6)
        l8 = self.batch_norm3(l7)
        l9 = self.leaky3(l8)
        l10 = self.conv4(l9)
        l11 = self.batch_norm4(l10)
        l12 = self.leaky4(l11)
        flattened = self.flatten(l12)
        mean = self.mean(flattened)
        logsigma = self.logsigma(flattened)
        return logsigma

    def decoder(encoder_output):
        pass



class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()

        # Kernel size
        self.kernel_size = 64
        # Layers
        self.conv1 = tf.keras.layers.Conv2d(filters = 5, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="relu")
        self.conv2 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 4*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv3 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv4 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv5 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 1, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="sigmoid")
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(lr = 2e-4, beta_1 = 0.5)

    def call(inputs):
        """
        This method does the forward pass for the discriminator model generation.
        """
        first_out = self.conv1(inputs)
        second_out = self.conv2(first_out)
        third_out = self.conv3(second_out)
        fourth_out = self.conv4(third_out)
        output = self.conv5(fourth_out)
        return output

    def loss(discrim_real, discrim_fake):
        self.bce = tf.keras.losses.BinaryCrossEntropy()
        predict_fake = tf.reduce_mean(bce(y_true = tf.zeros_like(discrim_fake), y_pred=discrim_fake))
        predict_real = tf.reduce_mean(bce(y_true = tf.ones_like(deiscrim_real), y_pred=discrim_real))
        return tf.reduce_mean(-(tf.log(predict_real) + tf.log(1-predict_fake)))

def train():

    pass

def test():

    pass

def crop_img(images, x, y):
    images[:, y:, x:, :] = 0.0
    return images


def test(images):
    for i in range(0, args.batch_size):
        img_i = np.array(images[i]).astype(np.uint8)
        s = args.out_dir+'/'+str(i)+'.jpg'
        imwrite(s, img_i)

def main():
    train_data = get_data('./cars_train/preprocessed', resize=False)
    #test_data = get_data('./cars_test/preprocessed', resize=False)
    cropped = crop_img(np.array(train_data[:args.batch_size]), int(args.img_width/2), int(args.img_height/2))

    test(cropped)
if __name__ == '__main__':
   main()