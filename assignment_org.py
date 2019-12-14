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
        self.filter_size = 32
        self.kernel_size = 5
        self.channel = 3
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)
        self.dense_out = 512

        # Sequential Encoder Layers
        self.encoder_model = Sequential()
        self.encoder_model.add(Conv2d(filters = self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 2*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 4*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Conv2d(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))
        self.encoder_model.add(Flatten())

        # Intermediate Layers:
        self.mean = tf.keras.layers.Dense(self.dense_out)
        self.logsigma = tf.keras.layers.Dense(self.dense_out, activation="tanh")

        # Sequential Decoder Layers:
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(8*self.filter_size*args.img_width*args.img_height))
        self.decoder_model.add(Reshape((args.img_width, args.img_height, 8*filter_size)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 4*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = 2*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(Activation('relu'))
        self.decoder_model.add(Conv2DTranspose(filter = self.channel, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(Activation('tanh'))

    def sample(m, logsigma):
        eps = tf.random.normal(tf.shape(m), .0, 1.0)
        return m + tf.math.exp(logsigma / 2) * eps

    def kullback_leibler(self, m, logsigma):
        return -tf.reduce_sum(logsigma - tf.math.pow(m, 2) - tf.math.exp(logsigma) + 1)/2

    def encode(self, inputs):
        intermediate_output = self.encoder_model(inputs)
        mean = self.mean(intermediate_output)
        logsigma = self.logsigma(intermediate_output)
        encoder_output = sample(mean, logsigma)
        return mean, logsigma, encoder_output

class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()

        # Hyperparameters
        self.filter_size = 64
        self.kernel_size = 5
        self.channel = 1
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)

        # Layers
        # self.conv1 = tf.keras.layers.Conv2d(filters = 5, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="relu")
        # self.conv2 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 4*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.conv3 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.conv4 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        # self.conv5 = tf.keras.layers.Conv2d(filters = 5, kernel_size = 1, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="sigmoid")

        # Sequential Discriminator Model
        self.discrim_model = Sequential()
        self.discrim_model.add(Conv2d(filters = self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(Activation('relu'))
        self.discrim_model.add(Conv2d(filters = 2*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(Conv2d(filters = 4*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(Conv2d(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(Conv2d(filters = self.channel, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(Activation('sigmoid'))

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

    def sequential_call(inputs):
        output = self.discrim_model(inputs)
        return output

    def loss(discrim_real, discrim_fake):
        self.bce = tf.keras.losses.BinaryCrossEntropy()
        predict_fake = tf.reduce_mean(bce(y_true = tf.zeros_like(discrim_fake), y_pred=discrim_fake))
        predict_real = tf.reduce_mean(bce(y_true = tf.ones_like(deiscrim_real), y_pred=discrim_real))
        return tf.reduce_mean(-(tf.log(predict_real) + tf.log(1-predict_fake)))

def train(generator, discriminator, train_data):
    ### BATCH THIS
    batched_train = train_data
    ### DO PREPROCESS ON TRAIN LIKE SHUFFLE AND FLIP AND BATCH THEM
    enc_mean, enc_logsigma, enc_Z = generator.encode(batched_train)
    output = generator.decoder_model(enc_Z)
    decoded_sample = generator.decoder_model(enc_mean + enc_logsigma)
    discrim_fake = discriminator.discrim_model(output)
    discrim_gen = discriminator.discrim_model(decoded_sample)
    discrim_real = discriminator.discrim_model(batched_train)


    pass

def test():

    pass

def crop_img(images, x, y):
    images_copy = np.copy(images)
    images_copy[:, y:, x:, :] = 0.0
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