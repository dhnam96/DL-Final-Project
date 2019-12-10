import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Conv2d, Flatten, Reshape, Conv2DTranspose
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
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
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--out-dir', type=str, default='./output',
                    help='Data where sampled output images will be written')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')
args = parser.parse_args()

def sample(m, logsigma):
    eps = tf.random.normal(tf.shape(m), .0, 1.0)
    return m + tf.math.exp(logsigma / 2) * eps

def kullback_leibler_loss(m, logsigma):
    return -tf.reduce_sum(logsigma - tf.math.pow(m, 2) - tf.math.exp(logsigma) + 1)/2

def latent_layer_loss(feature_real, feature_tilde):
    return -tf.reduce_mean(tf.reduce_sum(tf.square(feature_fake - feature_real), [1,2,3]))


class Encoder(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel):
        super(Encoder, self).__init__()

        # Variables
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.channel = channel

        # Hyperparameters:
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)

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
        self.mean = Dense(channel)
        self.logsigma = Dense(channel, activation="tanh")

    @tf.function
    def call(self, inputs):
        intermediate_output = self.encoder_model(inputs)
        mean = self.mean(intermediate_output)
        logsigma = self.logsigma(intermediate_output)
        encoder_output = sample(mean, logsigma)
        return mean, logsigma, encoder_output

    def loss_function(self, kl_loss, latent_loss):
        return kl_loss/(self.channel*args.batch_size) - latent_loss

class Decoder(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel):
        super(Decoder, self).__init__()

        # Variables
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.channel = channel

        # Hyperparameters:
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)

        # Sequential Decoder Layers:
        self.decoder_model = Sequential()
        self.decoder_model.add(Dense(8*self.filter_size*args.img_width*args.img_height))
        self.decoder_model.add(Reshape((args.img_width, args.img_height, 8*self.filter_size)))
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

        self.fake_loss = BinaryCrossentropy()
        self.tilde_loss = BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        return self.decoder_model(inputs)

    def loss_function(self, disc_fake_output, disc_tilde_output, latent_loss):
        return self.fake_loss(tf.zeros_like(disc_fake_output), disc_fake_output) + \
            self.tilde_loss(tf.zeros_like(disc_tilde_output), disc_tilde_output) - 1e-6 * latent_loss


class Discriminator(tf.keras.Model):
    def __init__(self, filter_size, kernel_size, channel):
        super(Discriminator, self).__init__()

        # Variables
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.channel = channel

        # Hyperparameters:
        self.optimizer = Adam(lr = 2e-4, beta_1 = 0.5)

        # Feature
        self.discrim_model = Sequential()
        self.discrim_model.add(Conv2d(filters = 2*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))
        self.discrim_model.add(Conv2d(filters = 4*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(BatchNormalization(epsilon = 1e-5))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))
        self.discrim_model.add(Conv2d(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(BatchNormalization(epsilon = 1e-5))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))
        self.discrim_model.add(Conv2d(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))

        # Additional Layers to pass through after the sequential model
        self.batch_norm = BatchNormalization(epsilon = 1e-5)
        self.leaky_relu = LeakyReLU(alpha = 0.2)
        self.flatten = Flatten()
        self.dense = Dense(self.channel, activation = 'sigmoid')

        # Define loss
        self.real_loss = BinaryCrossentropy()
        self.fake_loss = BinaryCrossentropy()
        self.tilde_loss = BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        middle_conv = self.discrim_model(inputs)
        output = self.batch_norm(features)
        output = self.leaky_relu(output)
        output = self.flatten(output)
        output = self.dense(output)
        return middle_conv, output

    def loss_function(self, disc_real_output, disc_fake_output, disc_tilde_output):
        return self.real_loss(tf.ones_like(disc_real_output), disc_real_output) + \
            self.fake_loss(tf.zeros_like(disc_fake_output), disc_fake_output) + \
            self.tilde_loss(tf.zeros_like(disc_tilde_output), disc_tilde_output)

def train(encoder, decoder, discriminator, real_images, cropped):
    # here images should be a numpy array
    for x in range(0, int(images.shape[0]/args.batch_size)):
        batch_real = real_images[x*args.batch_size: (x+1)*args.batch_size]
        batch_cropped = cropped[x*args.batch_size: (x+1)*args.batch_size]

        with tf.GradientTape() as enc_tape, tf.GraidentTape() as dec_tape, tf.GradientTape() as disc_tape:
            mean, logsigma, enc_out = encoder.call(batch_cropped)
            zp = tf.random_normal(shape=enc_out.shape)
            dec_out = decoder.call(enc_out)
            dec_noise = decoder.call(zp)
            feature_tilde, disc_tilde_out = discriminator.call(dec_out)
            feature_real, disc_real_out = discriminator.call(batch_real)
            feature_fake, disc_fake_out = discriminator.call(dec_noise)
            kl_loss = kullback_leibler_loss(mean, logsigma)
            ll_loss = latent_layer_loss(feature_real, feature_tilde)
            enc_loss = encoder.loss(kl_loss, ll_loss)
            dec_loss = decoder.loss(disc_fake_out, disc_tilde_out, ll_loss)
            disc_loss = discriminator.loss(disc_real_out, disc_fake_out, disc_tilde_out)
        enc_grads = enc_tape.gradient(enc_loss, encoder.trainable_variables)
        dec_grads = dec_tape.gradient(dec_loss, decoder.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        encoder.optimizer.apply_gradients(zip(enc_grads, encoder.trainable_variables))
        decoder.optimizer.apply_gradients(zip(dec_grads, decoder.trainable_variables))
        discriminator.optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

        if x % 10 == 0:
            print("Training %3.3f percent complete" % (100*x/(images.shape[0]/args.batch_size)))
            print("Encoder Loss:")
            print(enc_loss)
            print("Decoder Loss:")
            print(dec_loss)
            print("Discriminator Loss:")
            print(disc_loss)


def test():
    pass

def crop_img(images, x, y):
    images_copy = np.copy(images)
    images_copy[:, y:, x:, :] = 0.0
    return images


def main():
    # Get data
    train_data = get_data('./cars_train/preprocessed', resize=False)
    cropped = crop_img(np.array(train_data[:args.batch_size]), int(args.img_width/2), int(args.img_height/2))

    # Initialize model
    encoder = Encoder(_, )
    decoder = Decoder(_, )
    discriminator = Discriminator(_, )

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint or args.mode == 'test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint) 

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    train(generator, decoder, discriminator, images)
                    # print("Average FID for Epoch: " + str(avg_fid))
                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
            if args.mode == 'test':
                test(encoder, decoder)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()