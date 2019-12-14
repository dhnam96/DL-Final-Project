import matplotlib
matplotlib.use('Agg')
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU, BatchNormalization, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from preprocess import get_data
from imageio import imwrite
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)
parser = argparse.ArgumentParser(description='GAN')
parser.add_argument('--batch-size', type=int, default=256,
                    help='Sizes of image batches fed through the network')
parser.add_argument('--out-dir', type=str, default='./output',
                    help='Data where sampled output images will be written')
parser.add_argument('--img-width', type=int, default=64,
                    help='Width of images in pixels')
parser.add_argument('--img-height', type=int, default=64,
                    help='Height of images in pixels')
parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')
parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')
parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')
parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')
parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')
parser.add_argument('--num-gen-updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')
parser.add_argument('--lambdas', type=float, default=1000,
                    help='Height of images in pixels')
args = parser.parse_args()

def sample(m, logsigma):
    eps = tf.random.normal(tf.shape(m), .0, 1.0)
    return m + tf.math.exp(logsigma / 2) * eps

def kullback_leibler_loss(m, logsigma):
    return -tf.reduce_sum(logsigma - tf.math.pow(m, 2) - tf.math.exp(logsigma) + 1)/2

def latent_layer_loss(feature_real, feature_tilde):
    return tf.reduce_mean(tf.reduce_sum(-tf.square(feature_tilde - feature_real), [1,2,3])/2 + (-0.5 * tf.math.log(2*np.pi)))

########################## FID ##########################
module = tf.keras.Sequential([hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/classification/4", output_shape=[1001])])
def fid_function(real_image_batch, generated_image_batch):
    """
    Given a batch of real images and a batch of generated images, this function pulls down a pre-trained inception
    v3 network and then uses it to extract the activations for both the real and generated images. The distance of
    these activations is then computed. The distance is a measure of how "realistic" the generated images are.

    :param real_image_batch: a batch of real images from the dataset, shape=[batch_size, height, width, channels]
    :param generated_image_batch: a batch of images generated by the generator network, shape=[batch_size, height, width, channels]

    :return: the inception distance between the real and generated images, scalar
    """
    INCEPTION_IMAGE_SIZE = (299, 299)
    real_resized = tf.image.resize(real_image_batch, INCEPTION_IMAGE_SIZE)
    fake_resized = tf.image.resize(generated_image_batch, INCEPTION_IMAGE_SIZE)
    module.build([None, 299, 299, 3])
    real_features = module(real_resized)
    fake_features = module(fake_resized)
    return tfgan.eval.frechet_classifier_distance_from_activations(real_features, fake_features)
########################## FID ##########################

########################## Encoder ##########################
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
        self.encoder_model.add(Conv2D(filters = filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))

        self.encoder_model.add(Conv2D(filters = 2*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))

        self.encoder_model.add(Conv2D(filters = 4*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))

        self.encoder_model.add(Conv2D(filters = 8*filter_size, kernel_size = kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.encoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.encoder_model.add(LeakyReLU(alpha = 0.2))

        self.encoder_model.add(Flatten())
        self.encoder_model.add(Dense(channel, activation='tanh'))

        # Intermediate Layers:
        # self.mean = Dense(channel)
        # self.logsigma = Dense(channel, activation="tanh")

    @tf.function
    def call(self, inputs):
        return self.encoder_model(inputs)

    @tf.function
    def loss_function(self, pred_patch, true_patch, disc_fake_output):
        return self.loss(tf.ones_like(disc_fake_output), disc_fake_output) + \
            tf.reduce_mean(tf.abs(pred_patch - true_patch)) * args.lambdas


########################## Decoder ##########################
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
        self.decoder_model.add(Dense(8*self.filter_size*args.img_width*args.img_height/16/16))
        self.decoder_model.add(Reshape((int(args.img_width/16), int(args.img_height/16), 8*self.filter_size)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(LeakyReLU(alpha = 0.2))

        self.decoder_model.add(Conv2DTranspose(filters = 4*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(LeakyReLU(alpha = 0.2))

        self.decoder_model.add(Conv2DTranspose(filters = 2*self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(LeakyReLU(alpha = 0.2))

        self.decoder_model.add(Conv2DTranspose(filters = self.filter_size, kernel_size = self.kernel_size, strides = [2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.decoder_model.add(BatchNormalization(epsilon = 1e-5))
        self.decoder_model.add(LeakyReLU(alpha = 0.2))

        self.decoder_model.add(Conv2DTranspose(filters = self.channel, kernel_size = self.kernel_size, strides = [2, 2], padding="same", activation='tanh', kernel_initializer = tf.random_normal_initializer(0, 0.02)))

        self.fake_loss = BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        return self.decoder_model(inputs)

    @tf.function
    def loss_function(self, disc_fake_output):
        return self.fake_loss(tf.ones_like(disc_fake_output), disc_fake_output)

########################## Discriminator ##########################
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
        self.discrim_model.add(Conv2D(filters = 2*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))

        self.discrim_model.add(Conv2D(filters = 4*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(BatchNormalization(epsilon = 1e-5))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))

        self.discrim_model.add(Conv2D(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="same", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(BatchNormalization(epsilon = 1e-5))
        self.discrim_model.add(LeakyReLU(alpha = 0.2))

        self.discrim_model.add(Conv2D(filters = 8*self.filter_size, kernel_size = self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02)))
        self.discrim_model.add(BatchNormalization())
        self.discrim_model.add(LeakyReLU(alpha=0.2))

        self.discrim_model.add(Flatten())
        self.discrim_model.add(Dense(self.channel, activation='sigmoid'))

        # Additional Layers to pass through after the sequential model
        # self.batch_norm = BatchNormalization(epsilon = 1e-5)
        # self.leaky_relu = LeakyReLU(alpha = 0.2)
        # self.flatten = Flatten()
        # self.dense = Dense(self.channel, activation = 'sigmoid')

        # Define loss
        self.real_loss = BinaryCrossentropy()
        self.fake_loss = BinaryCrossentropy()

    @tf.function
    def call(self, inputs):
        return self.discrim_model(inputs)

    @tf.function
    def loss_function(self, disc_real_output, disc_fake_output):
        return self.real_loss(tf.ones_like(disc_real_output), disc_real_output) + \
            self.fake_loss(tf.zeros_like(disc_fake_output), disc_fake_output)

def train(decoder, discriminator, real_images, channel, manager):
    fid_list = []
    dec_loss_list = []
    disc_loss_list = []

    # here images should be a numpy array
    for x in range(0, int(real_images.shape[0]/args.batch_size)):
        batch_real = real_images[x*args.batch_size: (x+1)*args.batch_size]
        random_noise = np.random.uniform(-1, 1, size=(args.batch_size, channel)).astype(np.float32)

        with tf.GradientTape() as dec_tape, tf.GradientTape() as disc_tape:
            pred_img = decoder.call(random_noise)
            disc_fake = discriminator.call(pred_img)
            disc_real = discriminator.call(batch_real)

            dec_loss = decoder.loss_function(disc_fake)
            disc_loss = discriminator.loss_function(disc_real, disc_fake)
        # Optimize discriminator
        if x % args.num_gen_updates == 0:
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            discriminator.optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        # Optimize generator
        dec_gradients = dec_tape.gradient(dec_loss, decoder.trainable_variables)
        decoder.optimizer.apply_gradients(zip(dec_gradients, decoder.trainable_variables))

        # Save
        if x % args.save_every == 0:
            manager.save()

        # print("Training %d/%d complete" % (x, int(real_images.shape[0]/args.batch_size)) )
        if x % 10 == 0 and x > 0:
            print("Training %3.3f percent complete" % (100*x/(real_images.shape[0]/args.batch_size)))
            print("Decoder Loss:")
            print(dec_loss)
            print("Discriminator Loss:")
            print(disc_loss)
            fid_ = fid_function(batch_real, pred_img)
            fid_list.append(fid_.numpy())
            print("FID score:")
            print(fid_)

        dec_loss_list.append(dec_loss.numpy())
        disc_loss_list.append(disc_loss.numpy())

    return dec_loss_list, disc_loss_list, fid_list

# Train the model for one epoch.
def train_encoder(encoder, decoder, discriminator, real_images, mask):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return: The average FID score over the epoch
    """
    # Loop over our data until we run out
    fid_list = []
    loss_list = []

    # here images should be a numpy array
    for x in range(0, int(real_images.shape[0]/args.batch_size)):
        batch_real = real_images[x*args.batch_size: (x+1)*args.batch_size]

        with tf.GradientTape() as tape:
            latent = encoder.call(batch_real * mask)
            gen_tmp = generator.call(latent)
            gen_output = batch * mask + gen_tmp * ( 1 - mask )
            disc_fake = discriminator.call(gen_output)
            e_loss = encoder.loss_function(gen_output, batch_real, d_fake)

        loss_list.append(e_loss.numpy())

        # Optimize generator
        e_gradients = tape.gradient(e_loss, encoder.trainable_variables)
        encoder.optimizer.apply_gradients(zip(e_gradients, encoder.trainable_variables))

        # Save
        if iteration % args.save_every == 0:
            manager.save()

        # print("Training %d/%d complete" % (x, int(real_images.shape[0]/args.batch_size)) )
        if x % 100 == 0 and x > 0:
            print("Training %3.3f percent complete" % (100*x/(real_images.shape[0]/args.batch_size)))
            print("Encoder Loss:")
            print(e_loss)
            fid_ = fid_function(batch_real, gen_output)
            fid_list.append(fid_.numpy())
            print('FID score')
            print(fid_)

    return fid_list, loss_list

def test(encoder, decoder, cropped, mask):
    for x in range(0, int(cropped.shape[0]/args.batch_size/4)):
        batch_cropped = cropped[x*args.batch_size: (x+1)*args.batch_size]
        mean, logsigma, enc_out = encoder.call(batch_cropped)
        dec_out = decoder.call(enc_out)
        generated = dec_out * 255
        output = generated.numpy().astype(np.uint8)
        for i in range(0, args.batch_size):
            image = output[i]
            s = args.out_dir + '/' + str(i) + '.png'
            imwrite(s, image)
        print("Training %d/%d complete" % (x, int(batch_cropped.shape[0]/args.batch_size)))

def crop_img(images, x, y):
    images_copy = np.copy(images)
    images_copy[:, y:, x:, :] = 0.0
    return images_copy

########################## Printing plot ##########################
def plot(l, n, epoch):
    x_axis = np.array(range(1, len(l) + 1)) / len(l) * (epoch + 1)

    plt.figure()
    plt.plot(x_axis, l, linewidth=3)
    plt.xticks(np.arange(0,epoch + 1,1))
    plt.ylabel('{}'.format(n))
    plt.xlabel('Number of epochs')
    plt.title('{}'.format(n))
    plt.savefig('{}.png'.format(n))
    plt.clf()
########################## Printing plot ##########################


def main():
    # Get data
    image_data = get_data('./lfw', target_size=(args.img_width, args.img_height), processed=True)
    partition = int(len(image_data) * 0.8)
    train_data = np.copy(image_data[:partition])
    test_data = np.copy(image_data[partition:])
    print('Train and test data retrieved')
    # Define mask
    mask = np.ones((64, 64, 3), dtype=np.float32)
    mask[32: , 32: , : ] = 0

    # Initialize model
    encoder = Encoder(64, 5, 512)
    decoder = Decoder(64, 5, 3)
    discriminator = Discriminator(64, 5, 1)

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(encoder=encoder, decoder=decoder, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ########################## Printing plot ##########################
    enc_loss_list = []
    dec_loss_list = []
    disc_loss_list = []
    fid_list = []
    ########################## Printing plot ##########################

    if args.restore_checkpoint or args.mode == 'test' or args.mode == 'train_completion' or args.mode == 'train_test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint)

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    dec_loss, disc_loss, fid = train(decoder, discriminator, train_data, 512, manager)
                    print("Average FID for Epoch: " + str(np.mean(fid)))
                    fid_list += fid
                    dec_loss_list += dec_loss
                    disc_loss_list += disc_loss
                    plot(fid_list, 'FID', epoch)
                    plot(dec_loss_list, 'Decoder Loss', epoch)
                    plot(disc_loss_list, 'Discriminator Loss', epoch)

                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
            if args.mode == 'test':
                test(generator)

            if args.mode == 'train_completion':
                loss_list = []
                fid_list = []
                for epoch in range(0, args.num_epochs):
                    print('========================== COMPLETION EPOCH %d  ==========================' % epoch)
                    f, l = train_encoder(encoder, decoder, discriminator, train_data, mask, manager)
                    fid_list += f
                    enc_loss_list += l
                    plot(fid_list, 'FID', epoch)
                    plot(enc_loss_list, 'Encoder Loss', epoch)
                    # print("Average FID for Epoch: " + str(avg_fid))
                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()

            if args.mode == 'test_completion':
                test_completion(encoder, generator, dataset_iterator, mask)


    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()