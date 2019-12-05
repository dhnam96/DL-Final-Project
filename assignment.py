import os
import sys
import numpy as np
import tensorflow as tf
from preprocess import get_data

class Generator_Model(tf.keras.Model):
    def __init__(self):
        super(Generator_Model, self).__init__()

class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()

        # Kernel size
        self.kernel_size = 64
        # Layers
        self.conv1 = tf.keras.Layers.Conv2d(filters = 5, kernel_size= self.kernel_size, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="relu")
        self.conv2 = tf.keras.Layers.Conv2d(filters = 5, kernel_size= 4*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv3 = tf.keras.Layers.Conv2d(filters = 5, kernel_size= 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv4 = tf.keras.Layers.Conv2d(filters = 5, kernel_size= 8*self.kernel_size, strides=[1, 1], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02))
        self.conv5 = tf.keras.Layers.Conv2d(filters = 5, kernel_size= 1, strides=[2, 2], padding="valid", kernel_initializer = tf.random_normal_initializer(0, 0.02), activation="sigmoid")
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

def main():
    train_data = get_data('./cars_train', resize=True)
    print('train data returned')
    test_data = get_data('./cars_test', resize=True)

    print(train_data.shape)
    print(test_data.shape)

if __name__ == '__main__':
   main()