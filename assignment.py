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

def main():
    train_data = get_data('./cars_train')
    test_data = get_data('./cars_test')

if __name__ == '__main__':
   main()