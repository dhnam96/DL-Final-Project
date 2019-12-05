import os, sys
import numpy as np
import tensorflow as tf
from preprocess import get_data 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Generator_Model(tf.keras.Model):
    def __init__(self):
        super(Generator_Model, self).__init__()

class Discriminator_Model(tf.keras.Model):
    def __init__(self):
        super(Discriminator_Model, self).__init__()

def main():
    train_data = get_data('./cars_train', True)
    test_data = get_data('./cars_test', True)

if __name__ == '__main__':
   main()