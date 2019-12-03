import numpy as np
import tensorflow as tf
import os, os.path

def get_data(train_file, test_file):

    image_list = []
    test_path = "/stanford-cars-dataset/cars_test"
    train_path = "/stanford-cars-dataset/cars_train"
    valid_images = [".jpg"]
    for image in os.listdir(test_path):
        # TODO: Preprocess
    return train_data, train_label, test_data, test_labels