import numpy as np
import tensorflow as tf
import os, os.path
import glob
from PIL import Image

def get_data():

    test_images = []
    test_path = "/stanford-cars-dataset/cars_test/*.jpg"
    train_path = "/stanford-cars-dataset/cars_train/*.jpg"
    valid_images = [".jpg"]
    for test_file in glob.glob(test_path):
        # TODO: Preprocess
        image = Image.open(test_file)
        test_images.append(image)

    return train_data, train_label, test_data, test_labels