import numpy as np
import tensorflow as tf
import cv2
import os
import shutil
from tensorflow.keras.preprocessing import image
import glob 

def crop_center(image, target_size):
    h, w = image.shape[-3], image.shape[-2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return tf.image.resize(cropped_image, target_size)


def get_data(dir_path):
    def load_and_process_image(file_path, target_size = (32, 32)):
        img = image.load_img(file_path)
        img = image.img_to_array(img)
        img = crop_center(img, target_size)
        save_path = os.path.join(os.path.dirname(file_path), 'preprocessed', os.path.basename(file_path))
        image.save_img(save_path, img)
        return img

    file_path = dir_path + '/*.jpg'
    dataset = glob.glob(file_path)

    preprocess_dir = os.path.join(dir_path, 'preprocessed')
    if os.path.exists(preprocess_dir):
        shutil.rmtree(preprocess_dir)
    os.mkdir(os.path.join(dir_path, 'preprocessed'))

    dataset= np.array([load_and_process_image(img) for img in dataset])

    return dataset