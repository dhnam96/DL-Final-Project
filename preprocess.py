import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import glob 

def get_data(dir_path):
    def load_and_process_image(file_path, target_size = (640, 420)):
        img = image.load_img(file_path, target_size=target_size)
        img = image.img_to_array(img)
        return img
    file_path = dir_path + '/*.jpg'
    dataset = glob.glob(file_path)
    dataset= np.array([load_and_process_image(img) for img in dataset])

    return dataset