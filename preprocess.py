import os, shutil, glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def crop_center(image, target_size):
    h, w = image.shape[-3], image.shape[-2]
    if h > w:
        cropped_image = tf.image.crop_to_bounding_box(image, (h - w) // 2, 0, w, w)
    else:
        cropped_image = tf.image.crop_to_bounding_box(image, 0, (w - h) // 2, h, h)
    return tf.image.resize(cropped_image, target_size)

# def crop(image, target_size):
    # h, w = image.shape[-3], image.shape[-2]
    # cropped_image = tf.image.crop_to_bounding_box(image, int(0.2*h), int(0.2*w), int(0.6*h), int(0.6*w))
    # return tf.image.resize(cropped_image, target_size)

def get_data(dir_path, target_size=(250,250), processed=False):

    if not processed:
        processed_dir = os.path.join(os.path.dirname(dir_path), 'processed')
    else:
        processed_dir = dir_path

    def load_and_process_image(file_path):
        img = image.load_img(file_path)
        img = image.img_to_array(img)
        if not processed:
            img = crop_center(img, target_size)
            save_path = os.path.join(processed_dir, os.path.basename(file_path))
            image.save_img(save_path, img)
        return img

    if not processed:
        file_path = dir_path + '/*.jpg'
        dataset = glob.glob(file_path)
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
        os.mkdir(processed_dir)
    else:
        file_path = processed_dir + '/*.jpg'
        dataset = glob.glob(file_path)

    num_img = len(dataset)
    data = np.zeros(shape=(num_img, target_size[0], target_size[1], 3), dtype=np.float32)

    for (index, img) in enumerate(dataset):
        img_data = load_and_process_image(img)
        data[index] = (img_data / 255 - 0.5) * 2
        if index % 100 == 0:
            print("Preprocessing %3.2f percent completed" %(index/num_img*100))
    print("Preprocessing 100 percent completed")
    return data
