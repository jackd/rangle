import tensorflow as tf
import numpy as np


def load_data():
    """
    Load the data into numpy arrays.

    Current implementataion random for testing.
    """
    images = np.random.random((1000, 64, 64, 3))
    labels = (np.random.random(1000)*2 - 1)*np.pi
    return images.astype(np.float32), labels.astype(np.float32)


def load_data_as_filenames():
    """Same as `load_data` but gets filenames instead of images."""
    paths = ['filename1.jpg', 'filename2.jpg']
    labels = [0.1, -0.4]
    return paths, labels


def get_input_fn_as_filenames(
        batch_size, num_epochs=None, shuffle=True, num_threads=16):
    """Same as `get_input_fn` but loads lazily."""
    filenames, labels = load_data()
    shuffle_size = len(filenames)

    def input_fn():
        def _read_data(filename, label):
            image_data = tf.read_file(filename)
            image = tf.image.decode_jpeg(image_data)
            image = tf.image.per_image_standardization(image)
            return image, label

        dataset = tf.contrib.data.Dataset.from_tensor_slices(
            (filenames, labels))
        dataset = dataset.map(_read_data)

        dataset = dataset.shuffle(shuffle_size).repeat(num_epochs).batch(
                batch_size)
        images_tf, labels_tf = dataset.make_one_shot_iterator().get_next()
        return images_tf, labels_tf


def get_input_fn(batch_size, num_epochs=None, shuffle=True, num_threads=16):
    images, labels = load_data()
    shuffle_size = len(images)

    def input_fn():
        dataset = tf.contrib.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(shuffle_size).repeat(num_epochs).batch(
                batch_size)
        images_tf, labels_tf = dataset.make_one_shot_iterator().get_next()
        return images_tf, labels_tf

    return input_fn
