import tensorflow as tf
import numpy as np
import os


def prepare_data(path):
    """
        This function loads the mnist dataset (train and test) and saves it at a given path.
    """
    (train, _), (test, _) = tf.keras.datasets.mnist.load_data()

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "train.npy"), train)
    np.save(os.path.join(path, "eval.npy"), test)
