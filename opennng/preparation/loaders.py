import numpy as np
import tensorflow as tf


def load_data(path):
    """
        This function loads a .npy file from a path, creates and returns a Dataset.
    """
    data = np.load(path)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset