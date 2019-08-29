import numpy as np
import tensorflow as tf


def load_data(path):
    data = np.load(path)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset