import tensorflow as tf
import numpy as np
import os


mnist_raw_data_path = os.path.join("..", "..", "data", "mnist", "raw")


def prepare_data(path):
    (train, _), (test, _) = tf.keras.datasets.mnist.load_data()

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "train.npy"), train)
    np.save(os.path.join(path, "eval.npy"), test)


if __name__ == "__main__":
    mnist_raw_data_path = os.path.join("..", "..", "data", "mnist", "raw")

    prepare_data(mnist_raw_data_path)
