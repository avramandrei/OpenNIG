import tensorflow as tf
import numpy as np
import os


mnist_raw_data_path = os.path.join("..", "..", "data", "mnist", "raw")


def prepare_data(path):
    (train, _), (test, _) = tf.keras.datasets.mnist.load_data()

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "train.npy"), train)
    np.save(os.path.join(path, "test.npy"), test)


def load_data():
    train = np.load(os.path.join(mnist_raw_data_path, "train.npy"))
    test = np.load(os.path.join(mnist_raw_data_path, "test.npy"))

    train_dataset = tf.data.Dataset.from_tensor_slices(train).batch()
    test_dataset = tf.data.Dataset.from_tensor_slices(test)

    return train_dataset, test_dataset


if __name__ == "__main__":
    mnist_raw_data_path = os.path.join("..", "..", "data", "mnist", "raw")

    prepare_data(mnist_raw_data_path)
