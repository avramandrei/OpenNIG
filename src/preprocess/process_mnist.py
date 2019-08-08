import numpy as np
import os


def process_data(load_path, save_path):
    raw_train = np.load(os.path.join(load_path, "train.npy"))
    raw_test = np.load(os.path.join(load_path, "test.npy"))

    train = raw_train / 255
    test = raw_test / 255

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "train.npy"), train)
    np.save(os.path.join(save_path, "test.npy"), test)


if __name__ == "__main__":
    mnist_raw_data_path = os.path.join("..", "..", "data", "mnist", "raw")
    mnist_processed_data_path = os.path.join("..", "..", "data", "mnist", "processed")

    process_data(mnist_raw_data_path, mnist_processed_data_path)

