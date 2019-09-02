import numpy as np
import os


def process_data(load_path, save_path, normalize=True):
    """
        This function loads the mnist .npy files and process them.

        Args:
            load_path (str): Path of the mnist .npy raw files.
            save_path (str): Path of the mnist .npy processed files.
            normalize (bool): Whether to normalize the data in [-1, 1] interval.
    """
    raw_train = np.load(os.path.join(load_path, "train.npy"))
    raw_eval = np.load(os.path.join(load_path, "eval.npy"))

    if normalize:
        train = (raw_train - 127.5) / 255
        eval = (raw_eval - 127.5) / 255

    if len(train.shape) == 3:
        train = np.expand_dims(train, axis=3)
    if len(eval.shape) == 3:
        eval = np.expand_dims(eval, axis=3)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(os.path.join(save_path, "train.npy"), np.float32(train))
    np.save(os.path.join(save_path, "eval.npy"), np.float32(eval))


