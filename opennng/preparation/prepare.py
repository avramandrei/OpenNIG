import numpy as np
import tensorflow as tf
import os
from zipfile import ZipFile
import sys
import requests
from PIL import Image
import shutil


def _download_dataset(url, filename, name):
    print("Downloading dataset `{}`.".format(name))

    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}{}] {}%'.format('=' * (done-1), ">",  '.' * (50 - done), done*2))
                sys.stdout.flush()
    sys.stdout.write('\n\n')


def load_data(path):
    """
        This function loads a .npy file from a path, creates and returns a Dataset.
    """
    data = np.load(path)

    dataset = tf.data.Dataset.from_tensor_slices(data)

    return dataset


def prepare_mnist(path):
    """
        This function loads the mnist dataset (train and test) and saves it at a given path.
    """
    (train, _), (test, _) = tf.keras.datasets.mnist.load_data()

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "train.npy"), train)
    np.save(os.path.join(path, "eval.npy"), test)


def prepare_facade(path):
    """
        This function loads the facade dataset (train and test) and saves it at a given path.
    """

    base_dataset_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
    extended_dataset_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip"

    base_save_path = os.path.join("data", "temp", "base.zip")
    extended_save_path = os.path.join("data", "temp", "extended.zip")

    if not os.path.exists(os.path.join("data", "temp")):
        os.makedirs(os.path.join("data", "temp"))

    if not os.path.exists(base_save_path):
        _download_dataset(base_dataset_url, base_save_path, "facade base")

    if not os.path.exists(extended_save_path):
        _download_dataset(extended_dataset_url, extended_save_path, "facade extended")

    print("Working on raw data for facade dataset...")

    base_extract_path = os.path.join("data", "temp", "base")
    extended_extract_path = os.path.join("data", "temp", "extended")
    with ZipFile(base_save_path, 'r') as zip_ref:
        zip_ref.extractall(base_extract_path)
    with ZipFile(extended_save_path, 'r') as zip_ref:
        zip_ref.extractall(extended_extract_path)

    # read and add the X and y from the facade base to a list
    base_X = []
    base_y = []

    for filename in os.listdir(os.path.join(base_extract_path, "base")):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(base_extract_path, "base", filename), "r")
            base_y.append(np.array(img))

        if filename.endswith(".png"):
            img = Image.open(os.path.join(base_extract_path, "base", filename), "r")
            base_X.append(np.array(img))

    # read and add the X and y from the facade extended to a list
    extended_X = []
    extended_y = []
    for filename in os.listdir(os.path.join(extended_extract_path, "extended")):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(extended_extract_path, "extended", filename), "r")
            extended_y.append(np.array(img))

        if filename.endswith(".png"):
            img = Image.open(os.path.join(extended_extract_path, "extended", filename), "r")
            extended_X.append(np.array(img))

    X = base_X + extended_X
    y = base_y + extended_y

    assert len(X) == len(y)

    train_len = int(len(X) * 0.9)

    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, "train_X.npy"), np.array(X[:train_len]))
    np.save(os.path.join(path, "eval_X.npy"), np.array(X[train_len:]))
    np.save(os.path.join(path, "train_y.npy"), np.array(y[:train_len]))
    np.save(os.path.join(path, "eval_y.npy"), np.array(y[train_len:]))

    shutil.rmtree(os.path.join("data", "temp"), )
