import numpy as np
import os
from zipfile import ZipFile
from PIL import Image
import shutil
import tarfile
from opennng.preparation.util import download_dataset, make_dirs


def prepare_mnist():
    """
        This function loads the mnist dataset (train and test) and saves it at a given path.
    """
    temp_path, data_path, temp_data_path = make_dirs("mnist")

    download_dataset("https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz", temp_data_path, "mnist")

    print("Extracting `mnist` data...")
    with tarfile.open(temp_data_path, "r:gz") as tar:
        tar.extractall(temp_path)

    for digit_dir in os.listdir(os.path.join(temp_path, "mnist_png", "training")):
        for image in os.listdir(os.path.join(temp_path, "mnist_png", "training", digit_dir)):
            shutil.copy(os.path.join(temp_path, "mnist_png", "training", digit_dir, image),
                        os.path.join(data_path, "train", image))

    for digit_dir in os.listdir(os.path.join(temp_path, "mnist_png", "testing")):
        for image in os.listdir(os.path.join(temp_path, "mnist_png", "testing", digit_dir)):
            shutil.copy(os.path.join(temp_path, "mnist_png", "testing", digit_dir, image),
                        os.path.join(data_path, "valid", image))

    shutil.rmtree(temp_path)


def prepare_facade(path):
    """
        This function loads the facade dataset (train and test) and saves it at a given path.
    """

    base_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip"
    extended_url = "http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_extended.zip"

    base_temp_save_path = os.path.join("data", "temp", "base.zip")
    extended_temp_save_path = os.path.join("data", "temp", "extended.zip")

    if not os.path.exists(os.path.join("data", "temp")):
        os.makedirs(os.path.join("data", "temp"))

    download_dataset(base_url, base_temp_save_path, "facade base")

    download_dataset(extended_url, extended_temp_save_path, "facade extended")

    print("Working on raw data for facade dataset...")

    base_extract_path = os.path.join("data", "temp", "base")
    extended_extract_path = os.path.join("data", "temp", "extended")
    with ZipFile(base_temp_save_path, 'r') as zip_ref:
        zip_ref.extractall(base_extract_path)
    with ZipFile(extended_temp_save_path, 'r') as zip_ref:
        zip_ref.extractall(extended_extract_path)

    # read and add the X and y from the facade base to a list
    base_X = []
    base_y = []

    img_height = 256
    img_width = 256

    for filename in os.listdir(os.path.join(base_extract_path, "base")):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(base_extract_path, "base", filename), "r")
            base_y.append(np.array(img.resize((img_height, img_width))))

        if filename.endswith(".png"):
            img = Image.open(os.path.join(base_extract_path, "base", filename), "r")
            base_X.append(np.array(img.resize((img_height, img_width))))

    # read and add the X and y from the facade extended to a list
    extended_X = []
    extended_y = []
    for filename in os.listdir(os.path.join(extended_extract_path, "extended")):
        if filename.endswith(".jpg"):
            img = Image.open(os.path.join(extended_extract_path, "extended", filename), "r")
            img.resize((img_height, img_width))
            extended_y.append(np.array(img.resize((img_height, img_width))))

        if filename.endswith(".png"):
            img = Image.open(os.path.join(extended_extract_path, "extended", filename), "r")
            extended_X.append(np.array(img.resize((img_height, img_width))))

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


def prepare_cifar10():
    temp_path, data_path, temp_data_path = make_dirs("cifar10")

    download_dataset("http://pjreddie.com/media/files/cifar.tgz", temp_data_path, "cifar10")

    print("Extracting `cifar10` data...")
    with tarfile.open(temp_data_path, "r") as tar:
        tar.extractall(data_path)

    if os.path.exists(os.path.join(data_path, "train")):
        shutil.rmtree(os.path.join(data_path, "train"))

    if os.path.exists(os.path.join(data_path, "valid")):
        shutil.rmtree(os.path.join(data_path, "valid"))

    shutil.move(os.path.join(data_path, "cifar", "train"), data_path)
    shutil.move(os.path.join(data_path, "cifar", "test"), data_path)

    os.rename(os.path.join(data_path, "test"), os.path.join(data_path, "valid"))

    shutil.rmtree(os.path.join(data_path, "cifar"))
    shutil.rmtree(temp_path)