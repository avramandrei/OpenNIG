import numpy as np
import os
from zipfile import ZipFile
import tensorflow as tf
import shutil
import tarfile
from opennng.preparation.util import download_dataset, make_dirs
from PIL import Image


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


def prepare_fashion_mnist():
    temp_path, data_path, temp_data_path = make_dirs("fashion-mnist")

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("Extracting `fashion-mnist` data...")
    for i, (img_arr, img_label) in enumerate(zip(train_images, train_labels)):
        img = Image.fromarray(img_arr)
        img.save(os.path.join(data_path, "train", "img_{}_{}.png".format(i, img_label)))

    for i, (img_arr, img_label) in enumerate(zip(test_images, test_labels)):
        img = Image.fromarray(img_arr)
        img.save(os.path.join(data_path, "valid", "img_{}_{}.png".format(i, img_label)))

    shutil.rmtree(temp_path)


def prepare_facade():
    """
        This function loads the facade dataset (train and test) and saves it at a given path.
    """

    temp_path, data_path, temp_data_path = make_dirs("facade", from_noise=False)

    download_dataset("http://cmp.felk.cvut.cz/~tylecr1/facade/CMP_facade_DB_base.zip", temp_data_path, "facade")

    print("Extracting dataset `facade`...")
    with ZipFile(temp_data_path, "r") as zip:
        zip.extractall(os.path.join(temp_path, "facade"))

    X_counter, y_counter = 0, 0
    for filename in os.listdir(os.path.join(temp_path, "facade", "base")):
        if "png" in filename:
            if X_counter < 350:
                shutil.move(os.path.join(temp_path, "facade", "base", filename), os.path.join(data_path, "train_X"))
            else:
                shutil.move(os.path.join(temp_path, "facade", "base", filename), os.path.join(data_path, "valid_X"))
            X_counter += 1

        if "jpg" in filename:
            if y_counter < 350:
                shutil.move(os.path.join(temp_path, "facade", "base", filename), os.path.join(data_path, "train_y"))
            else:
                shutil.move(os.path.join(temp_path, "facade", "base", filename), os.path.join(data_path, "valid_y"))
            y_counter += 1

    shutil.rmtree(temp_path)


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