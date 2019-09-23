import argparse
import os
from PIL import Image
import numpy as np
from distutils.util import strtobool


def load_images_from_path(path, shape):
    data = []

    width = int(shape.split(",")[0][1:])
    height = int(shape.split(",")[1][:-1])

    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))

        if shape is not None:
            img = img.resize((width, height))
        data.append(np.array(img))

    return np.array(data, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("processed_data_path", type=str)
    parser.add_argument("--from_noise", type=str, default="True")
    parser.add_argument("--normalize", type=str, default="[-1,1]")
    parser.add_argument("--reshape_X", type=str, default=None)
    parser.add_argument("--reshape_y", type=str, default=None)
    parser.add_argument("--flip_left_right", type=str, default="False")

    args = parser.parse_args()

    if strtobool(args.from_noise):
        print("Started processing data from '{}'...".format(args.raw_data_path))

        print("Reading train and valid data from '{}'...".format(args.raw_data_path))
        train_y = load_images_from_path(os.path.join(args.raw_data_path, "train"), args.reshape_y)
        valid_y = load_images_from_path(os.path.join(args.raw_data_path, "valid"), args.reshape_y)

        maximum = np.amax(np.concatenate((train_y, valid_y), axis=0))

        if args.normalize == "[-1,1]":
            print("Normalizing data values to [-1, 1]...")
            train_y = (train_y - 127.5) / 127.5
            valid_y = (valid_y - 127.5) / 127.5
        elif args.normalize == "[0,1]":
            print("Normalizing data values to [0, 1]...")
            train_y = train_y / 255
            valid_y = valid_y / 255
        else:
            raise ValueError("Argument `--normalize` must be either [-1,1] or [0,1].")

        assert len(train_y.shape) == len(valid_y.shape)

        if len(train_y.shape) == 3:
            train_y = np.expand_dims(train_y, axis=3)
            valid_y = np.expand_dims(valid_y, axis=3)

        if not os.path.exists(args.processed_data_path):
            os.makedirs(args.processed_data_path)

        print("Saving processed data to '{}'...".format(args.processed_data_path))
        np.save(os.path.join(args.processed_data_path, "train_y.npy"), train_y)
        np.save(os.path.join(args.processed_data_path, "valid_y.npy"), valid_y)
    else:
        print("Started processing data from '{}'...".format(args.raw_data_path))

        print("Reading train and valid data from '{}'...".format(args.raw_data_path))
        train_X = load_images_from_path(os.path.join(args.raw_data_path, "train_X"), args.reshape_X)
        valid_X = load_images_from_path(os.path.join(args.raw_data_path, "valid_X"), args.reshape_X)
        train_y = load_images_from_path(os.path.join(args.raw_data_path, "train_y"), args.reshape_y)
        valid_y = load_images_from_path(os.path.join(args.raw_data_path, "valid_y"), args.reshape_y)

        if args.normalize == "[-1,1]":
            print("Normalizing data values to [-1, 1]...")
            train_X = (train_X - 127.5) / 127.5
            valid_X = (valid_X - 127.5) / 127.5
            train_y = (train_y - 127.5) / 127.5
            valid_y = (valid_y - 127.5) / 127.5
        elif args.normalize == "[0,1]":
            print("Normalizing data values to [0, 1]...")
            train_y = train_y / 255
            valid_y = valid_y / 255
            train_y = train_y / 255
            valid_y = valid_y / 255
        else:
            raise ValueError("Argument `--normalize` must be either [-1,1] or [0,1].")

        assert train_y.shape[0] == train_X.shape[0]
        assert valid_y.shape[0] == valid_X.shape[0]

        if len(train_y.shape) == 3:
            train_y = np.expand_dims(train_y, axis=3)
            valid_y = np.expand_dims(valid_y, axis=3)
        if len(train_X.shape) == 3:
            train_X = np.expand_dims(train_X, axis=3)
            valid_X = np.expand_dims(valid_X, axis=3)

        if not os.path.exists(args.processed_data_path):
            os.makedirs(args.processed_data_path)

        print("Saving processed data to '{}'...".format(args.processed_data_path))
        np.save(os.path.join(args.processed_data_path, "train_X.npy"), train_X)
        np.save(os.path.join(args.processed_data_path, "valid_X.npy"), valid_X)
        np.save(os.path.join(args.processed_data_path, "train_y.npy"), train_y)
        np.save(os.path.join(args.processed_data_path, "valid_y.npy"), valid_y)

