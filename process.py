import argparse
import os
from PIL import Image
import numpy as np
from distutils.util import strtobool


def load_images_from_path(path, shape, flip_left_right):
    data = []
    if shape is not None:
        width = int(shape.split(",")[0][1:])
        height = int(shape.split(",")[1][:-1])

    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))

        if shape is not None:
            img = img.resize((width, height))
        data.append(np.array(img))

        if flip_left_right and np.random.normal() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            data.append(np.array(img))

    return np.array(data, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("processed_data_path", type=str)
    parser.add_argument("--normalize", type=str, default="[-1,1]")
    parser.add_argument("--reshape_X", type=str, default=None)
    parser.add_argument("--reshape_y", type=str, default=None)
    parser.add_argument("--flip_left_right", type=str, default="False")

    args = parser.parse_args()

    print("Started processing data from '{}'...".format(args.raw_data_path))

    print("Reading train and valid data from '{}'...".format(args.raw_data_path))
    train = load_images_from_path(os.path.join(args.raw_data_path, "train"),
                                  args.reshape_y, strtobool(args.flip_left_right))
    valid = load_images_from_path(os.path.join(args.raw_data_path, "valid"),
                                  args.reshape_y, strtobool(args.flip_left_right))

    maximum = np.amax(np.concatenate((train, valid), axis=0))

    if args.normalize == "[-1,1]":
        print("Normalizing data values to [-1, 1]...")
        train = (train - (maximum / 2)) / (maximum / 2)
        valid = (valid - (maximum / 2)) / (maximum / 2)
    elif args.normalize == "[0,1]":
        print("Normalizing data values to [0, 1]...")
        train = train / maximum
        valid = valid / maximum
    else:
        raise ValueError("Argument `--normalize` must be either [-1,1] or [0,1].")

    assert len(train.shape) == len(valid.shape)

    if len(train.shape) == 3:
        train = np.expand_dims(train, axis=3)
        valid = np.expand_dims(valid, axis=3)

    if not os.path.exists(args.processed_data_path):
        os.makedirs(args.processed_data_path)

    print("Saving processed data to '{}'...".format(args.processed_data_path))
    np.save(os.path.join(args.processed_data_path, "train.npy"), train)
    np.save(os.path.join(args.processed_data_path, "valid.npy"), valid)

