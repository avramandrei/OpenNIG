import argparse
import os
from PIL import Image
import numpy as np


def load_images_from_path(path):
    data = []

    for filename in os.listdir(path):
        img = Image.open(os.path.join(path, filename))
        data.append(np.array(img))

    return np.array(data, dtype=np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("processed_data_path", type=str)
    parser.add_argument("--from_noise", type=bool, default=True)
    parser.add_argument("--normalize", type=bool, default=True)

    args = parser.parse_args()

    if args.from_noise:
        print("Started processing data from '{}'...".format(args.raw_data_path))

        print("Reading train and valid data from '{}'...".format(args.raw_data_path))
        train_y = load_images_from_path(os.path.join(args.raw_data_path, "train"))
        valid_y = load_images_from_path(os.path.join(args.raw_data_path, "valid"))

        maximum = np.amax(np.concatenate((train_y, valid_y), axis=0))

        if args.normalize:
            print("Normalizing data values to [0, 1]...")
            train_y = train_y / maximum
            valid_y = valid_y / maximum

        assert len(train_y.shape) == len(valid_y.shape)

        if len(train_y.shape) == 3:
            train_y = np.expand_dims(train_y, axis=3)
            valid_y = np.expand_dims(valid_y, axis=3)

        if not os.path.exists(args.processed_data_path):
            os.makedirs(args.processed_data_path)

        print("Saving processed data to '{}'...".format(args.processed_data_path))
        np.save(os.path.join(args.processed_data_path, "train_y.npy"), train_y)
        np.save(os.path.join(args.processed_data_path, "valid_y.npy"), valid_y)

