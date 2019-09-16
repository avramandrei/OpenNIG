"""
    This is a script that automatizes the dataset download process. The dataset download process can be configured by
    using a yaml file.
"""


import argparse
from opennng.preparation.prepare import prepare_mnist, prepare_facade
from opennng.preprocess.process import process_mnist, process_facade
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    # download the mnist dataset. It will be downloaded in two parts, raw and processed, in data/mnist relative path
    if args.dataset == "mnist":
        prepare_mnist()

    # download the facade dataset. It will be downloaded in two parts, raw and processed, in data/facade relative path
    if args.dataset == "facade":
        raw_data_path = os.path.join("data", "facade", "raw")
        processed_data_path = os.path.join("data", "facade", "processed")

        prepare_facade(raw_data_path)