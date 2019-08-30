"""
    This is a script that automatizes the dataset download process. The dataset download process can be configured by
    using a yaml file.
"""


import argparse
import opennng.preparation.prepare_mnist as prepare_mnist
import opennng.preprocess.process_mnist as process_mnist
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    # download the mnist dataset. It will be downloaded in two parts, raw and processed, in data/mnist relative path
    if args.dataset == "mnist":
        mnist_raw_data_path = os.path.join("data", "mnist", "raw")
        mnist_processed_data_path = os.path.join("data", "mnist", "processed")

        prepare_mnist.prepare_data(mnist_raw_data_path)
        process_mnist.process_data(mnist_raw_data_path, mnist_processed_data_path)
