import argparse
import opengen.preparation.prepare_mnist as prepare_mnist
import opengen.preprocess.process_mnist as process_mnist
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    if args.dataset == "mnist":
        mnist_raw_data_path = os.path.join("data", "mnist", "raw")
        mnist_processed_data_path = os.path.join("data", "mnist", "processed")

        prepare_mnist.prepare_data(mnist_raw_data_path)
        process_mnist.process_data(mnist_raw_data_path, mnist_processed_data_path)
