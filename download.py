"""
    This is a script that automatizes the dataset download process. The dataset download process can be configured by
    using a yaml file.
"""


import argparse
from opennng.preparation.prepare import prepare_mnist, prepare_facade, prepare_cifar10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    # download the mnist dataset in 'data/mnist/raw`.
    if args.dataset == "mnist":
        prepare_mnist()

    # download the cifar10 dataset in `data/cifar10/
    if args.dataset == "cifar10":
        prepare_cifar10()

    # download the facade dataset in 'data/facade/raw'.
    if args.dataset == "facade":
        prepare_facade()