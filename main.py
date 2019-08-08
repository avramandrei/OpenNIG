from src.models.catalog import CVAE
import argparse
from src.preparation import prepare_mnist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode")
    parser.add_argument("--model")
    parser.add_argument("--data")

    args = parser.parse_args()

    if args.model == "CVAE":
        model = CVAE()
    else:
        raise ValueError("Selected model not found")

    if args.data == "mnist":
        train_data, test_data = prepare_mnist.load_data()
    else:
        raise ValueError("Selected data not found")

    if args.mode == "train":
        pass


