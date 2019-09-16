"""
    This is a script that automatizes the training process. The training process can be configured by using a yaml file.
"""

from opennng.util.trainer import train_model
from opennng.util.parser import parse_data
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("dev_path", type=str)
    parser.add_argument("--processed", type=bool, default=False)
    args = parser.parse_args()

    train_dataset, dev_dataset = parse_data(args)

    train_model(model, train_step, loss_fcn,
                train_dataset, eval_dataset,
                optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                eval_batch_size, eval_steps,
                generate_train_samples, num_train_samples)

