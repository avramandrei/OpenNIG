import tensorflow as tf
from src.util.trainer import train_model
import yaml
import argparse
import src.util.yaml_parser as yaml_parser
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)

            train_dataset, eval_dataset, processed = yaml_parser.parse_data(config)

            model, train_step, loss_fcn = yaml_parser.parse_model(config)
            print("Model selected: {}\n".format(config["model"]["type"]))
            model.summary()

            optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path = yaml_parser.parse_train(config)
            new_save_checkpoint_path = os.path.join(*os.path.split(save_checkpoint_path)[:-1])
            if not os.path.exists(new_save_checkpoint_path):
                os.makedirs(new_save_checkpoint_path)

            eval_steps, eval_batch_size = yaml_parser.parse_eval(config)

        except yaml.YAMLError as exc:
            print(exc)
            exit(0)

    train_model(model, train_step, loss_fcn,
                train_dataset, eval_dataset, processed,
                optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                eval_batch_size, eval_steps)

