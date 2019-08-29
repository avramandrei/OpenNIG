from opengen.util.trainer import train_model
import yaml
import argparse
import opengen.util.yaml_parser as yaml_parser


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

            optimizer, iterations, batch_size, \
            save_checkpoint_steps, save_checkpoint_path, \
            generate_train_samples, num_train_samples = yaml_parser.parse_train(config)

            eval_steps, eval_batch_size = yaml_parser.parse_eval(config)

        except yaml.YAMLError as exc:
            print(exc)
            exit(0)

    train_model(model, train_step, loss_fcn,
                train_dataset, eval_dataset, processed,
                optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                eval_batch_size, eval_steps,
                generate_train_samples, num_train_samples)

