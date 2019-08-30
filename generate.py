"""
    This is a script that automatizes the sample generation process. The sample generation process can be configured by
    using a yaml file.
"""


from opennng.util.generator import generate_png_samples
import yaml
import argparse
import opennng.util.yaml_parser as yaml_parser

import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    with open(args.config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)

            # parse the model information
            model, _, _= yaml_parser.parse_model(config)
            print("Model selected: {}\n".format(config["model"]["type"]))
            model.summary()

            # parse the sample generation information
            num_sample, sample_save_path = yaml_parser.parse_generate(config)
            if not os.path.exists(sample_save_path):
                os.makedirs(sample_save_path)

        except yaml.YAMLError as exc:
            print(exc)
            exit(0)

    generate_png_samples(model, num_sample, sample_save_path)