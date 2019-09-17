"""
    This is a script that automatizes the sample generation process. The sample generation process can be configured by
    using a yaml file.
"""


from opennng.util.generator import generate_png_samples
import argparse
from opennng.util.parser import parse_model, parse_generate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("model_path", type=str)
    parser.add_argument("--num_sample", type=int, default=10)
    parser.add_argument("--sample_save_path", type=str, default="samples")

    args = parser.parse_args()

    model, _ = parse_model(args)
    num_sample, sample_save_path = parse_generate(args)

    generate_png_samples(model, num_sample, sample_save_path)