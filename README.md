# OpenNGen (Work in progress...)

OpenNGen (Open Neural Generators) is a general purpose data generator toolkit using TensorFlow 2.0. Supported architectures:

- [variational autoencoder](https://arxiv.org/abs/1312.6114)

## Key features

OpenNGen focuses on modularity to support advanced modeling and training capabilities:

 - usage of predefined models
 - creation of custom architectures
 - domain adaptation

## Usage

OpenNGen requires:
 - Python >= 3.6
 - TensorFlow >= 2.0
 
### Data preparation

Data must be saved in Numpy `.npy` files. In this example we will use the mnist dataset to generate new images. For this purpose, use 
the `download.py` script with `mnist` as argument. This command will automatically download mnist (raw and processed) dataset in `data/` directory.

```
python3 download.py mnist
```

### Training

To train a model you need to create a `YAML` configuration file. [Here]() you can find a simple train configuration file and [here]() is an exhaustive list of all the configuration parameters. One you have created the configuration file, run `train.py` with the created file as parameter.

```
python3 train.py <path_to_yaml_config_file>
```

### Generate

To generate a new sample run `generate.py` with a `YAML` configuration file as parameter.

```
python3 generate.py <path_to_yaml_config_file>
```

