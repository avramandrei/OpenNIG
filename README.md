# OpenNGen (Work in progress...)

OpenNGen (Open Neural Generators) is a general purpose data generator toolkit using TensorFlow 2.0. Supported architectures:

- [variational autoencoder](https://arxiv.org/abs/1312.6114)
- [generative adversarial network](https://arxiv.org/abs/1406.2661)

## Key features

OpenNGen focuses on modularity to support advanced modeling and training capabilities:

 - usage of predefined models
 - creation of custom architectures
 - domain adaptation

## Usage

OpenNGen requires:
 - Python >= 3.6
 - TensorFlow >= 2.0
 
### Data processing

Data must be saved in Numpy `.npy` files. In this example we will use the mnist dataset to generate new images. For this purpose, use 
the `download.py` script with `mnist` as argument. This command will automatically download mnist (raw and processed) dataset in `data/` directory.

```
python3 download.py mnist
```

### Configuration

To train and generate new samples, a `YAML` configuration file must be provided. [Here](https://github.com/avramandrei/OpenNGen/blob/master/examples/yaml_config/config_docs.yml) is an exhaustive list of all the configuration parameters.

### Train

To train, run `train.py` with a `YAML` configuration file as parameter.

```
python3 train.py <path_to_yaml_config_file>
```

This script will automatically generate 10 samples that shows how the training process evolves at evrey checkpoint. To disable this functionality, set `generate_train_samples` to `False` in `YAML` configuration file. The following samples show how the training process evolved for several models that can be found in `OpenNGen`.

### Generate

To generate a new sample, run `generate.py` with a `YAML` configuration file as parameter.

```
python3 generate.py <path_to_yaml_config_file>
```

