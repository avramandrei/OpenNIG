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

Data must be saved in Numpy `.npy` files. In this example we will use the mnist data set to generate new images. For this purpose, two 
script have been created: `prepare_mnist.py` and `process_mnist.py`, located in 
[src/preparation](https://github.com/avramus/OpenNGen/tree/master/src/preparation) and 
[src/preprocess](https://github.com/avramus/OpenNGen/tree/master/src/preprocess), respectievly. Run them to create the train and
the evaluation `.npy` files in `src/data/processed/`.

```
python3 prepare_mnist.py
python3 process_mnist.py
```

### Training

To train a model you need to create a `YAML` configuration file. [Here]() you can find a simple train configuration file and [here]() you 
can find all the configuration parameters. One you have created the configuration file, run `train.py` with the created file as parameter.

```
train.py <path_to_yaml_config_file>
```

### Generate

To generate a new sample run `generate.py` with a `YAML` configuration file as parameter.

```
generate.py <path_to_yaml_config_file>
```

