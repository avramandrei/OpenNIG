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

This script will automatically generate 10 samples that shows how the training process evolves at evrey checkpoint. To disable this functionality, set `generate_train_samples` to `False` in `YAML` configuration file. 

| Model | Samples |
| --- | --- |
| ConvVAESmall | ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_1.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_2.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_3.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_4.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_5.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_6.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_7.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_8.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_9.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_vae/train_sameple_10.gif?raw=true) |
| ConvGANSmall | ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_1.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_2.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_3.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_4.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_5.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_6.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_7.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_8.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_9.gif) ![alt text](https://github.com/avramandrei/OpenNGen/blob/master/examples/train_samples/conv_gan/train_sameple_10.gif) |


### Generate

To generate a new sample, run `generate.py` with a `YAML` configuration file as parameter.

```
python3 generate.py <path_to_yaml_config_file>
```

