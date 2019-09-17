# OpenNNG (Work in progress...)

OpenNNG (Open Neural Network Generator) is a general purpose data generator toolkit that uses TensorFlow 2.0. Supported architectures:

- [variational autoencoder](https://arxiv.org/abs/1312.6114)
- [generative adversarial network](https://arxiv.org/abs/1406.2661)

## Key features

OpenNNG focuses on modularity to support advanced modeling and training capabilities:

 - usage of predefined models
 - creation of custom architectures
 - domain adaptation
 
## Installation

### Clone repository

If you want to use OpenNNG as a command line interface where the processes of training, evaluating etc. are all automated, run the following commands:

```
git clone https://github.com/avramandrei/OpenNNG.git
pip install -r requirements.txt
```

### pip

If you want to use OpenNNG as an API and have more flexibility, install it via pip:

```
pip install opennng
```

## Usage

OpenNNG requires:
 - Python >= 3.6
 - TensorFlow >= 2.0.0rc0
 - Pillow >=6.1
 
### Data downloading

OpenNNG offers a veriety of databases that can be downloaded with the `download.py` script. [Here]() is a list of the available databases.

```
python3 download.py [database]
```
 
### Data processing

Processed data must be saved in Numpy `.npy` files. Data can be automatically processed using the `process.py` script. 

```
python3 process.py [raw_data_path] [processed_data_path] [--from_noise] [--normalize]
```

| Named Argument | Type | Description |
| --- | --- | -- |
| raw_data_path | str | Path to the raw data. Two(train, valid)/four(train_X, valid_X, train_y, valid_y) folders are expected here. |
| processed_data_path | str | Path where processed data will be saved |
| --from_noise | bool | Whether the generator will produce data from noise or from given data. If set to `True`, two directories are expected in `raw_data_path`, else four directories are expected. |
| --normalize | bool | Whether to normalize the data. |

### Configuration

To train and generate new samples, a `YAML` configuration file must be provided. [Here](https://github.com/avramandrei/OpenNNG/blob/master/examples/yaml_config/config_docs.yml) is an exhaustive list of all the configuration parameters.

### Train

To train, run `train.py` with a `YAML` configuration file as parameter.

```
python3 train.py <path_to_yaml_config_file>
```

This script will automatically generate 10 samples that shows how the training process evolves at evrey checkpoint. To disable this functionality, set `generate_train_samples` to `False` in `YAML` configuration file. 

| Model | Samples |
| --- | --- |
| ConvVAESmall | ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_1.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_2.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_3.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_4.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_5.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_6.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_7.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_8.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_9.gif?raw=true) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_vae/train_sameple_10.gif?raw=true) |
| ConvGANSmall | ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_1.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_2.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_3.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_4.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_5.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_6.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_7.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_8.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_9.gif) ![alt text](https://github.com/avramandrei/OpenNNG/blob/master/examples/train_samples/conv_gan/train_sameple_10.gif) |


### Generate

To generate a new sample, run `generate.py` with a `YAML` configuration file as parameter.

```
python3 generate.py <path_to_yaml_config_file>
```

