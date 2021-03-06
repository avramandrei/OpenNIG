# OpenNIG 

OpenNIG (Open Neural Image Generator) is a toolkit that generates new images from a given distribution. Its role is to accelerate research in that direction by offering a flexible and easy to use ecosystem for such models. 

### Installation

Simply clone the repository and install the requirements.

```
git clone https://github.com/avramandrei/OpenNIG.git
cd OpenNIG/
pip3 install -r requirements.txt
```

### Evolution of the generated images

| Model | Dataset | Iterations | Samples |
| --- | --- | --- | :---: |
| DCVAESmall | mnist | 1k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_1.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_2.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_3.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_4.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_5.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_6.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_7.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_8.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_9.gif?raw=true" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_mnist_samples/train_sample_10.gif?raw=true" height="42" width="42"> |
| DCGANSmall | mnist | 2.5k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_1.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_2.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_3.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_4.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_5.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_6.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_7.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_8.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_9.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_mnist_samples/train_sample_10.gif" height="42" width="42"> |
| DCVAESmall | fashion-mnist | 5k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_1.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_2.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_3.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_4.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_5.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_6.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_7.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_8.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_9.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_small_fashion-mnist_samples/train_sample_10.gif" height="42" width="42"> |
| DCGANSmall | fashion-mnist | 10k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_1.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_2.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_3.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_4.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_5.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_6.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_7.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_8.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_9.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_small_fashion-mnist_samples/train_sample_10.gif" height="42" width="42"> |
| DCVAEMedium | cifar10 | 100k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_1.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_2.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_3.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_4.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_5.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_6.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_7.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_8.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_9.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcvae_medium_cifar10_samples/train_sample_10.gif" height="42" width="42"> |
| DCGANMedium | cifar10 | 150k | <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_1.gif" height="42" width="42"> <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_2.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_3.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_4.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_5.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_6.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_7.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_8.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_9.gif" height="42" width="42">  <img src="https://github.com/avramandrei/OpenNIG/blob/master/examples/dcgan_medium_cifar10_samples/train_sample_10.gif" height="42" width="42"> |


Note: The above samples are just some examples of the generated images during training. The final results can be improved by tuning the hyperparameters of the models.

### Data downloading

OpenNIG offers three demos databases that can be downloaded with the `download.py` script: `mnist`, `fashion-mnist` and `cifar10`. The images will be saved in two directories, `train` and `valid`, in `data/raw/<database>`.

```
python3 download.py [database]
```

|  Argument | Type | Description |
| --- | --- | -- |
| dataset | str | Dataset to be downloaded: `mnist`, `fashion-mnist` or `cifar10`. |
 
### Data processing

This step is optional, if you already have your own preprocessing pipeline, please proceed to the next section, but remember that OpenNIG uses NumPy's `.npy` data format.

If you want to use other datasets, you have to manually create two directories, `train` and `valid`, that contain the training and the validation images, respectevly. Run the `process.py` script to process the images in these directories. The script will create two files, `train.npy` and `valid.npy`.

```
python3 process.py [train_dir_path] [valid_dir_path] 
                   [output_path] 
                   [--normalize]
                   [--reshape]
                   [--flip_left_right]
                   [--flip_top_bottom]
```

|  Argument | Type | Description |
| -------------------- | --- | -- |
| train_dir_path | str | Path to the `train` directory containing the training images. |
| valid_dir_path | str | Path to the `valid` directory containing the validation images. |
| output_path | str | Path where processed data will be saved |
| --normalize | str | Normalize data to `[-1,1]` or `[0,1]`. Default: `[-1,1]`. |
| --reshape_y | str | Reshape x data to specified shape. Shape must be specified as `(width,height)`. Default: `None`. |
| --reshape_x | str | Reshape y data to specified shape. Shape must be specified as `(width,height)`. Default: `None`. |
| --flip_left_right | bool | Adds 50% horizontally flipped images to the dataset. Default: `False`. |
| --flip_top_bottom | bool | Adds 50% vertically flipped images to the dataset. Default: `False`. |

### Train

To train, run the `train.py` script. This script automatically generates 10 GIF images in `<save_checkpoint_path>/samples`, that show how the training process evolves at every checkpoint.

```
python3 train.py [model] 
                 [train_path] [valid_path]
                 [--optimizer] [--learning_rate] [--iterations] [--batch_size] [--label_smooth]
                 [--save_checkpoint_steps] [--save_checkpoint_path]
                 [--valid_batch_size] [--valid_steps] 
                 [--generate_train_samples] [--num_train_samples]
```

|  Argument | Type | Description |
| --- | --- | -- |
| model | str | Type of the model: DCVAESmall, DCVAEMedium, DCVAEBig, DCGANSmall, DCGANMedium, DCGANBig.  |
| --model_path | str | Load the model weights from this path. |
| train_path | str | Path to the train data, saved as a `.npy` file. |
| valid_path | str | Path to the validation data, saved as a `.npy` file. |
| --optimizer | str | Name of the optimizer, as described in https://keras.io/optimizers/. Default value: `"Adam"` |
| --learning_rate | float | Learning rate of the optimizer. Default: `0.001`. |
| --iterations | int | Number of training steps. Default: `100000`. |
| --batch_size | int | Batch size for training. Defaul: `32`. |
| --save_checkpoint_steps | int | Save a checkpoint every X steps. Default: `1000` |
| --save_checkpoint_path | str | Save the model at this path every `--save_checkpoint_steps`. Default: `trained_model/model` |
| --valid_batch_size | int | Batch size for validation. Defaul: `32`. |
| --valid_steps | int | Perfom validation every X steps. Default: `250`. |
| --generate_train_samples | bool | Whether to generate samples during training. Default: `True`. |
| --num_train_samples | int | Number of generated training samples. Default: `10`. |
| --label_smooth | float | Number used for label smoothing. Default: 0 |


### Generate

To generate new images, run `generate.py`.

```
python3 generate.py [model] [model_path] [--num_sample] [--sample_save_path][--normalize]
```

|  Argument | Type | Description | 
| --- | --- | -- |
| model | str | Type of the model. [Here](docs/models.md) is a list of all the available models. |
| model_path | str | Load the model from this path. |
| --num_sample | int | Number of samples to generate.Default: `10`. |
| --sample_save_path | str | Save the samples at this path. Default: `samples`. |

## Credits

If you found this work useful, please consider citing the following [paper](https://ieeexplore.ieee.org/abstract/document/9142009):
```
@inproceedings{avram2020opennig,
  title={OpenNIG-Open Neural Image Generator},
  author={Avram, Andrei-Marius and Morogan, Luciana and Toma, Stefan-Adrian},
  booktitle={2020 13th International Conference on Communications (COMM)},
  pages={177--181},
  year={2020},
  organization={IEEE}
}
```



