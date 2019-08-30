"""
    This module is responsible for parsing the configuration file.
"""


from opennng.preparation.prepare import load_data
from opennng.models.conv_vae import ConvVAESmall, ConvVAEMedium
from opennng.models.conv_gan import ConvGANSmall
from opennng.util.losses import vae_loss_fcn, gan_loss_fcn
import tensorflow as tf
from opennng.util.train_steps import vae_train_step, gan_train_step


def parse_data(config):
    """
        This function parses the data from the yaml configuration file.

        Args:
            config: The configuration file.

        Returns:
            The path to the train and evaluation data set, whether the data is processed.
    """
    train_dataset = load_data(config["data"]["train_path"])
    eval_dataset = load_data(config["data"]["eval_path"])

    try:
        processed = config["data"]["processed"]
    except KeyError:
        processed = False

    return train_dataset, eval_dataset, processed


def parse_model(config):
    """
        This function parses the model information from the yaml configuration file.

        Args:
            config: The configuration file.

        Returns:
            The model, the train step and the loss function.
    """
    input_shape = config["model"]["input_shape"]

    if config["model"]["type"] == "ConvVAESmall":
        model = ConvVAESmall(input_shape)
        loss_fcn = vae_loss_fcn
        train_step = vae_train_step
    if config["model"]["type"] == "ConvVAEMedium":
        model = ConvVAEMedium(input_shape)
        loss_fcn = vae_loss_fcn
        train_step = vae_train_step
    if config["model"]["type"] == "ConvGANSmall":
        model = ConvGANSmall(input_shape)
        loss_fcn = gan_loss_fcn
        train_step = gan_train_step

    try:
        model.load_weights(config["model"]["load_path"])
    except KeyError:
        pass

    return model, train_step, loss_fcn


def parse_train(config):
    """
        This function parses the train information from the configuration file.

        Args:
            config: The configuration file.

        Returns:
            The optimizer, the number of iterations, the batch size, the checkpoint steps, the checkpoint path,
            the number of train samples.
    """
    # extract learning_rate
    try:
        learning_rate = config["train"]["learning_rate"]
    except KeyError:
        learning_rate = 1e-3

    # construct the optimizer
    if config["train"]["optimizer"] == "Adam":
        try:
            beta1 = config["train"]["optimizer_params"]["beta1"]
        except KeyError:
            beta1 = 0.9

        try:
            beta2 = config["train"]["optimizer_params"]["beta2"]
        except KeyError:
            beta2 = 0.99

        if config["model"]["type"] == "ConvGANSmall":
            optimizer = list()
            optimizer.append(tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2))
            optimizer.append(tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2))
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

    # extract the iterations
    try:
        iterations = config["train"]["iterations"]
    except KeyError:
        iterations = 1e5

    # extract the batch_size
    try:
        batch_size = config["train"]["batch_size"]
    except KeyError:
        batch_size = 32

    # extract checkpoint steps
    try:
        save_checkpoint_steps = config["train"]["save_checkpoint_steps"]
    except KeyError:
        save_checkpoint_steps = 1000

    try:
        save_checkpoint_path = config["train"]["save_checkpoint_path"]
    except KeyError:
        save_checkpoint_path = "trained_models/model"

    try:
        generate_train_samples = config["train"]["generate_train_samples"]
    except KeyError:
        generate_train_samples = True

    try:
        num_train_samples = config["train"]["num_train_samples"]
    except KeyError:
        num_train_samples = 10

    return optimizer, \
           int(iterations), batch_size,\
           save_checkpoint_steps, save_checkpoint_path, \
           generate_train_samples, num_train_samples


def parse_eval(config):
    """
        This function parses the evaluation information from the configuration file.

        Args:
            config: The configuration file.

        Returns:
            The evaluation steps, the batch size.
    """
    try:
        batch_size = config["eval"]["batch_size"]
    except KeyError:
        batch_size = 32

    try:
        eval_steps = config["eval"]["eval_steps"]
    except KeyError:
        eval_steps = 1000

    return eval_steps, batch_size


def parse_generate(config):
    """
            This function parses the generate samples information from the configuration file.

            Args:
                config: The configuration file.

            Returns:
                The number of samples to generate and the path.
        """
    try:
        num_sample = config["generate"]["num_samples"]
    except KeyError:
        num_sample = 10

    try:
        save_path = config["generate"]["save_path"]
    except KeyError:
        save_path = "samples"

    return num_sample, save_path