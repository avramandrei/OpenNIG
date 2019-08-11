from src.preparation.prepare import load_data
from src.models.cvae import CVAESmall, CVAEMedium
from src.util.losses import vae_loss_fcn
import tensorflow as tf


def parse_data(config):
    train_dataset = load_data(config["data"]["train"])
    eval_dataset = load_data(config["data"]["eval"])

    try:
        processed = config["data"]["processed"]
    except KeyError:
        processed = False

    return train_dataset, eval_dataset, processed


def parse_model(config):
    if config["model"]["type"] == "CVAESmall":
        model = CVAESmall(config["data"]["shape"])
        loss_fcn = vae_loss_fcn
    if config["model"]["type"] == "CVAEMedium":
        model = CVAEMedium(config["data"]["shape"])
        loss_fcn = vae_loss_fcn

    return model, loss_fcn


def parse_train(config):
    # extract learning_rate
    try:
        learning_rate = config["train"]["learning_rate"]
    except KeyError:
        learning_rate = 1e-3

    # construct the optimizer
    if config["train"]["optimizer"] == "AdamOptimizer":
        try:
            beta1 = config["train"]["optimizer_params"]["beta1"]
        except KeyError:
            beta1 = 0.9

        try:
            beta2 = config["train"]["optimizer_params"]["beta2"]
        except KeyError:
            beta2 = 0.99

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)

    # extract the iterations
    try:
        iterations = config["train"]["iterations"]
    except KeyError:
        iterations = 100000

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

    return optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path


def parse_eval(config):
    try:
        batch_size = config["eval"]["batch_size"]
    except KeyError:
        batch_size = 32

    try:
        eval_steps = config["eval"]["eval_steps"]
    except KeyError:
        eval_steps = 32

    return eval_steps, batch_size