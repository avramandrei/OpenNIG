import numpy as np
import tensorflow as tf
from opennng.models.conv_gan import ConvGANSmall, ConvGANMedium
from opennng.models.conv_vae import ConvVAESmall, ConvVAEMedium
from opennng.util.trainer import conv_vae_trainer, conv_gan_trainer
import pickle
import os


def parse_data(args):
    if args.from_noise:
        if args.train_X_path is not None or args.train_X_path is not None:
            raise ValueError("Parameters 'from_noise' can't be 'True' while 'train_X_path' and 'valid_X_path' are "
                             "different from 'None'")

        train_y_np = np.load(args.train_y_path)
        valid_y_np = np.load(args.valid_y_path)

        return None, None, \
               tf.data.Dataset.from_tensor_slices(train_y_np), \
               tf.data.Dataset.from_tensor_slices(valid_y_np)


def parse_model(args):
    try:
        input_shape = np.load(args.valid_y_path).shape[1:]
    except AttributeError:
        with open(os.path.join(os.path.dirname(args.model_path), "model.meta"), "rb") as model_meta:
            input_shape = pickle.load(model_meta)

    if args.model == "ConvVAESmall":
        model = ConvVAESmall(input_shape)
        trainer = conv_vae_trainer
    if args.model == "ConvVAEMedium":
        model = ConvVAEMedium(input_shape)
        trainer = conv_vae_trainer

    if args.model == "ConvGANSmall":
        model = ConvGANSmall(input_shape)
        trainer = conv_gan_trainer
    if args.model == "ConvGANMedium":
        model = ConvGANMedium(input_shape)
        trainer = conv_gan_trainer

    try:
        model.load_weights(args.model_path)
    except AttributeError:
        pass

    return model, trainer


def parse_train(args):
    if args.optimizer == "Adam" and "GAN" in args.model:
        gen_optimizer = tf.keras.optimizers.Adam(args.learning_rate*2)
        disc_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        optimizer = (gen_optimizer, disc_optimizer)
    else:
        optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    iterations = args.iterations
    batch_size = args.batch_size
    save_checkpoint_steps = args.save_checkpoint_steps
    save_checkpoint_path = args.save_checkpoint_path
    label_smooth = args.label_smooth

    return optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path, label_smooth


def parse_valid(args):
    valid_batch_size = args.valid_batch_size
    valid_steps = args.valid_steps

    return valid_batch_size, valid_steps


def parse_generate_samples(args):
    generate_train_samples = args.generate_train_samples
    num_train_samples = args.num_train_samples

    return generate_train_samples, num_train_samples


def parse_generate(args):
    num_sample = args.num_sample
    sample_save_path = args.sample_save_path

    return num_sample, sample_save_path
