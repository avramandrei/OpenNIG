import numpy as np
import tensorflow as tf
from opennng.models.conv_gan import ConvGANSmall
from opennng.models.conv_vae import ConvVAESmall
from opennng.util.losses import gan_loss_fcn, vae_loss_fcn
from opennng.util.train_steps import gan_train_step, vae_train_step


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
    if args.from_noise:
        input_shape = np.load(args.valid_y_path).shape[1:]

    if args.model == "ConvVAESmall":
        model = ConvVAESmall(input_shape)
        train_step = vae_train_step
        loss_fcn = vae_loss_fcn

    return model, train_step, loss_fcn


def parse_train(args):
    if args.optimizer == "Adam":
        optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    iterations = args.iterations
    batch_size = args.batch_size
    save_checkpoint_steps = args.save_checkpoint_steps
    save_checkpoint_path = args.save_checkpoint_path

    return optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path


def parse_valid(args):
    valid_batch_size = args.valid_batch_size
    valid_steps = args.valid_steps

    return valid_batch_size, valid_steps


def parse_generate_samples(args):
    generate_train_samples = args.generate_train_samples
    num_train_samples = args.num_train_samples

    return generate_train_samples, num_train_samples

