import numpy as np
import tensorflow as tf
from opennng.models.dcgan import DCGANSmall, DCGANMedium
from opennng.models.dcvae import DCVAESmall, DCVAEMedium
from opennng.util.trainer import dcvae_trainer, dcgan_trainer
import pickle
import os


def parse_data(args):
    train_np = np.load(args.train_path).astype(np.float32)
    valid_np = np.load(args.valid_path).astype(np.float32)

    return tf.data.Dataset.from_tensor_slices(train_np), \
           tf.data.Dataset.from_tensor_slices(valid_np)


def parse_model(args):
    try:
        input_shape = np.load(args.valid_path).shape[1:]
    except AttributeError:
        with open(os.path.join(os.path.dirname(args.model_path), "model.meta"), "rb") as model_meta:
            input_shape = pickle.load(model_meta)

    if args.model == "DCVAESmall":
        model = DCVAESmall(input_shape)
        trainer = dcvae_trainer
    if args.model == "DCVAEMedium":
        model = DCVAEMedium(input_shape)
        trainer = dcvae_trainer

    if args.model == "DCGANSmall":
        model = DCGANSmall(input_shape)
        trainer = dcgan_trainer
    if args.model == "DCGANMedium":
        model = DCGANMedium(input_shape)
        trainer = dcgan_trainer

    try:
        model.load_weights(args.model_path)
    except AttributeError:
        pass

    return model, trainer


def parse_train(args):
    if args.optimizer == "Adam" and ("GAN" in args.model or "Pix" in args.model):
        gen_optimizer = tf.keras.optimizers.Adam(args.learning_rate)
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
    normalize = args.normalize

    return num_sample, sample_save_path, normalize
