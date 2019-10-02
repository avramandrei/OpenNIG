"""
    This module is responsible for generating training steps for each model.
"""


from opennng.util.losses import vae_loss, gan_loss, pix2pix_disc_loss, pix2pix_gen_loss, pix2pix_loss
import tensorflow as tf


def vae_train_step(model, x, optimizer):
    """
        This function calculates a training step for the Variational Autoencoder.

        Args:
            model (tf.keras.model): The Variational Autoencoder model.
            x (tf.Tensor): The input.
            optimizer: The optimizer used by the training.

        Returns:
            The loss value.
    """
    with tf.GradientTape() as tape:
        loss = vae_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def gan_train_step(model, x, optimizer, train_disc, iterations, iter):
    """
    This function calculates a training step for the Generative Adversarial Network.

        Args:
            model (tf.keras.model): The Variational Autoencoder model.
            x (tf.Tensor): The input.
            optimizer: The optimizer used by the training.

        Returns:
            The loss value.
    """
    generator_optimizer = optimizer[0]
    discriminator_optimizer = optimizer[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss, disc_loss = gan_loss(model, x, iterations, iter)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generative_net.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generative_net.trainable_variables))

    if train_disc:
        gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminative_net.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminative_net.trainable_variables))

    return gen_loss, disc_loss