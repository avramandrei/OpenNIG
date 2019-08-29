"""
    This module is responsible for implementing the losses used by the models.
"""


import tensorflow as tf
import numpy as np


def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def vae_loss_fcn(model, x):
    """
        This function defines the Variation Autoencoder loss. It is calculated as the Mone Carlo estimator of the
        expectation: log p(x|z) + log p(z) + log q(z|x).

        Args:
            model (tf.keras.Model): The Variational Autoencoder model.
            x (tf.Tensor): The input.

        Returns:
            The loss function of the training step.
    """
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)

    # calculate log p(x|z)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    # calculate log p(z)
    logpz = _log_normal_pdf(z, 0., 0.)
    # calculate log q(z|x)
    logqz_x = _log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def gan_disc_loss_fcn(real_output, fake_output):
    """
        This function defines the loss of the discriminative network: log D(real_output) + log D(1-fake_output).

        Args:
            real_output (tf.Tensor): The real output.
            fake_output (tf.Tensor): The fake output.

        Returns:
            The loss function of the discriminative network.
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_output), real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_output), fake_output)
    loss = real_loss + fake_loss

    return tf.reduce_mean(loss)


def gan_gen_loss_fcn(fake_output):
    """
        This function defines the loss of the generative network: log D(fake_output).

        Args:
            fake_output (tf.Tensor): The fake output.

        Returns:
            The loss function of the generative network.
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_output), fake_output))


@tf.function
def gan_loss_fcn(model, x):
    """
        This function defines the total loss of the GAN: generative network loss + discriminative network loss.

        Args:
            model (tf.keras.Model): The Generative Adversarial Network model.
            x (tf.Tensor): The input.

        Returns:
            The loss function of the discriminative and generative networks at the training step.
    """

    batch_size = x.shape[0]

    # create a latent sample
    noise = tf.random.normal([batch_size, model.latent_dim])

    # generate a new observation
    generated_x = model.generative_net(noise, training=True)

    # classify the real and fake inputs
    real_output = model.discriminative_net(x, training=True)
    fake_output = model.discriminative_net(generated_x, training=True)

    # calculate the discriminative and generative networks losses
    gen_loss = gan_gen_loss_fcn(fake_output)
    disc_loss = gan_disc_loss_fcn(real_output, fake_output)

    return gen_loss, disc_loss