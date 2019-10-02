"""
    This module is responsible for implementing the losses used by the models.
"""


import tensorflow as tf
import numpy as np

label_smooth = 0


def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def vae_loss(model, x):
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


def gan_disc_loss(real_output, fake_output):
    """
        This function defines the loss of the discriminative network: log D(real_output) + log D(1-fake_output).

        Args:
            real_output (tf.Tensor): The real output.
            fake_output (tf.Tensor): The fake output.

        Returns:
            The loss function of the discriminative network.
    """
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_output) - label_smooth, real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_output), fake_output)
    loss = real_loss + fake_loss

    return tf.reduce_mean(loss)


def gan_gen_loss(fake_output):
    """
        This function defines the loss of the generative network: log D(fake_output).

        Args:
            fake_output (tf.Tensor): The fake output.

        Returns:
            The loss function of the generative network.
    """
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_output), fake_output))


def gan_loss(model, x, iterations, iter):
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
    generated_x = model.generative_net(noise)

    # classify the real and fake inputs
    real_output = model.discriminative_net(x)
    fake_output = model.discriminative_net(generated_x)

    # calculate the discriminative and generative networks losses
    gen_loss = gan_gen_loss(fake_output)
    disc_loss = gan_disc_loss(real_output, fake_output)

    return gen_loss, disc_loss


def pix2pix_disc_loss(disc_real_output, disc_generated_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = cross_entropy(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def pix2pix_gen_loss(disc_generated_output, gen_output, target):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 100

    gan_loss = cross_entropy(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss


def pix2pix_loss(model, X, y):
    gen_y = model.generative_net(X)

    disc_real_output = model.discriminative_net([X, y])
    disc_generated_output = model.discriminative_net([X, gen_y])

    gen_loss = pix2pix_gen_loss(disc_generated_output, gen_y, y)
    disc_loss = pix2pix_disc_loss(disc_real_output, disc_generated_output)

    return disc_loss, gen_loss