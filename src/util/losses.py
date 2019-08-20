import tensorflow as tf
import numpy as np


def _log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


@tf.function
def vae_loss_fcn(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = _log_normal_pdf(z, 0., 0.)
    logqz_x = _log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def disc_loss_fcn(real_output, fake_output):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(real_output), real_output)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(fake_output), fake_output)
    loss = real_loss + fake_loss

    return tf.reduce_mean(loss)


def gen_loss_fcn(fake_output):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(fake_output), fake_output))


@tf.function
def gan_loss_fcn(model, x):
    batch_size = x.shape[0]

    noise = tf.random.normal([batch_size, model.noise_size])

    generated_images = model.generative_net(noise, training=True)

    real_output = model.discriminative_net(x, training=True)
    fake_output = model.discriminative_net(generated_images, training=True)

    gen_loss = gen_loss_fcn(fake_output)
    disc_loss = disc_loss_fcn(real_output, fake_output)

    return gen_loss, disc_loss