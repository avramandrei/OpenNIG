from src.util.losses import vae_loss_fcn, gan_loss_fcn
import tensorflow as tf


@tf.function
def vae_train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = vae_loss_fcn(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def gan_train_step(model, x, optimizer):
    generator_optimizer = optimizer[0]
    discriminator_optimizer = optimizer[1]

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_loss, disc_loss = gan_loss_fcn(model, x)

    gradients_of_generator = gen_tape.gradient(gen_loss, model.generative_net.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, model.discriminative_net.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, model.generative_net.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, model.discriminative_net.trainable_variables))

    return (gen_loss + disc_loss)/2