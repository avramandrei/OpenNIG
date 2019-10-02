import tensorflow as tf
import time
import os
from opennng.util.generator import generate_gif_train_samples
from opennng.util.train_steps import vae_train_step, gan_train_step, pix2pix_train_step
from opennng.util.losses import vae_loss, gan_loss, pix2pix_loss
import pickle


def dcvae_trainer(model,
                  train_y, valid_y,
                  optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                  valid_batch_size, valid_steps,
                  generate_train_samples, num_train_samples):

    """
        This function is used to train a model.

        Args:
            model (tf.keras.Model): The model to be trained.
            train_step: The train step of the model.
            loss_fcn: The loss function used by the model.
            train_dataset (tf.data.Dataset): The dataset used for training the model.
            valid_y (tf.data.Dataset): The dataset used for evaluating the model.
            optimizer: The optimizer used for train the model.
            iterations (int): The number of iterations that the model will be trained on.
            batch_size (int): The batch size of the training process.
            save_checkpoint_steps (int): A checkpoint will be generated every this many steps.
            save_checkpoint_path (str): The checkpoint path.
            valid_batch_size (int): The batch size of the evaluation process.
            valid_steps (int): An evaluation of the model will be performed every this many steps.
            generate_train_samples (bool): Whether to generate gif samples during the training process.
            num_train_samples (int): The number of samples to generate during the training process.
    """
    train_dataset = train_y.batch(batch_size).repeat()
    valid_dataset = valid_y.batch(valid_batch_size).repeat(1)

    # generate a noise (latent sample) from where train samples will be created
    if generate_train_samples:
        noise = tf.random.normal(shape=[num_train_samples, 1, model.latent_dim], seed=42)

    # iterate the train dataset
    for iter, train_batch in enumerate(train_dataset):
        if iter > iterations:
            break

        # perform a train step
        train_loss = vae_train_step(model, train_batch, optimizer)

        if iter % valid_steps == 0:
            print("Iter: {}/{} - Train loss: {:.3f}".format(iter, iterations, train_loss))

        # if the current step is a saving checkpoint step, save the model and add a new frame to the gif samples
        if iter % save_checkpoint_steps == 0:
            print("Iter: {}/{} - Checkpoint reached. Saving the model...".format(iter, iterations))
            model.save_weights(os.path.join(save_checkpoint_path, "model", "model_iter_{}".format(iter)))

            with open(os.path.join(save_checkpoint_path, "model", "model.meta"), "wb") as model_meta_file:
                pickle.dump(train_batch[0].shape, model_meta_file)

            if generate_train_samples:
                print("Iter: {}/{} - Generating {} train gif samples with model {}..."
                      .format(iter, iterations, num_train_samples, model.name))

                generate_gif_train_samples(model, num_train_samples,
                                           noise, os.path.join(save_checkpoint_path, "train_samples"),
                                           "[0,1]")

            loss_mean = tf.keras.metrics.Mean()
            for valid_batch in valid_dataset:
                loss_mean(vae_loss(model, valid_batch))

            end = time.time()

            print("Iter: {}/{} - Train loss: {:.3f}, Valid loss: {:.3f}, Time: {:.3f}".
                  format(iter, iterations, train_loss, loss_mean.result(), 0 if iter == 0 else end - start))

            start = time.time()


def dcgan_trainer(model,
                  train_y, valid_y,
                  optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                  valid_batch_size, valid_steps,
                  generate_train_samples, num_train_samples):

    """
        This function is used to train a model.

        Args:
            model (tf.keras.Model): The model to be trained.
            train_step: The train step of the model.
            loss_fcn: The loss function used by the model.
            train_dataset (tf.data.Dataset): The dataset used for training the model.
            valid_y (tf.data.Dataset): The dataset used for evaluating the model.
            optimizer: The optimizer used for train the model.
            iterations (int): The number of iterations that the model will be trained on.
            batch_size (int): The batch size of the training process.
            save_checkpoint_steps (int): A checkpoint will be generated every this many steps.
            save_checkpoint_path (str): The checkpoint path.
            valid_batch_size (int): The batch size of the evaluation process.
            valid_steps (int): An evaluation of the model will be performed every this many steps.
            generate_train_samples (bool): Whether to generate gif samples during the training process.
            num_train_samples (int): The number of samples to generate during the training process.
    """
    train_dataset = train_y.batch(batch_size).repeat()
    valid_dataset = valid_y.batch(valid_batch_size).repeat(1)

    # generate a noise (latent sample) from where train samples will be created
    if generate_train_samples:
        noise = tf.random.normal(shape=[num_train_samples, 1, model.latent_dim], seed=42)

    # iterate the train dataset
    for iter, train_batch in enumerate(train_dataset):
        if iter > iterations:
            break

       # train_disc = True if iter % 2 == 0 else False
        train_disc = True

        # perform a train step
        gen_loss, disc_loss = gan_train_step(model, train_batch, optimizer, train_disc, iterations, iter+1)

        if iter % valid_steps == 0:
            print("Iter: {}/{} - Train loss: (gen {:.3f}, disc {:.3f})".format(iter, iterations, gen_loss, disc_loss))

        # if the current step is a saving checkpoint step, save the model and add a new frame to the gif samples
        if iter % save_checkpoint_steps == 0:
            print("Iter: {}/{} - Checkpoint reached. Saving the model...".format(iter, iterations))
            model.save_weights(os.path.join(save_checkpoint_path, "model", "gan"))

            if generate_train_samples:
                print("Iter: {}/{} - Generating {} train gif samples with model {}..."
                      .format(iter, iterations, num_train_samples, model.name))

                generate_gif_train_samples(model, num_train_samples,
                                           noise, os.path.join(save_checkpoint_path, "train_samples"),
                                           "[-1,1]")

            valid_gen_mean_loss = tf.keras.metrics.Mean()
            disc_gen_mean_loss = tf.keras.metrics.Mean()

            for valid_batch in valid_dataset:
                gen_loss, disc_loss = gan_loss(model, valid_batch, iterations, iter+1)
                valid_gen_mean_loss(gen_loss)
                disc_gen_mean_loss(disc_loss)

            end = time.time()

            print("Iter: {}/{} - Train loss: (gen {:.3f}, disc {:.3f}), "
                  "Valid loss: (gen {:.3f}, disc {:.3f}), Time: {:.3f}".
                  format(iter, iterations, gen_loss, disc_loss,
                         valid_gen_mean_loss.result(), disc_gen_mean_loss.result(), 0 if iter == 0 else end - start))

            start = time.time()


def pix2pix_trainer(model,
                  train_X, valid_X, train_y, valid_y,
                  optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                  valid_batch_size, valid_steps,
                  generate_train_samples, num_train_samples):

    """
        This function is used to train a model.

        Args:
            model (tf.keras.Model): The model to be trained.
            train_step: The train step of the model.
            loss_fcn: The loss function used by the model.
            train_dataset (tf.data.Dataset): The dataset used for training the model.
            valid_y (tf.data.Dataset): The dataset used for evaluating the model.
            optimizer: The optimizer used for train the model.
            iterations (int): The number of iterations that the model will be trained on.
            batch_size (int): The batch size of the training process.
            save_checkpoint_steps (int): A checkpoint will be generated every this many steps.
            save_checkpoint_path (str): The checkpoint path.
            valid_batch_size (int): The batch size of the evaluation process.
            valid_steps (int): An evaluation of the model will be performed every this many steps.
            generate_train_samples (bool): Whether to generate gif samples during the training process.
            num_train_samples (int): The number of samples to generate during the training process.
    """
    train_dataset = zip(train_X.batch(batch_size).repeat(), train_y.batch(batch_size).repeat())
    valid_dataset = zip(valid_X.batch(valid_batch_size).repeat(1), valid_y.batch(valid_batch_size).repeat(1))

    # iterate the train dataset
    for iter, train_batch in enumerate(train_dataset):
        if iter > iterations:
            break

        train_disc = True if iter % 2 == 0 else False

        # perform a train step
        gen_loss, disc_loss = pix2pix_train_step(model, train_batch, optimizer, train_disc)

        if iter % valid_steps == 0:
            print("Iter: {}/{} - Train loss: (gen {:.3f}, disc {:.3f})".format(iter, iterations, gen_loss, disc_loss))

        # if the current step is a saving checkpoint step, save the model and add a new frame to the gif samples
        if iter % save_checkpoint_steps == 0:
            print("Iter: {}/{} - Checkpoint reached. Saving the model...".format(iter, iterations))
            model.save_weights(os.path.join(save_checkpoint_path, "model", "gan"))

            if generate_train_samples:
                print("Iter: {}/{} - Generating {} train gif samples with model {}..."
                      .format(iter, iterations, num_train_samples, model.name))

                generate_gif_train_samples(model, num_train_samples,
                                           tf.expand_dims(train_batch[0][:num_train_samples], axis=1),
                                           os.path.join(save_checkpoint_path, "train_samples"),
                                           "[-1,1]")

            valid_gen_mean_loss = tf.keras.metrics.Mean()
            disc_gen_mean_loss = tf.keras.metrics.Mean()

            for (valid_X, valid_y) in valid_dataset:
                gen_loss, disc_loss = pix2pix_loss(model, valid_X, valid_y)
                valid_gen_mean_loss(gen_loss)
                disc_gen_mean_loss(disc_loss)

            end = time.time()

            print("Iter: {}/{} - Train loss: (gen {:.3f}, disc {:.3f}), "
                  "Valid loss: (gen {:.3f}, disc {:.3f}), Time: {:.3f}".
                  format(iter, iterations, gen_loss, disc_loss,
                         valid_gen_mean_loss.result(), disc_gen_mean_loss.result(), 0 if iter == 0 else end - start))

            start = time.time()