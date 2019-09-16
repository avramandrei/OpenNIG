import tensorflow as tf
import time
import os
from opennng.util.generator import generate_gif_train_samples


def train_model(model, train_step, loss_fcn,
                train_dataset, eval_dataset,
                optimizer, iterations, batch_size, save_checkpoint_steps, save_checkpoint_path,
                eval_batch_size, eval_steps,
                generate_train_samples, num_train_samples):

    """
        This function is used to train a model.

        Args:
            model (tf.keras.Model): The model to be trained.
            train_step: The train step of the model.
            loss_fcn: The loss function used by the model.
            train_dataset (tf.data.Dataset): The dataset used for training the model.
            eval_dataset (tf.data.Dataset): The dataset used for evaluating the model.
            optimizer: The optimizer used for train the model.
            iterations (int): The number of iterations that the model will be trained on.
            batch_size (int): The batch size of the training process.
            save_checkpoint_steps (int): A checkpoint will be generated every this many steps.
            save_checkpoint_path (str): The checkpoint path.
            eval_batch_size (int): The batch size of the evaluation process.
            eval_steps (int): An evaluation of the model will be performed every this many steps.
            generate_train_samples (bool): Whether to generate gif samples during the training process.
            num_train_samples (int): The number of samples to generate during the training process.
    """
    # process the data if it has not already been processed
    if not processed:
        pass

    train_dataset = train_dataset.batch(batch_size).repeat()
    eval_dataset = eval_dataset.batch(eval_batch_size).repeat(1)

    # generate a noise (latent sample) from where train samples will be created
    if generate_train_samples:
        noise = tf.random.normal(shape=[num_train_samples, 1, model.latent_dim], seed=42)

    # iterate the train dataset
    for iter, train_batch in enumerate(train_dataset):
        if iter > iterations:
            break

        # perform a train step
        train_loss = train_step(model, train_batch, optimizer)

        # if the current step is a saving checkpoint step, save the model and add a new frame to the gif samples
        if iter % save_checkpoint_steps == 0:
            print("Iter: {}/{} - Checkpoint reached. Saving the model...".format(iter, iterations))
            model.save_weights(os.path.join(save_checkpoint_path, "model", "model_iter_{}".format(iter)))

            if generate_train_samples:
                print("Iter: {}/{} - Generating {} train gif samples with model {}..."
                      .format(iter, iterations, num_train_samples, model.name))

                generate_gif_train_samples(model, num_train_samples,
                                           noise, os.path.join(save_checkpoint_path, "train_samples"))

        # if the current step is an evaluation step, evaluate the model
        if iter % eval_steps == 0:
            loss_mean = tf.keras.metrics.Mean()
            for eval_batch in eval_dataset:
                loss_mean(loss_fcn(model, eval_batch))

            end = time.time()

            print("Iter: {}/{} - Train loss: {}, Eval loss: {}, Time: {}".
                  format(iter, iterations, train_loss, loss_mean.result(), 0 if iter == 0 else end-start))

            start = time.time()
