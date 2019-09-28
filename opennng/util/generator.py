"""
    This module is responsible for generate samples using various models.
"""

import os
import numpy as np
from PIL import Image, ImageSequence


def generate_png_samples(model, num_sample, samples_save_path, normalize):
    """
        This function generates png images, given a model.

        Args:
            model (tf.keras.Model): The model to generate images from.
            num_sample (int): Number of samples to generate.
            samples_save_path (s+tr): The path to save the samples at.
    """
    print("\nGenerating: {} png samples with model {}...".format(num_sample, model.name))

    if not os.path.exists(samples_save_path):
        os.makedirs(samples_save_path)

    # for each latent sample, generate a new image and save it to the given path
    for i in range(num_sample):
        if "GAN" in model.name:
            sample = np.squeeze(model.generate()) * 127.5 + 127.5
        else:
            sample = np.squeeze(model.generate()) * 255

        if len(sample.shape) == 2:
            img = Image.fromarray(sample).convert("L")
        else:
            img = Image.fromarray(sample.astype(np.uint8), "RGB")

        img.save(os.path.join(samples_save_path, "sample_{}.png".format(i+1)), "PNG")

        print("\tSample {}: generated".format(i+1))

    print("Finished!")


def generate_gif_train_samples(model, num_sample, noise, train_samples_path, normalize):
    """
        This function generate gif samples. It is used to observe the progress of the model at training time. The
        function reads a gif image and adds the new frame to the gif.

        Args:
            model (tf.keras.Model): The model that generates the new frame.
            num_sample (int): Number of samples to generate.
            noise (tf.Tensor): The latent sample that is used to generate a new frame. Shape: (num_sample, 1,
                latent_dim).
            train_samples_path (str): The path to the gifs.
    """
    if not os.path.exists(train_samples_path):
        os.makedirs(train_samples_path)

    # for each latent sample, read the gif that corresponds to the sample, generate a new frame, add it to the gif and
    # save the resulted gif to the given path
    for i in range(num_sample):
        if normalize == "[-1, 1]":
            sample = np.squeeze(model.generate(noise[i])) * 127.5 + 127.5
        else:
            sample = np.squeeze(model.generate(noise[i])) * 255

        img_path = os.path.join(train_samples_path, "train_sample_{}.gif".format(i+1))

        if len(sample.shape) == 2:
            new_frame = Image.fromarray(sample).convert("L")
        else:
            new_frame = Image.fromarray(sample.astype(np.uint8), "RGB")

        if os.path.exists(img_path):
            img = Image.open(img_path)

            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            frames.append(new_frame)

            frames[0].save(img_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        else:
            new_frame.save(img_path, save_all=True, append_images=[new_frame], duration=100, loop=0)


