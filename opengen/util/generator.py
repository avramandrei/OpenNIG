import os
import numpy as np
from PIL import Image, ImageSequence


def generate_png_samples(model, num_sample, samples_save_path):
    print("\nGenerating: {} png samples with model {}...".format(num_sample, model.name))

    for i in range(num_sample):
        sample = np.squeeze(model.generate()) * 255
        img = Image.fromarray(sample).convert("L")

        img.save(os.path.join(samples_save_path, "sample_{}.png".format(i+1)), "PNG")

        print("\tSample {}: generated".format(i+1))

    print("Finished!")


def generate_gif_train_samples(model, num_sample, noise, train_samples_path):
    if not os.path.exists(train_samples_path):
        os.makedirs(train_samples_path)

    for i in range(num_sample):
        sample = np.squeeze(model.generate(noise[i])) * 255

        img_path = os.path.join(train_samples_path, "train_sameple_{}.gif".format(i+1))
        new_frame = Image.fromarray(sample).convert("L")

        if os.path.exists(img_path):
            img = Image.open(img_path)

            frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
            frames.append(new_frame)

            frames[0].save(img_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        else:
            new_frame.save(img_path, save_all=True, append_images=[new_frame], duration=100, loop=0)


