import numpy as np


def padd_images(images, max_height, max_width):
    padded_images = np.zeros((images.shape[0], max_height, max_width))

    for i, image in enumerate(images):
        height_pad = max_height - image.shape[0]
        width_pad = max_width - image.shape[1]

        padded_image = np.pad(image,
                              ((height_pad - height_pad/2, height_pad/2),
                               (width_pad - width_pad/2, width_pad/2)))

        padded_images[i] = padded_image

    return padded_images