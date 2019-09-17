"""

This module implements Generative Adversarial Networks (GANs) that use convolutional layers: ConvGANSmall, ConvGANMedium,
ConvGANBig.

Original paper: https://arxiv.org/abs/1406.2661.

"""

import tensorflow as tf


class ConvGANBase(tf.keras.Model):
    """
        This is the base class for all the Convolutional GANs. It is composed of two submodels, a generative network
        (generative_net) and a discriminative network (discriminative_net) whose architecture is defined in each
        subclass.

        The generative network transforms the latent sample space (noise) into images. Input shape: (batch_size,
        latent_dim), output_shape: (batch_size, height, width, depth).

        The discriminative network classifies an image in either real or fake. Input shape: (batch_size, height, width,
        depth), output_shape: (batch_size, 1).
    """
    def __init__(self):
        super(ConvGANBase, self).__init__()

        self.build()

    def generate(self, noise=None):
        """
            This function generates an observation from the latent sample space, using the generative network. The
            function can receive the value of the sample (noise) or generate one.

            Args:
                noise (tf.Tensor): Tensor that contains the value of the latent sample. Shape: (1, latent_dim).

            Returns:
                A tensor that contains the observed sample. Shape: (batch_size, height, width, depth).
        """
        if noise is None:
            noise = tf.random.normal([1, self.latent_dim])
        return self.generative_net(noise)

    def call(self, inputs, training=None, mask=None):
        pass

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.generative_net.summary()
        self.discriminative_net.summary()


class ConvGANSmall(ConvGANBase):
    """
        This class is the small version of the Convolutional GAN.
    """
    def __init__(self, input_shape):
        super(ConvGANSmall, self).__init__(input_shape)

        self.latent_dim = 100

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.latent_dim),

                tf.keras.layers.Dense(7 * 7 * 256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape((7, 7, 256)),

                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(2, 2), padding="SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2), padding="SAME"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),

                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid")
            ],
            name="generative_net"
        )

        self.discriminative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding="SAME"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding="SAME"),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ],
            name="discriminative_net"
        )



