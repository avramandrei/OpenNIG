"""

This module implements Generative Adversarial Networks (GANs) that use convolutional layers: ConvGANSmall, ConvGANMedium,
ConvGANBig.

Original paper: https://arxiv.org/abs/1406.2661.

"""

import tensorflow as tf


class DCGANBase(tf.keras.Model):
    """
        This is the base class for all the Convolutional GANs. It is composed of two submodels, a generative network
        (generative_net) and a discriminative network (discriminative_net) whose architecture is defined in each
        subclass.

        The generative network transforms the latent sample space (noise) into images. Input shape: (batch_size,
        latent_dim), output_shape: (batch_size, height, width, depth).

        The discriminative network classifies an image in either real or fake. Input shape: (batch_size, height, width,
        depth), output_shape: (batch_size, 1).
    """
    def __init__(self, input_shape):
        super(DCGANBase, self).__init__()

        self.build(input_shape)

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


class DCGANSmall(DCGANBase):
    """
        This class is the small version of the Convolutional GAN.
    """
    def __init__(self, input_shape):
        super(DCGANSmall, self).__init__(input_shape)

        self.latent_dim = 100

        if input_shape[0] % 4 == 0:
            gen_input_height = int(input_shape[0]/4)
        else:
            raise ValueError("First dimension of the input data must be divisible by 4")

        if input_shape[1] % 4 == 0:
            gen_input_width = int(input_shape[1] / 4)
        else:
            raise ValueError("Second dimension of the input data must be divisible by 4")

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.latent_dim),

                tf.keras.layers.Dense(gen_input_height * gen_input_width * 256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Reshape((gen_input_height, gen_input_width, 256)),

                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=(1, 1),
                                                padding="SAME", activation="tanh", use_bias=False)
            ],
            name="generative_net"
        )

        self.discriminative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                tf.keras.layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding="SAME", use_bias=False),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv2D(32, kernel_size=3, strides=(1, 1), padding="SAME", use_bias=False),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ],
            name="discriminative_net"
        )


class DCGANMedium(DCGANBase):
    """
        This class is the medium version of the Convolutional GAN.
    """
    def __init__(self, input_shape):
        super(DCGANMedium, self).__init__(input_shape)

        self.latent_dim = 100

        if input_shape[0] % 8 == 0:
            gen_input_height = int(input_shape[0]/8)
        else:
            raise ValueError("First dimension of the input data must be divisible by 8")

        if input_shape[1] % 8 == 0:
            gen_input_width = int(input_shape[1] / 8)
        else:
            raise ValueError("Second dimension of the input data must be divisible by 8")
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.latent_dim),

                tf.keras.layers.Dense(gen_input_height * gen_input_width * 128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Reshape((gen_input_height, gen_input_width, 128)),

                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2, 2), padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=5, strides=(2, 2), padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=5, strides=(2, 2), padding="SAME", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),

                tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=5, strides=(1, 1),
                                                padding="SAME", activation="tanh", use_bias=False)
            ],
            name="generative_net"
        )

        self.discriminative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                tf.keras.layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="SAME"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv2D(128, kernel_size=5, strides=(2, 2), padding="SAME"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Conv2D(64, kernel_size=5, strides=(2, 2), padding="SAME"),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1),
            ],
            name="discriminative_net"
        )



