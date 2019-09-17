"""

This module implements Variational Autoencoders (VAE) that use convolutional layers: ConvVAESmall, ConvVAEMedium,
ConvVAEBig.

Original paper: https://arxiv.org/abs/1312.6114.

"""

import tensorflow as tf


class ConvVAEBase(tf.keras.Model):
    """
        This is the base class for all the Convolutional VAEs. It is composed of two submodels, an inference network
        (inference_net) and a generative network (generative_net) whose architecture is defined in each
        subclass.

        The inference network transforms the images into a latent normal distribution. Input_shape: (batch_size, height,
        width, depth), output_shape: (batch_size, latent_dim).

        The generative network takes a sample from the normal latent sample space and creates an image from it.
        Input_shape: (batch_size, latent_dim), output_shape: (batch_size, height, width, depth).
    """

    def __init__(self, input_shape):
        super(ConvVAEBase, self).__init__()

        if input_shape is not None:
            self.build(input_shape)

    @tf.function
    def generate(self, noise=None):
        """
            This function generates an observation from the latent sample space, using the generative network. The
            function can receive the value of the sample (noise) or generate one.

            Args:
                noise (tf.Tensor): Tensor that contains the value of the latent sample. Shape: (batch_size, latent_dim).

            Returns:
                A tensor that contains the observed sample. Shape: (batch_size, height, width, depth).
        """
        if noise is None:
            noise = tf.random.normal(shape=(1, self.latent_dim))

        return self.decode(noise, apply_sigmoid=True)

    def encode(self, x):
        """
            This function encodes an image in the set of mean and log-variation of the latent normal distribution.

            Args:
                 x (tf.Tensor): The input image. Shape: (batch_size, height, width, depth).

            Returns:
                The mean and the log-variance of the latent distribution. Shape: (batch_size, latent_dim)
        """
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        """
            This function is the reparametrization trick that allows us to sample from the latent distribution and use
            the backpropagation algorithm.

            Args:
                mean (tf.Tensor): A tensor that contains the mean of the latent distribution. Shape: (batch_size,
                latent_dim).
                logvar (tf.Tensor): A tensor that contains the log-variance of the latent distribution. Shape:
                (batch_size, latent_dim).

            Returns:
                A sample from the latent distribution. Shape: (batch_size, latent_dim).
        """
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        """
            This is the function that constructs the output image from the sample of the latent distribution.

            Args:
                z (tf.Tensor): A tensor that contains the latent sample. Shape: (batch_size, latent_dim).
                apply_sigmoid (bool): Whether to apply sigmoid or not to the logits.

            Returns:
                The output image. Shape: (batch_size, height, width, depth).
        """

        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits

    def call(self, inputs, training=None, mask=None):
        pass

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.inference_net.summary()
        self.generative_net.summary()


class ConvVAESmall(ConvVAEBase):
    """
        This class is the small version of the Convolutional VAE.
    """

    def __init__(self, input_shape):
        super(ConvVAESmall, self).__init__(input_shape)

        self.latent_dim = 50
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ],
            name="inference_network"
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),

                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ],
            name="generative_network"
        )


class ConvVAEMedium(ConvVAEBase):
    """
        This class is the medium version of the Convolutional VAE.
    """
    def __init__(self, input_shape):
        super(ConvVAEMedium, self).__init__(input_shape)

        self.latent_dim = 100
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=input_shape),

                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=(2, 2), activation='relu'),

                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ],
            name="inference_network"
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 64)),

                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ],
            name="generative_network"
        )
