import tensorflow as tf


class ConvVAEBase(tf.keras.Model):
    def __init__(self, input_shape):
        super(ConvVAEBase, self).__init__()

        self.build(input_shape)

    @tf.function
    def generate(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(1, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
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
