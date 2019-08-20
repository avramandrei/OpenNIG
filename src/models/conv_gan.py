import tensorflow as tf


class ConvGANBase(tf.keras.Model):
    def __init__(self, input_shape):
        super(ConvGANBase, self).__init__()

        self.build(input_shape)

    def generate(self):
        noise = tf.random.normal([1, self.noise_size])
        return self.generative_net(noise)

    def call(self, inputs, training=None, mask=None):
        pass

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.generative_net.summary()
        self.discriminative_net.summary()


class ConvGANSmall(ConvGANBase):
    def __init__(self, input_shape):
        super(ConvGANSmall, self).__init__(input_shape)

        self.noise_size = 100
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.noise_size),

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

