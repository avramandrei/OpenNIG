import tensorflow as tf


class Pix2Pix(tf.keras.Model):
    def __init__(self, input_shape):
        super(Pix2Pix, self).__init__()

        self.generative_net = self.generator(input_shape[3])

        self.discriminative_net = self.discriminator()

        self.build(input_shape)

    def _upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def _downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def generator(self, output_channels):
        down_stack = [
            self._downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
            self._downsample(128, 4),  # (bs, 64, 64, 128)
            self._downsample(256, 4),  # (bs, 32, 32, 256)
            self._downsample(512, 4),  # (bs, 16, 16, 512)
            self._downsample(512, 4),  # (bs, 8, 8, 512)
            self._downsample(512, 4),  # (bs, 4, 4, 512)
            self._downsample(512, 4),  # (bs, 2, 2, 512)
            self._downsample(512, 4),  # (bs, 1, 1, 512)
        ]

        up_stack = [
            self._upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
            self._upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
            self._upsample(512, 4),  # (bs, 16, 16, 1024)
            self._upsample(256, 4),  # (bs, 32, 32, 512)
            self._upsample(128, 4),  # (bs, 64, 64, 256)
            self._upsample(64, 4),  # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(output_channels, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=[None, None, 3])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x, name="generative_net")

    def discriminator(self, ):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (bs, 256, 256, channels*2)

        down1 = self._downsample(64, 4, False)(x)  # (bs, 128, 128, 64)
        down2 = self._downsample(128, 4)(down1)  # (bs, 64, 64, 128)
        down3 = self._downsample(256, 4)(down2)  # (bs, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def generate(self, X):
        return self.generative_net(X)

