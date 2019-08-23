import tensorflow as tf
import numpy as np


def encode_image(base_func):

    def new_base_func(*args, **kwargs):

        # hard-code it for now
        embedding_granularity = 256
        embedding_dim = 10
        embedding_initial_value = np.random.rand(embedding_granularity, embedding_dim).astype(np.float32)
        embedding = tf.Variable(embedding_initial_value)

        self, images, *args = args
        self.images = images
        images = tf.cast(images * 255, tf.int32)
        images = tf.clip_by_value(images, 0, 255)
        encoded = tf.contrib.layers.embedding_lookup_unique(embedding, images)
        shape = encoded.get_shape()
        encoded_images = tf.reshape(encoded, (shape[0], shape[1], shape[2], -1))
        quantized_images = self.activation(encoded_images)

        x = base_func(self, quantized_images, args, **kwargs)

        return x

    return new_base_func
