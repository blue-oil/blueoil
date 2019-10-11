import os
import re
import sys
import glob
import tqdm
import imageio
import functools
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.eager as tfe

import matplotlib.pyplot as plt

tfe.enable_eager_execution()

sys.path.extend(["./lmnet", "/dlk/python/dlk"])


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_samples", type=int, required=True)
args = parser.parse_args()


class Image(slim.tfexample_decoder.ItemHandler):
    def __init__(self, image_key=None, format_key=None, shape=None, channels=3,
                 dtype=tf.uint8, repeated=False):
        """Initializes the image.
        Args:
          image_key: the name of the TF-Example feature in which the encoded image
            is stored.
          shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
          channels: the number of channels in the image.
          dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
              See tf.image.decode_image,
                  tf.decode_raw,
          repeated: if False, decodes a single image. If True, decodes a
            variable number of image strings from a 1D tensor of strings.
        """
        if not image_key:
            image_key = 'image/encoded'

        super(Image, self).__init__([image_key])
        self._image_key = image_key
        self._shape = shape
        self._channels = channels
        self._dtype = dtype
        self._repeated = repeated

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]

        if self._repeated:
            return functional_ops.map_fn(
                lambda x: self._decode(x), image_buffer, dtype=self._dtype)
        else:
            return self._decode(image_buffer)

    def _decode(self, image_buffer):
        """Decodes the image buffer.
        Args:
          image_buffer: The tensor representing the encoded image tensor.
        Returns:
          A tensor that represents decoded image of self._shape, or
          (?, ?, self._channels) if self._shape is not specified.
        """
        def decode_raw():
            """Decodes a raw image."""
            return tf.decode_raw(image_buffer, out_type=self._dtype)
        image = decode_raw()
        if self._shape is not None:
            image = tf.reshape(image, self._shape)
        return image


def __get_dataset(data_path, num_samples, image_size=(384, 512)):
    with tf.name_scope('__get_dataset'):
        height, width = image_size
        reader = tf.TFRecordReader
        keys_to_features = {
            'image_a': tf.FixedLenFeature((), tf.string),
            'image_b': tf.FixedLenFeature((), tf.string),
            'flow': tf.FixedLenFeature((), tf.string)
        }
        items_to_handlers = {
            'image_a': Image(
                image_key='image_a', dtype=tf.float32,
                shape=[height, width, 3], channels=3),
            'image_b': Image(
                image_key='image_b', dtype=tf.float32,
                shape=[height, width, 3], channels=3),
            'flow': Image(
                image_key='flow', dtype=tf.float32,
                shape=[height, width, 2], channels=2),
        }
        items_to_descriptions = {
            'image_a': 'A 3-channel image.',
            'image_b': 'A 3-channel image.',
            'flow': 'A 2-channel optical flow field'
        },
        decoder = slim.tfexample_decoder.TFExampleDecoder(
            keys_to_features, items_to_handlers)
    return slim.dataset.Dataset(
        data_sources=data_path, reader=reader, decoder=decoder,
        num_samples=num_samples, items_to_descriptions=items_to_descriptions)


# if __name__ == '__main__':
#     num_threads = 32
#     with tf.name_scope('load_batch'):
#         reader_kwargs = {'options': tf.python_io.TFRecordOptions(
#             tf.python_io.TFRecordCompressionType.ZLIB)}
#         dataset = __get_dataset(args.data_path, args.num_samples)
#         print(dataset)
#         data_provider = slim.dataset_data_provider.DatasetDataProvider(
#             dataset, num_readers=num_threads, common_queue_capacity=2048,
#             common_queue_min=1024, reader_kwargs=reader_kwargs)
#         image_a, image_b, flow = data_provider.get(['image_a', 'image_b', 'flow'])
#         image_a, image_b, flow = map(tf.to_float, [image_a, image_b, flow])
#         print(image_a.eval())

if __name__ == '__main__':
    raw_dataset = tf.data.TFRecordDataset(
        [args.data_path], compression_type="ZLIB")
    height, width = 384, 512
    items_to_handlers = {
        'image_a': tf.FixedLenFeature([], tf.string, default_value=''),
        'image_b': tf.FixedLenFeature([], tf.string, default_value=''),
        'flow': tf.FixedLenFeature([], tf.string, default_value=''),
    }

    def _parse_function(example_proto):
        return tf.parse_single_example(example_proto, items_to_handlers)

    parsed_dataset = raw_dataset.map(_parse_function)
    print(parsed_dataset[99])
    # for parsed_features in parsed_dataset.take(100):
    #     print(type(parsed_features["image_a"].numpy())
#     example = next(tf.python_io.tf_record_iterator(args.data_path))
#     x = tf.train.Example.FromString(example)
#     print(x)
