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

from PIL import Image, ImageEnhance, ImageFilter

sys.path.extend(["./lmnet", "/dlk/python/dlk"])

import matplotlib.pyplot as plt
from lmnet.networks.optical_flow_estimation.flow_to_image import *
from lmnet.networks.optical_flow_estimation.pre_processor import *
from lmnet.networks.optical_flow_estimation.data_augmentor import *


parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, required=True)
parser.add_argument("--load_dir", type=str,
                    default="./dataset/")
parser.add_argument("--save_dir", type=str,
                    default="./dataset/")
parser.add_argument("--max_size", type=int, default=None)
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()


@functools.lru_cache(maxsize=None)
def open_image_file(file_name, dtype=np.uint8):
    return np.array(Image.open(file_name), dtype=dtype)


@functools.lru_cache(maxsize=None)
def open_flo_file(file_name, dtype=np.float32):
    with open(file_name, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, \
            "Magic number incorrect. Invalid .flo file"
        width = np.fromfile(f, np.int32, count=1)[0]
        height = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, dtype, count=2 * width * height)
    return np.resize(data, (height, width, 2))


@functools.lru_cache(maxsize=None)
def open_pfm_file(file_name, dtype=np.float32):
    color, width, height, scale, endian = None, None, None, None, None
    with open(file_name, "rb") as f:
        # loading header information
        header = f.readline().rstrip().decode("utf-8")
        assert header == "PF" or header == "Pf", "Not a PFM file."
        color = (header == "PF")

        # loading wdth and height information
        dim_match = re.match(r"^(\d+)\s(\d+)\s$", f.readline().decode("utf-8"))
        assert dim_match is not None, "Malformed PFM header."
        width, height = map(int, dim_match.groups())

        scale = float(f.readline().rstrip().decode("utf-8"))
        if scale < 0:
            endian = "<"
            scale = -scale
        else:
            endian = ">"
        data = np.fromfile(f, endian + "f").astype(dtype)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)[::-1, :, :2]
    data[:, :, 1] *= -1
    return data


def create_tf_exapmle(image_a_path, image_b_path, flow_path):
    image_a = open_image_file(image_a_path)
    image_b = open_image_file(image_b_path)
    root_dir, ext = os.path.splitext(flow_path)
    if ext == ".flo":
        flow = open_flo_file(flow_path)
    elif ext == ".pfm":
        flow = open_pfm_file(flow_path)
    data = {
        "image": np.concatenate([image_a, image_b], axis=2),
        "label": flow
    }

    augmentor_list = [
        # Geometric transformation
        # FlipLeftRight(0.5),
        # FlipTopBottom(0.5),
        Translate(-0.1, 0.1),
        Rotate(-10, +10),
        # Scale(1.0, 2.0),
        # Pixel-wise augmentation
        Brightness(0.8, 1.2),
        Contrast(0.2, 1.4),
        Color(0.5, 2.0),
        Gamma(0.7, 1.5),
        # Hue(-128.0, 128.0),
        GaussianNoise(0.0, 10.0),
        DevideBy255(),
        DiscretizeFlow(0.01, 10)
    ]

    for _augmentor in augmentor_list:
        data = _augmentor(**data)

    image_a_p = data["image"][..., :3]
    image_b_p = data["image"][..., 3:]

    image_f = flow_to_image(flow)
    # image_f_p = flow_to_image(data["label"])
    # image_f = discretized_flow_to_image(flow, 10)
    image_f_p = discretized_flow_to_image(data["label"], 10)

    plt.figure(1, figsize=[8, 12])
    plt.subplot(3, 2, 1)
    plt.imshow(image_a)
    plt.subplot(3, 2, 2)
    plt.imshow(image_a_p)
    plt.subplot(3, 2, 3)
    plt.imshow(image_b)
    plt.subplot(3, 2, 4)
    plt.imshow(image_b_p)
    plt.subplot(3, 2, 5)
    plt.imshow(image_f)
    plt.subplot(3, 2, 6)
    plt.imshow(image_f_p)
    plt.show()


def write_tfrecord(file_list, save_name):
    write_opts = tf.python_io.TFRecordOptions(
        tf.python_io.TFRecordCompressionType.ZLIB)
    with tf.python_io.TFRecordWriter(save_name, options=write_opts) as writer:
        # with tf.python_io.TFRecordWriter(save_name) as writer:
        pbar = tqdm.tqdm(file_list[:args.max_size])
        for image_a_path, image_b_path, flow_path in pbar:
            pbar.set_description("loading {}".format(
                os.path.basename(flow_path)))
            create_tf_exapmle(image_a_path, image_b_path, flow_path)


def build_flying_chairs():
    data_dir = "{}/FlyingChairs".format(args.load_dir)
    train_val_split = np.loadtxt(
        "{}/FlyingChairs_train_val.txt".format(data_dir))
    train_idxs = np.flatnonzero(train_val_split == 1)
    valid_idxs = np.flatnonzero(train_val_split == 2)

    training_list = []
    for index in train_idxs:
        prefix = "{}/data/{:05d}".format(data_dir, index + 1)
        image_a_path = "{}_img1.ppm".format(prefix)
        image_b_path = "{}_img2.ppm".format(prefix)
        flow_path = "{}_flow.flo".format(prefix)
        training_list.append([image_a_path, image_b_path, flow_path])
    save_name = "{}/train.tfrecord".format(data_dir)
    write_tfrecord(training_list, save_name)

    validation_list = []
    for index in valid_idxs:
        prefix = "{}/data/{:05d}".format(data_dir, index + 1)
        image_a_path = "{}_img1.ppm".format(prefix)
        image_b_path = "{}_img2.ppm".format(prefix)
        flow_path = "{}_flow.flo".format(prefix)
        validation_list.append([image_a_path, image_b_path, flow_path])
    save_name = "{}/valid.tfrecord".format(data_dir)
    write_tfrecord(validation_list, save_name)


def build_chairs_sdhom():
    data_dir = "{}/ChairsSDHom/".format(args.load_dir)

    image_a_list = sorted(glob.glob("{}/data/train/t0/*.png".format(data_dir)))
    image_b_list = sorted(glob.glob("{}/data/train/t1/*.png".format(data_dir)))
    flow_list = sorted(glob.glob("{}/data/train/flow/*.pfm".format(data_dir)))
    training_list = list(zip(image_a_list, image_b_list, flow_list))
    save_name = "{}/train.tfrecord".format(data_dir)
    # print(training_list)
    write_tfrecord(training_list, save_name)

    image_a_list = sorted(glob.glob("{}/data/test/t0/*.png".format(data_dir)))
    image_b_list = sorted(glob.glob("{}/data/test/t1/*.png".format(data_dir)))
    flow_list = sorted(glob.glob("{}/data/test/flow/*.pfm".format(data_dir)))
    validation_list = list(zip(image_a_list, image_b_list, flow_list))
    # print(validation_list)
    save_name = "{}/test.tfrecord".format(data_dir)
    write_tfrecord(validation_list, save_name)


if __name__ == "__main__":
    os.makedirs(args.save_dir, exist_ok=True)
    eval("build_{}".format(args.target))()
