# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import csv
import math
import os
import xml.etree.ElementTree as ET

import tensorflow as tf
import tensorflow_datasets as tfds


class Pascalvoc(tfds.core.GeneratorBasedBuilder):
    """Generic TFDS builder for object detection dataset"""
    VERSION = tfds.core.Version("0.1.0")

    def __init__(self, dataset_name, raw_data_path, **kwargs):
        self.name = dataset_name
        self.raw_data_path = raw_data_path
        super().__init__(**kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Generic TFDS builder for object detection dataset",
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(),
                "objects": tfds.features.SequenceDict({
                    "label": tfds.features.ClassLabel(),
                    "bbox": tfds.features.BBoxFeature(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        data_path = self.raw_data_path
        classes = []
        class_files_pattern = os.path.join(data_path, "ImageSets/Main/*_trainval.txt")
        for filepath in tf.io.gfile.glob(class_files_pattern):
            class_name = os.path.basename(filepath).replace("_trainval.txt", "")
            classes.append(class_name)

        self.info.features["objects"]["label"].names = sorted(classes)

        available_splits = {
            "train": tfds.Split.TRAIN,
            "val": tfds.Split.VALIDATION,
            "test": tfds.Split.TEST,
            "trainval": "trainval"
        }

        splits = []
        set_files_pattern = os.path.join(data_path, "ImageSets/Main/*.txt")
        for filepath in tf.io.gfile.glob(set_files_pattern):
            split_name = os.path.basename(filepath).replace(".txt", "")

            if split_name in available_splits:
                splits.append(
                    tfds.core.SplitGenerator(
                        name=available_splits[split_name],
                        num_shards=self._num_shards(data_path, split_name),
                        gen_kwargs=dict(data_path=data_path, split_name=split_name)
                    )
                )

        return splits

    def _num_shards(self, data_path, split_name):
        total_size = 0
        max_shard_size = 256 * 1024 * 1024 # 256MiB
        for image_file in self._image_files(data_path, split_name):
            total_size += tf.io.gfile.Gfile(image_file).size()

        return int(math.ceil(total_size / max_shard_size))

    def _image_files(self, data_path, split_name):
        set_filepath = os.path.join(data_path, "ImageSets/Main/{}.txt".format(split_name))
        with tf.io.gfile.GFile(set_filepath) as f:
            for line in f:
                image_id = line.strip()
                yield self._image_file(data_path, image_id)

    def _image_file(self, data_path, image_id):
        anno_filepath = os.path.join(data_path, "Annotations/{}.xml".format(image_id))
        with tf.io.gfile.GFile(anno_filepath) as f:
            xml = ET.ElementTree(file=f)
            return os.path.join(data_path, "JPEGImages", xml.find("filename").text)

    def _generate_examples(self, data_path, split_name):
        set_filepath = os.path.join(data_path, "ImageSets/Main/{}.txt".format(split_name))
        with tf.io.gfile.GFile(set_filepath) as f:
            for line in f:
                image_id = line.strip()
                yield self._generate_example(data_path, image_id)

    def _generate_example(self, data_path, image_id):
        anno_filepath = os.path.join(data_path, "Annotations/{}.xml".format(image_id))

        objects = []
        with tf.io.gfile.GFile(anno_filepath) as f:
            xml = ET.ElementTree(file=f)
            image_filepath = os.path.join(data_path, "JPEGImages", xml.find("filename").text)

            for obj in xml.findall("object"):
                label = obj.find("name").text
                width = int(xml.find("size").find("width").text)
                height = int(xml.find("size").find("height").text)

                xmin = float(obj.find("bndbox").find("xmin").text)
                ymin = float(obj.find("bndbox").find("ymin").text)
                xmax = float(obj.find("bndbox").find("xmax").text)
                ymax = float(obj.find("bndbox").find("ymax").text)

                bbox = tfds.features.BBox(
                    ymin / height,
                    xmin / width,
                    ymax / height,
                    xmax / width,
                )

                objects.append({
                    "label": label,
                    "bbox": bbox,
                })

        return {
            "image": image_filepath,
            "objects": objects
        }
