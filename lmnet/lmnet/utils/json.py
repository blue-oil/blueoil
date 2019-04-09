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
import os
from io import BytesIO
import base64
from datetime import datetime, timezone
import json

import numpy as np
import PIL.Image
import PIL.ImageDraw
from matplotlib import cm

from lmnet.common import Tasks


class JsonOutput():
    """Create callable instance to output predictions json object from post processed tensor(np.ndarray).

    The output predictions json format depends on task type.
    Plsease see [Output Data Specification](https://github.com/LeapMind/lmnet/wiki/Output-Data-Specification).
    """

    def __init__(self, task, classes, image_size, data_format):
        assert task in Tasks
        self.task = task
        self.classes = classes
        self.image_size = image_size
        self.data_format = data_format

    def _classification(self, outputs, raw_images, image_files):
        assert outputs.shape == (len(image_files), len(self.classes))

        results = []

        for output, raw_image, image_file in zip(outputs, raw_images, image_files):
            result_per_batch = {
                "file_path": image_file,
                "prediction": None,
            }

            prediction = []
            for i, class_name in enumerate(self.classes):
                prediction.append({
                    "class": {"id": i, "name": class_name},
                    "probability": str(output[i])
                })

            result_per_batch["prediction"] = prediction

            results.append(result_per_batch)

        return results

    def _object_detection(self, outputs, raw_images, image_files):
        results = []

        for output, raw_image, image_file in zip(outputs, raw_images, image_files):
            height_scale = raw_image.shape[0] / self.image_size[0]
            width_scale = raw_image.shape[1] / self.image_size[1]

            predict_boxes = np.copy(output)
            predict_boxes[:, 0] *= width_scale
            predict_boxes[:, 1] *= height_scale
            predict_boxes[:, 2] *= width_scale
            predict_boxes[:, 3] *= height_scale

            result_per_batch = {
                "file_path": image_file,
                "prediction": None,
            }
            prediction_per_batch = []
            for predict_box in predict_boxes:
                class_id = int(predict_box[4])
                score = predict_box[5]
                box = [x for x in predict_box[:4]]
                class_name = self.classes[class_id]

                prediction = {
                    "class": {
                        "id": class_id,
                        "name": class_name,
                    },
                    "score": str(score),
                    "box": box,
                }
                prediction_per_batch.append(prediction)
            result_per_batch["prediction"] = prediction_per_batch

            results.append(result_per_batch)

        return results

    def _semantic_segmentation(self, outputs, raw_images, image_files):
        results = []
        for output, raw_image, image_file in zip(outputs, raw_images, image_files):

            if self.data_format == "NCHW":
                output = np.transpose(output, [1, 2, 0])

            result_per_batch = {
                "file_path": image_file,
                "prediction": None,
            }

            prediction = []
            for i, class_name in enumerate(self.classes):
                img = output[:, :, i] * 255
                img = PIL.Image.fromarray(img, mode="F")
                img = img.convert("L")
                img = img.resize((raw_image.shape[1], raw_image.shape[0]))

                # base64 encode
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                encoded = base64.b64encode(buffered.getvalue())
                encoded = encoded.decode("ascii")

                prediction.append({
                    "class": {"id": i, "name": class_name},
                    "mask": encoded,
                })

            result_per_batch["prediction"] = prediction

            results.append(result_per_batch)

        return results

    def __call__(self, outputs, raw_images, image_files):
        """Output predictions json object from post processed tensor(np.ndarray).

        Args:
            outputs(np.ndarray or list): Post processed tensor.
            raw_images(list): List of np.ndarray of raw (non pre-processed) images.
            image_files(list): List of image file paths.
        """

        assert len(outputs) == len(raw_images)
        assert len(outputs) == len(image_files)

        result_json = {
            "version": 0.2,
            "task": str(self.task.value),
            "classes": [{"id": i, "name": class_name} for i, class_name in enumerate(self.classes)],
            "date": datetime.now(timezone.utc).isoformat(),
            "results": [],
        }

        if self.task == Tasks.CLASSIFICATION:

            results = self._classification(outputs, raw_images, image_files)

        if self.task == Tasks.OBJECT_DETECTION:

            results = self._object_detection(outputs, raw_images, image_files)

        if self.task == Tasks.SEMANTIC_SEGMENTATION:

            results = self._semantic_segmentation(outputs, raw_images, image_files)

        result_json["results"] = results
        result_json = json.dumps(result_json, indent=4, sort_keys=True)
        return result_json


class ImageFromJson():
    """Create callable instance to return list of tuple (file_name, PIL image object) from prediction json."""

    def __init__(self, task, classes, image_size):
        assert task in Tasks
        self.task = task
        self.classes = classes
        self.image_size = image_size

    def _classification(self, result_json, raw_images, image_files):
        outputs = json.loads(result_json)
        results = outputs["results"]
        filename_images = []

        for result, raw_image, image_file in zip(results, raw_images, image_files):
            predictions = result["prediction"]

            probs = [prediction["probability"] for prediction in predictions]
            highest_index = probs.index(max(probs))
            highest = predictions[highest_index]

            class_dir = highest["class"]["name"]
            base, _ = os.path.splitext(os.path.basename(image_file))
            filename = os.path.join(class_dir, "{}.png".format(base))
            image = PIL.Image.fromarray(raw_image)

            filename_images.append((filename, image))

        return filename_images

    def _semantic_segmentation(self, result_json, raw_images, image_files):
        outputs = json.loads(result_json)
        results = outputs["results"]
        filename_images = []

        for result, raw_image, image_file in zip(results, raw_images, image_files):
            base, _ = os.path.splitext(os.path.basename(image_file))

            out_file = os.path.join("mask", "{}.png".format(base))
            out_overlap_file = os.path.join("overlap", "{}.png".format(base))

            masks = []
            for i, class_name in enumerate(self.classes):
                mask_data = base64.b64decode(result["prediction"][i]["mask"])
                mask_image = PIL.Image.open(BytesIO(mask_data))
                masks.append(np.array(mask_image))

            # shape is (height, width, num_class)
            masks = np.stack(masks, axis=2)
            argmax = np.argmax(masks, axis=2)

            color_maps = (np.array(cm.tab20.colors) * 255).tolist()

            result = []

            output_image = np.zeros_like(raw_image)
            for i, class_name in enumerate(self.classes):
                color = color_maps[i % len(self.classes)]
                output_image[argmax == i] = color

            output_pil = PIL.Image.fromarray(output_image)
            filename_images.append((out_file, output_pil))

            overlap_image = 0.5 * raw_image + output_image * 0.5
            overlap_image = overlap_image.astype(np.uint8)
            overlap = PIL.Image.fromarray(overlap_image)

            filename_images.append((out_overlap_file, overlap))

        return filename_images

    def _object_detection(self, result_json, raw_images, image_files):
        outputs = json.loads(result_json)
        results = outputs["results"]
        filename_images = []

        for i, (result, raw_image, image_file) in enumerate(zip(results, raw_images, image_files)):
            base, _ = os.path.splitext(os.path.basename(image_file))
            file_name = "{}.png".format(base)

            image = PIL.Image.fromarray(raw_image)
            draw = PIL.ImageDraw.Draw(image)

            predictions = result["prediction"]
            color_maps = (np.array(cm.tab20.colors) * 255).astype(np.uint8).tolist()

            for prediction in predictions:
                box = prediction["box"]
                xy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
                top_left = [box[0], box[1]]
                class_id = prediction["class"]["id"]
                class_name = prediction["class"]["name"]
                score = prediction["score"]

                color = tuple(color_maps[class_id % len(self.classes)])

                draw.rectangle(xy, outline=color)
                txt = "class: {:s}, score: {:.3f}".format(class_name, float(score))
                draw.text(top_left, txt, fill=color)

            filename_images.append((file_name, image))

        return filename_images

    def __call__(self, json_results, raw_images, image_files):
        outputs = json.loads(json_results)
        results = outputs["results"]
        assert len(results) == len(raw_images) == len(image_files)

        if self.task == Tasks.CLASSIFICATION:
            filename_images = self._classification(json_results, raw_images, image_files)

        if self.task == Tasks.SEMANTIC_SEGMENTATION:
            filename_images = self._semantic_segmentation(json_results, raw_images, image_files)

        if self.task == Tasks.OBJECT_DETECTION:
            filename_images = self._object_detection(json_results, raw_images, image_files)

        return filename_images
