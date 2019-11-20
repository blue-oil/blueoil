# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
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
"""Inference results visualization (decoration) functions and helpers."""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from lmnet.common import get_color_map

FONT = "DejaVuSans.ttf"


def visualize_classification(image, post_processed, config):
    """Draw classfication result to inference input image.

    Args:
        image (np.ndarray): A inference input RGB image to be draw.
        post_processed (np.ndarray): A one batch output of model be already applied post process.
            format is defined at https://github.com/blue-oil/blueoil/blob/master/lmnet/docs/specification/output_data.md
        config (EasyDict): Inference config.

    Returns:
        PIL.Image.Image: drawn image object.

    """
    colormap = get_color_map(len(config.CLASSES))
    font_size = 16

    max_class_id = int(np.argmax(post_processed))
    image_width = image.shape[1]

    image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype(FONT, font_size)

    text_color = colormap[max_class_id]

    class_name = config.CLASSES[max_class_id]
    score = float(np.max(post_processed))
    text = "{}\n{:.3f}".format(class_name, score)

    text_size = draw.multiline_textsize(text, font=font)
    draw.multiline_text((image_width - text_size[0] - 10, 0), text, fill=text_color, font=font, align="right")

    return image


def visualize_object_detection(image, post_processed, config):
    """Draw object detection result boxes to image.

    Args:
        image (np.ndarray): A inference input RGB image to be draw.
        post_processed (np.ndarray): A one batch output of model be already applied post process.
            format is defined at https://github.com/blue-oil/blueoil/blob/master/lmnet/docs/specification/output_data.md
        config (EasyDict): Inference config.

    Returns:
        PIL.Image.Image: drawn image object.

    """
    colormap = get_color_map(len(config.CLASSES))

    height_scale = image.shape[0] / float(config.IMAGE_SIZE[0])
    width_scale = image.shape[1] / float(config.IMAGE_SIZE[1])

    predict_boxes = np.copy(post_processed)
    predict_boxes[:, 0] *= width_scale
    predict_boxes[:, 1] *= height_scale
    predict_boxes[:, 2] *= width_scale
    predict_boxes[:, 3] *= height_scale

    image = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(image)

    for predict_box in predict_boxes:
        class_id = int(predict_box[4])
        class_name = config.CLASSES[class_id]
        box = [x for x in predict_box[:4]]
        score = predict_box[5]
        color = tuple(colormap[class_id])
        xy = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        draw.rectangle(xy, outline=color)
        top_left = [box[0] - 10, box[1]]
        txt = "{:s}: {:.3f}".format(class_name, float(score))
        draw.text(top_left, txt, fill=color)

    return image


def visualize_semantic_segmentation(image, post_processed, config):
    """Draw semantic segmentation result mask to image.

    Args:
        image (np.ndarray): A inference input RGB image to be draw.
        post_processed (np.ndarray): A one batch output of model be already applied post process.
            format is defined at https://github.com/blue-oil/blueoil/blob/master/lmnet/docs/specification/output_data.md
        config (EasyDict): Inference config.

    Returns:
        PIL.Image.Image: drawn image object.

    """
    colormap = np.array(get_color_map(len(config.CLASSES)), dtype=np.uint8)

    alpha = 0.5
    image_height = image.shape[0]
    image_width = image.shape[1]
    mask_image = label_to_color_image(np.expand_dims(post_processed, 0), colormap)
    mask_img = PIL.Image.fromarray(mask_image)
    mask_img = mask_img.resize(size=(image_width, image_height))
    result = PIL.Image.blend(PIL.Image.fromarray(image), mask_img, alpha)

    return result


def visualize_keypoint_detection(image, joints, original_image_size=None):
    """Draw keypoint detection result joints to image.

    Args:
        image: a numpy array of shape (height, width, 3).
        joints: a numpy array of shape (num_joints, 3).
        original_image_size: a tuple, (original_height, original_width). If not None, joints will be scaled.

    Returns:
        drawed_image: a numpy array of shape (height, width, 3).

    """

    if original_image_size is not None:
        height, width, _ = image.shape
        scale_height = height / original_image_size[0]
        scale_width = width / original_image_size[1]
        joints[:, 0] *= scale_width
        joints[:, 1] *= scale_height

    image = PIL.Image.fromarray(image, mode="RGB")
    draw = PIL.ImageDraw.Draw(image)

    num_joints = joints.shape[0]

    # These joints are paired to draw a skeleton.
    # A line will be drew only if both of joints are visible.
    # For details:
    # {0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
    #  5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow", 9: "left_wrist", 10: "right_wrist",
    #  11: "left_hip", 12: "right_hip", 13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"}
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

    for pair in joint_pairs:
        if joints[pair[0], 2] > 0 and joints[pair[1], 2] > 0:
            joint0 = (joints[pair[0], 0], joints[pair[0], 1])
            joint1 = (joints[pair[1], 0], joints[pair[1], 1])
            draw.line([joint0, joint1], fill=(255, 191, 0), width=3)

    for i in range(num_joints):
        if joints[i, 2] > 0:
            center_x, center_y = joints[i, :2]
            draw.ellipse([center_x - 2, center_y - 2,
                          center_x + 2, center_y + 2], fill=(238, 130, 238))

    drawed_image = np.array(image, dtype=np.uint8)

    return drawed_image


def label_to_color_image(results, colormap):
    """Adds color defined by the colormap to the label.

    Args:
        results: A 2D array with float type, storing the segmentation label.
        colormap: An ndarray with integer type. The number of classes with
        respective colour label.

    Returns:
        A 2D array with integer type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the CamVid color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.

    """
    if results.ndim != 4:
        raise ValueError('Expect 4-D input results (1, height, width, classes).')

    label = np.argmax(results, axis=3)
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return np.squeeze(colormap[label])


def draw_fps(pil_image, fps, fps_only_network):
    """Draw FPS information to image object.

    Args:
        pil_image (PIL.Image.Image): Image object to be draw FPS.
        fps (float): Entire inference FPS .
        fps_only_network (float): FPS of network only (not pre/post process).

    Returns:

    """
    font_size = 20
    font_size_sub = 14
    text_color = (200, 200, 200)
    text = "FPS: {:.1f}".format(fps)
    text_sub = "FPS (Network only): {:.1f}".format(fps_only_network)

    draw = PIL.ImageDraw.Draw(pil_image)
    font = PIL.ImageFont.truetype(FONT, font_size)
    font_sub = PIL.ImageFont.truetype(FONT, font_size_sub)
    draw.text((10, pil_image.height - font_size - font_size_sub - 5), text, fill=text_color, font=font)
    draw.text((10, pil_image.height - font_size_sub - 5), text_sub, fill=text_color, font=font_sub)
