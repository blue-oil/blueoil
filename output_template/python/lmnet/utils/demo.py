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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from itertools import product as itr_prod

import cv2
import numpy as np


COLORS = [tuple(p) for p in itr_prod([0, 180, 255], repeat=3)]
COLORS = COLORS[1:]


def ltwh_to__tblr(ltwh):
    l, t, w, h = ltwh.tolist()
    b = int(t + h)
    r = int(l + w)
    return t, b, l, r


def add_fps(orig, fps):
    f_p_s_text = "FPS: {:.1f}".format(fps)
    text_color = (255, 144, 30)
    orig_h, orig_w = orig.shape[:2]
    cv2.putText(orig, f_p_s_text, (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    return orig


def check_range(upper, lower, checked_val):
    if upper < checked_val:
        checked_val = upper
    elif lower > checked_val:
        checked_val = lower
    return checked_val


def add_rectangle(classes, orig, preds, pred_shape):
    orig_h, orig_w = orig.shape[:2]
    locs = [pred[:, 0:4] for pred in preds]
    labels_n = np.array([pred[:, 4] for pred in preds]).astype(np.int)  # TODO magic-number
    labels_n = labels_n.flatten()

    labels = [classes[i_label] for i_label in labels_n]
    scores = preds[0][:, 5]

    pred_h, pred_w = pred_shape
    w_scale = orig_w / pred_w
    h_scale = orig_h / pred_h
    locs = (np.array(locs).reshape((-1, 4)) * [w_scale, h_scale, w_scale, h_scale]).astype(int)
    for idx, loc in enumerate(locs):
        t, b, le, r = ltwh_to__tblr(loc)

        le = check_range(orig_w, 0, le)
        r = check_range(orig_w, 0, r)
        t = check_range(orig_h, 0, t)
        b = check_range(orig_h, 0, b)

        color_r = COLORS[labels_n[idx] % len(COLORS)]
        thick = 2
        label_text = "{} : {:.1f}%".format(labels[idx], scores[idx] * 100)
        label_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(orig, (le, t), (r, b), color_r, thick)

        max_color = max(color_r)
        text_color = (255, 255, 255) if max_color < 255 else (0, 0, 0)

        cv2_filed_config = cv2.cv.CV_FILLED if hasattr(cv2, 'cv') else cv2.FILLED

        cv2.rectangle(orig, (le, t), (le + label_size[0], t + label_size[1]), color_r, cv2_filed_config)
        cv2.putText(orig, label_text, (le, t + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    return orig
