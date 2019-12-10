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

from lmnet.networks.base import BaseNetwork
import numpy as np
import tensorflow as tf


class MaskRCNN(BaseNetwork):
    """Mask R-CNN

    paper: https://arxiv.org/abs/1703.06870
    """

    def __init__(
            self,
            backbone='resnet18',
            down_sample_scale=32,
            compute_backbone_shape=None,
            backbone_strides=[4, 8, 16, 32, 64],
            fpn_classif_fc_layers_size=1024,
            top_down_pyramid_size=256,
            num_classes=1,
            rpn_anchor_scales=(32, 64, 128, 256, 512),
            rpn_anchor_ratios=[0.5, 1, 2],
            rpn_anchor_stride=1,
            rpn_nms_threshold=0.7,
            rpn_train_anchors_per_image=256,
            pre_nms_limit=6000,
            post_nms_rois_training=2000,
            post_nms_rois_inference=1000,
            image_resize_mode="square",
            # image_min_dim = 800,
            # image_max_dim = 1024,
            # image_min_scale = 0,
            image_channel_count=3,
            # mean_pixel = np.array([123.7, 116.8, 103.9]),
            train_rois_per_image=200,
            roi_positive_ratio=0.33,
            pool_size=7,
            mask_pool_size=14,
            mask_shape=[28, 28],
            max_gt_instances=100,
            rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
            bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
            detection_max_instances=100,
            detection_min_confidence=0.7,
            detection_nms_threshold=0.3,
            learning_rate=0.001,
            learning_momentum=0.9,
            weight_decay=0.0001,
            loss_weights={
                "rpn_class_loss": 1.,
                "rpn_bbox_loss": 1.,
                "mrcnn_class_loss": 1.,
                "mrcnn_bbox_loss": 1.,
                "mrcnn_mask_loss": 1.
            },
            use_rpn_rois=True,
            train_bn=False,
            gradient_clip_norm=5.0,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = backbone
        self.down_sample_scale = down_sample_scale
        self.compute_backbone_shape = compute_backbone_shape
        self.backbone_strides = backbone_strides
        self.fpn_classif_fc_layers_size = fpn_classif_fc_layers_size
        self.top_down_pyramid_size = top_down_pyramid_size
        self.num_classes = num_classes
        self.rpn_anchor_scales = rpn_anchor_scales
        self.rpn_anchor_ratios = rpn_anchor_ratios
        self.rpn_anchor_stride = rpn_anchor_stride
        self.rpn_nms_threshold = rpn_nms_threshold
        self.rpn_train_anchors_per_image = rpn_train_anchors_per_image
        self.pre_nms_limit = pre_nms_limit
        self.post_nms_rois_training = post_nms_rois_training
        self.post_nms_rois_inference = post_nms_rois_inference
        self.image_resize_mode = image_resize_mode
        self.image_channel_count = image_channel_count
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.pool_size = pool_size
        self.mask_pool_size = mask_pool_size
        self.mask_shape = mask_shape
        self.max_gt_instances = max_gt_instances
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.bbox_std_dev = bbox_std_dev
        self.detection_max_instances = detection_max_instances
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold
        self.learning_rate = learning_rate
        self.learning_momentum = learning_momentum
        self.weight_decay = weight_decay
        self.loss_weights = loss_weights
        self.use_rpn_rois = use_rpn_rois
        self.train_bn = train_bn
        self.gradient_clip_norm = gradient_clip_norm

        assert self.image_size[0] % down_sample_scale == 0
        assert self.image_size[1] % down_sample_scale == 0

    def placeholders(self):
        """placeholders"""

        images_placeholder = labels_placeholder = None

        return images_placeholder, labels_placeholder

    def summary(self, output, labels=None):
        super().summary(output, labels)

    def metrics(self, output, labels, thresholds=[0.3, 0.5, 0.7]):
        # return metrics_ops_dict, tf.group(*updates)
        return None

    def post_process(self, output):
        return None

    def loss(self, output, gt_boxes, global_step):
        return None

    def inference(self, images, is_training):
        return None

    def _resnet_graph(self, input_image):
        pass

    def base(self, images, is_training):
        self.images=images

        # backbone
        # C2,C3,C4,C5=

        output = images

        return output
