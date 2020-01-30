import tensorflow as tf
from lmnet.data_processor import Processor

def tf_resize_with_gt_boxes(image, gt_boxes, size=(256,256)):
    """Resize an image and gt_boxes.

    Args:
        image (np.ndarray): An image numpy array.
        gt_boxes (np.ndarray): Ground truth boxes in the image. shape is [num_boxes, 5(x, y, width, height)].
        size: [height, width]

    """
    orig_width, orig_height = image.get_shape().as_list()
    width, height = size

    image = tf.image.resize(image, size)
    if gt_boxes is None:
        return image, None

    scale = [height / orig_height, width / orig_width]
    if len(gt_boxes) > 0:
        gt_boxes = gt_boxes[:, 0].assign(gt_boxes[:, 0] * scale[1])
        gt_boxes = gt_boxes[:, 1].assign(gt_boxes[:, 1] * scale[0])
        gt_boxes = gt_boxes[:, 2].assign(gt_boxes[:, 2] * scale[1])
        gt_boxes = gt_boxes[:, 3].assign(gt_boxes[:, 3] * scale[0])

        gt_boxes = gt_boxes[:, 0].assign(tf.minimum(
                gt_boxes[:, 0], width - gt_boxes[:, 2]))
        gt_boxes = gt_boxes[:, 1].assign(tf.minimum(
                gt_boxes[:, 1], width - gt_boxes[:, 3]))

class TFResize(Processor):
    """Resize an image
    """

    def __init__(self, size=(256,256)):
        """
        Args:
            size: (height, width)
        """
        self.size = size

    def __call__(self, image, **kwargs):
        """
        Args:
            image (tf.Tensor): an image tensor sized (orig_height, orig_width, channel)
        """
        return tf.image.resize(image, self.size)

class TFPerImageStandardization(Processor):
    """Standardization per image.
    """

    def __call__(self, image, **kwargs):
        return tf.image.per_image_standardization(image)

class TFResizeWithGtBoxes(Processor):
    """Resize image with gt boxes.

    Use :func:`~resize_with_gt_boxes` inside.

    Args:
        size: Target size.

    """

    def __init__(self, size):
        self.size = size

    def __call__(self, image, gt_boxes=None, **kwargs):
        image, gt_boxes = tf_resize_with_gt_boxes(image, gt_boxes, self.size)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)

class TFDivideBy255(Processor):
    """Divide image by 255
    """

    def __call__(self, image, **kwargs):
        image = image / 255.0
        return dict({'image': image}, **kwargs)