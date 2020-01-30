import tensorflow as tf
from lmnet.data_processor import Processor

def _random_flip_left_right_bounding_box(image, gt_boxes, seed):
    """Flip left right only bounding box.

    Args:
        width    (tensor): width of an image
        gt_boxes (tensor): bounding boxes. shape is [num_boxes, 5(x, y, w, h, class_id)]
    """
    width = image.get_shape().as_list()[1]
    rand = tf.random.uniform([], minval=0, maxval=1, seed=seed)
    cond = tf.less(rand, .5)
    image = tf.cond(
        cond,
        lambda: tf.image.random_flip_left_right(image),
        lambda: image
    )
    gt_boxes = tf.cond(
        cond,
        lambda: gt_boxes[:, 0].assign(width - gt_boxes[:, 0] - gt_boxes[:, 2]),
        lambda: gt_boxes)
    return image, gt_boxes



class TFFlipLeftRight(Processor):
    """Flip left right with a probability 0.5.

    Args:
        seed (number): Seed for flipping.
    """

    def __init__(self, seed=0):
        self.seed = 0

    def __call__(self, image, gt_boxes=None, **kwargs):
        if gt_boxes is None:
            image = tf.image.random_flip_left_right(image, seed=self.seed)
        else:
            image, gt_boxes = tf.image.random_flip_left_right_bounding_box(image, gt_boxes, seed=self.seed)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)
