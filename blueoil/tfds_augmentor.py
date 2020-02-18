import tensorflow as tf
from blueoil.data_processor import Processor


class TFPad(Processor):
    """Add padding to images.

    Args:
        value (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill (int): Pixel fill value. Default is 0.
    """

    def __init__(self, value, fill=0):
        if type(value) is int:
            left = top = right = bottom = value

        elif type(value) is tuple:
            if len(value) == 2:
                left, top = right, bottom = value

            if len(value) == 4:
                left, top, right, bottom = value
        else:
            raise Exception("Expected int, tuple/list with 2 or 4 entries. Got %s." % (type(value)))
        self.paddings = [[top, bottom], [left, right], [0, 0]]
        self.fill = fill

    def __call__(self, image, **kwargs):
        image = tf.pad(image, tf.constant(self.paddings), constant_values=self.fill)
        return dict({'image': image}, **kwargs)


class TFCrop(Processor):
    """Randomly crop an image.

    Args:
        size (tuple | list): the size to crop. i.e. [crop_height, crop_width]
        seed  (int)        : seed of a random factor

    """

    def __init__(self, size, seed=0):
        height, width = size
        # channel size is 3
        self.size = [height, width, 3]
        self.seed = seed

    def __call__(self, image, **kwargs):
        image = tf.image.random_crop(image, self.size, seed=self.seed)
        return dict({'image': image}, **kwargs)


def _random_flip_left_right_bounding_box(image, gt_boxes, seed):
    """Flip left right only bounding box.

    Args:
        image    (tf.Tensor): image
        gt_boxes (tf.Tensor): bounding boxes. shape is [num_boxes, 5(x, y, w, h, class_id)]
        seed     (int)   : seed of a random factor
    """
    width = image.get_shape().as_list()[1]
    rand = tf.random.uniform([], minval=0, maxval=1, seed=seed)
    cond = tf.less(rand, .5)
    image = tf.cond(
        cond,
        lambda: tf.image.flip_left_right(image),
        lambda: image
    )
    gt_boxes = tf.cond(
        cond,
        lambda: tf.concat([tf.expand_dims(width - gt_boxes[:, 0] - gt_boxes[:, 2], 1), gt_boxes[:, 1:]], 1),
        lambda: gt_boxes)
    return image, gt_boxes


class TFFlipLeftRight(Processor):
    """Flip left right with a probability 0.5.

    Args:
        seed (int): seed of a random factor
    """

    def __init__(self, seed=0):
        self.seed = 0

    def __call__(self, image, gt_boxes=None, **kwargs):
        if gt_boxes is None:
            image = tf.image.random_flip_left_right(image, seed=self.seed)
        else:
            image, gt_boxes = _random_flip_left_right_bounding_box(image, gt_boxes, seed=self.seed)
        return dict({'image': image, 'gt_boxes': gt_boxes}, **kwargs)


class TFBrightness(Processor):
    """Adjust the brightness of images by a random factor.
       (picked from uniform distribution [-delta, delta) )

    Args:
        delta (float): max delta for distribution. must be positive
        seed  (int)  : seed of a random factor
    """

    def __init__(self, delta=0.25, seed=0):
        self.delta = delta
        self.seed = seed

    def __call__(self, image, **kwargs):
        image = tf.image.random_brightness(image, self.delta, seed=self.seed)
        return dict({'image': image}, **kwargs)


class TFHue(Processor):
    """Randomly change image hue.

    Args:
       delta (float): max delta for distribution. must be in [0, 0.5]
       seed  (int)  : seed of a random factor
    """

    def __init__(self, delta=10.0/255, seed=0):
        self.delta = delta
        self.seed = seed

    def __call__(self, image, **kwargs):
        image = tf.image.random_hue(image, self.delta, seed=self.seed)
        return dict({'image': image}, **kwargs)


class TFSaturation(Processor):
    """Randomly adjust the saturation of a RGB image.
       Random factor is picked from [lower, upper]
       0 <= lower < upper must be satisfied.

    Args:
       value (float|tuple|list):
          Range for random factor is taken as [1 - value, 1 + value] if value is float value
          and [1 - value[0], 1 + value[1]] if tuple or list with 2 elements
       seed  (int)  : seed of a random factor
    """

    def __init__(self, value=(0.75, 1.25), seed=0):
        if value is float:
            self.lower = 1 - value
            self.upper = 1 + value
        elif len(value) == 2:
            self.lower, self.upper = value
        else:
            raise Exception("Expected float, tuple/list with 2 entries. Got %s." % (type(value)))
        self.seed = seed

    def __call__(self, image, **kwargs):
        image = tf.image.random_saturation(image, self.lower, self.upper, seed=self.seed)
        return dict({'image': image}, **kwargs)


class TFContrast(Processor):
    """Randomly adjust the contrast of an image.
       Random factor is picked from [lower, upper]
       0 <= lower < upper must be satisfied.

    Args:
       value (float|tuple|list):
          Range for random factor is taken as [1 - value, 1 + value] if value is float value
          and [1 - value[0], 1 + value[1]] if tuple or list with 2 elements
       seed  (int)  : seed of a random factor
    """

    def __init__(self, value=(0.75, 1.25), seed=0):
        if value is float:
            self.lower = 1 - value
            self.upper = 1 + value
        elif len(value) == 2:
            self.lower, self.upper = value
        else:
            raise Exception("Expected float, tuple/list with 2 entries. Got %s." % (type(value)))
        self.seed = seed

    def __call__(self, image, **kwargs):
        image = tf.image.random_contrast(image, self.lower, self.upper, seed=self.seed)
        return dict({'image': image}, **kwargs)
