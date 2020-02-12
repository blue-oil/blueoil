import numpy as np


def iou(boxes, box):
    """Calculate overlap

    Args:
        boxes: boxes in the image. shape is [num_boxes, 4 or more(x, y, w, h, ...)]
        box: a single box in the image. shape is [4 or more (x, y, w, h, ...)]
    Returns:
        iou: shape is [num_boxes]
    """

    if boxes.size == 0:
        raise ValueError("Cannot calculate if ground truth boxes is zero")

    # format boxes (left, top, right, bottom)
    boxes = np.stack([
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 0] + boxes[:, 2],
        boxes[:, 1] + boxes[:, 3],
    ], axis=1)

    # format box (left, top, right, bottom)
    box = np.array([
        box[0],
        box[1],
        box[0] + box[2],
        box[1] + box[3],
    ])

    # calculate the left up point
    left_top = np.maximum(boxes[:, 0:2], box[0:2])

    # calculate the right bottom point
    right_bottom = np.minimum(boxes[:, 2:], box[2:])

    horizon_vertical = np.maximum(right_bottom - left_top, 0)

    intersection = horizon_vertical[:, 0] * horizon_vertical[:, 1]

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area = (box[2] - box[0]) * (box[3] - box[1])

    epsilon = 1e-10
    union = area + areas - intersection

    return intersection / (union + epsilon)


def fill_dummy_boxes(gt_boxes, num_max_boxes):

    dummy_gt_box = [0, 0, 0, 0, -1]

    if len(gt_boxes) == 0:
        gt_boxes = np.array(dummy_gt_box * num_max_boxes)
        return gt_boxes.reshape([num_max_boxes, 5])

    if len(gt_boxes) < num_max_boxes:
        diff = num_max_boxes - len(gt_boxes)
        gt_boxes = np.append(gt_boxes, [dummy_gt_box] * diff, axis=0)
        return gt_boxes

    return gt_boxes


def crop_boxes(boxes, crop_rect):
    """Crop boxes

    Args:
        boxes: ground truth boxes in the image. shape is [num_boxes, 5(x, y, w, h, class)]
        crop_rect: a crop rectangle. shape is [4(x, y, w, h)]
    Returns:
        cropped_boxes: shape is [num_boxes, 5(x, y, w, h, class)]
    """
    # check crop_rect overlap with boxes.
    if ((crop_rect[0] + crop_rect[2]) < boxes[:, 0]).any():
        raise ValueError("Crop_rect does not overlap with some boxes."
                         "Increasing x or w of crop_rect may be helpful.")
    if ((crop_rect[1] + crop_rect[3]) < boxes[:, 1]).any():
        raise ValueError("Crop_rect does not overlap with some boxes."
                         "Increasing y or h of crop_rect may be helpful.")
    if (crop_rect[0] > (boxes[:, 0] + boxes[:, 2])).any():
        raise ValueError("Crop_rect does not overlap with some boxes."
                         "Decreasing x of crop_rect may be helpful.")
    if (crop_rect[1] > (boxes[:, 1] + boxes[:, 3])).any():
        raise ValueError("Crop_rect does not overlap with some boxes."
                         "Decreasing y of crop_rect may be helpful.")

    # format to xmin, ymin, xmax, ymax
    cropped_boxes = np.stack([
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 0] + boxes[:, 2],
        boxes[:, 1] + boxes[:, 3],
    ], axis=1)

    # shift to crop size.
    cropped_boxes = np.stack([
        cropped_boxes[:, 0] - crop_rect[0],
        cropped_boxes[:, 1] - crop_rect[1],
        cropped_boxes[:, 2] - crop_rect[0],
        cropped_boxes[:, 3] - crop_rect[1],
    ], axis=1)

    # adjust beyond box
    cropped_boxes = np.stack([
        np.maximum(cropped_boxes[:, 0], 0),
        np.maximum(cropped_boxes[:, 1], 0),
        np.minimum(cropped_boxes[:, 2], crop_rect[2]),
        np.minimum(cropped_boxes[:, 3], crop_rect[3]),
    ], axis=1)

    # format to x, y, w, h
    cropped_boxes = np.stack([
        cropped_boxes[:, 0],
        cropped_boxes[:, 1],
        cropped_boxes[:, 2] - cropped_boxes[:, 0],
        cropped_boxes[:, 3] - cropped_boxes[:, 1],
        boxes[:, 4],
    ], axis=1)

    return cropped_boxes


def format_cxcywh_to_xywh(boxes, axis=1):
    """Format form (center_x, center_y, w, h) to (x, y, w, h) along specific dimension.

    Args:
    boxes: A tensor include boxes. [:, 4(x, y, w, h)]
    axis: Which dimension of the inputs Tensor is boxes.
    """
    results = np.split(boxes, [1, 2, 3, 4], axis=axis)
    center_x, center_y, w, h = results[0], results[1], results[2], results[3]
    x = center_x - (w / 2)
    y = center_y - (h / 2)

    return np.concatenate([x, y, w, h], axis=axis)
