import numpy as np


def compute_oks_batch(joints_gt, joints_pred, image_size=(160, 160)):
    """
    Compute a object keypoint similarity for a batch of examples.
    Args:
        joints_gt: a numpy array of shape (batch_size, num_joints, 3).
        joints_pred: a numpy array of shape (batch_size, num_joints, 3).
        image_size: a tuple, (height, width).

    Returns:
        oks_batch: float.

    """

    num_batch = joints_gt.shape[0]

    oks_batch = 0
    count = 0

    for i in range(num_batch):
        oks = compute_oks(joints_gt[i], joints_pred[i], image_size=image_size)
        if oks != -1:
            oks_batch += oks
            count += 1

    assert count != 0
    oks_batch /= count

    return np.float32(oks_batch)


def compute_oks(joints_gt, joints_pred, image_size=(160, 160)):
    """
    Compute a object keypoint similarity for one example.
    Args:
        joints_gt: a numpy array of shape (num_joints, 3).
        joints_pred: a numpy array of shape (num_joints, 3).
        image_size: a tuple, (height, width).

    Returns:
        oks: float.

    """

    num_joints = joints_gt.shape[0]

    x_gt = joints_gt[:, 0]
    y_gt = joints_gt[:, 1]
    v_gt = joints_gt[:, 2]

    x_pred = joints_pred[:, 0]
    y_pred = joints_pred[:, 1]

    area = image_size[0] * image_size[1]

    squared_distance = (x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2

    squared_distance /= area

    oks = 0
    count = 0

    for i in range(num_joints):
        if v_gt[i] > 0:
            oks += np.exp(-squared_distance[i], dtype=np.float32)
            count += 1

    if count == 0:
        return -1
    else:
        oks /= count
        return oks
