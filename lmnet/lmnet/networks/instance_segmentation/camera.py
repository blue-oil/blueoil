import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys

ROOT_DIR = os.path.abspath("../../../")

sys.path.append(ROOT_DIR)
from lmnet.networks.instance_segmentation import visualize
import lmnet.networks.instance_segmentation.model_quantized as modellib

from lmnet.networks.instance_segmentation import balloon

MODEL_DIR = os.path.join(os.path.abspath('./notebooks'), "logs")
BALLON_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'balloon20200106T1307/mask_rcnn_balloon_0020.h5')

config = balloon.BalloonConfig()


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    _, ax = plt.subplots(rows, cols)
    return ax


def get_model():
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    weights_path = model.find_last()
    model.load_weights(weights_path, by_name=True)
    return model


def grab_frame(cap):
    ret, frame = cap.read()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def show_webcam():
    cam = cv2.VideoCapture(0)
    model = get_model()
    # ax = plt.subplot(1, 2, 1)
    ax = get_ax()
    height, width = 960, 1280
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title("Predictions")
    # ax.imshow(grab_frame(cam))
    plt.ion()
    while True:
        image = grab_frame(cam)
        results = model.detect([image], verbose=1)
        r = results[0]
        ax.clear()
        visualize.display_instances_for_camera(image, r['rois'], r['masks'], r['class_ids'],
                                               ["BG", "balloon"], r['scores'], ax=ax,
                                               title="Predictions")
        # img.set_data(
        # )
        plt.pause(0.001)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


show_webcam()
