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

import time
from multiprocessing import Pool
# from multiprocessing.dummy import Pool
from time import sleep
from Queue import Queue

import click
import cv2
import numpy as np

from lmnet.nnlib import NNLib
from lmnet.utils.config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)

from lmnet.utils.demo import (
    add_rectangle,
    add_fps,
)

nn = None
pre_process = None
post_process = None


class MyTime:
    def __init__(self, function_name):
        self.start_time = time.time()
        self.function_name = function_name

    def show(self):
        print("TIME: ", self.function_name, time.time() - self.start_time)


def add_class_label(canvas,
                    text="Hello",
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.42,
                    font_color=(140, 40, 200),
                    line_type=1,
                    dl_corner=(50, 50)):
    cv2.putText(canvas, text, dl_corner, font, font_scale, font_color, line_type)


def create_label_colormap():
    colormap = np.array([
        [64, 64, 64],
        [128,   0,   0],
        [192, 192, 128],
        [128,  64, 128],
        [190, 153, 153],
        [128, 128,   0],
        [192, 128, 128],
        [ 64,  64, 128],
        [ 64,   0, 128],
        [ 64,  64,   0],
        [  0, 128, 192]], dtype=np.uint8)
    return colormap


def label_to_color_image(results):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if results.ndim != 4:
        raise ValueError('Expect 4-D input results (1, height, width, classes).')

    colormap = create_label_colormap()

    label = np.argmax(results, axis=3)
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    
    return np.squeeze(colormap[label])


import matplotlib.pyplot as plt

_color_map = plt.cm.jet
_color_map._init()


def label_color_map(results):

    person_label = np.round(results[:, :, :, 1] * 255)
    person_label = person_label.astype(np.uint8)
    lut = _color_map._lut[:, :3]
    lut = np.flip(lut, axis=1)
    tmp = np.take(lut, person_label, axis=0)
    tmp = np.round(tmp * 255).astype(np.uint8)
    tmp[results[:, :, :, 0] > 0.5] = 0
    return np.squeeze(tmp)


def run_inference(inputs):
    img = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB)
    global nn, pre_process, post_process
    start = time.time()

    data = pre_process(image=img)["image"]
    data = np.asarray(data, dtype=np.float32)
    data = np.expand_dims(data, axis=0)

    result = nn.run(data)

    # if post_process is not None:
    result = post_process(outputs=result)['outputs']

    fps = 1.0/(time.time() - start)
    return result, fps


def clear_queue(queue):
    while not queue.empty():
        queue.get()
    return queue


def swap_queue(q1, q2):
    return q2, q1  # These results are swapped


import signal
def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

import PIL.Image


def _seg_post_process(result, input_width, input_height):
    result = np.squeeze(result)
    results = np.split(result, 2, axis=2)
    output = []
    for result in results:
        result = np.squeeze(result)
        result = PIL.Image.fromarray(result, mode="F")
        result = result.resize([input_width, input_height], PIL.Image.BILINEAR)
        result = np.array(result)
        result = np.expand_dims(result, axis=2)
        output.append(result)
    result = np.concatenate(output, axis=2)
    result = np.expand_dims(result, axis=0)
    result = softmax(result)
    threshold = 0.7

    # print("sum", np.sum(result < threshold))
    result[result < threshold] = 0.
    return result


from threading import Thread


class FileVideoStream:
    def __init__(self, stream, queue_size=128):
        self.stream = stream
        self.stopped = False

        self.Q = Queue(maxsize=queue_size)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stopped:
                break

            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    continue
                self.Q.put(frame)
            else:
                time.sleep(1)

        self.stream.release()

    def read(self):
        return self.Q.get()

    def running(self):
        return self.more() or not self.stopped

    def more(self):
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(1)
            tries += 1
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        self.thread.join()


def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.expand_dims(exp.sum(axis=-1), -1)


from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global config, vc, pool, camera_width, camera_height

        print("get" * 100)

        input_width = config.IMAGE_SIZE[1]
        input_height = config.IMAGE_SIZE[0]

        stream = vc
        result = None
        fps = 1.0

        _, camera_img = stream.read()
        pool_result = pool.apply_async(run_inference, (camera_img, ))

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
        self.end_headers()
        while True:
            try:
                # if not flg:
                #     continue

                pool_result.wait()

                if pool_result.ready():
                    window_img = camera_img
                    result, fps = pool_result.get()

                    flg, camera_img = stream.read()
                    pool_result = pool.apply_async(run_inference, (camera_img, ))

                else:
                    result = None

                if result is not None:
                    post_processed = _seg_post_process(result, input_height, input_width)

                    # seg_img = label_to_color_image(result)
                    seg_img = label_color_map(post_processed)
                    seg_img = cv2.resize(seg_img, (camera_width, camera_height))

                    window_img = cv2.addWeighted(window_img, 1, seg_img, 0.8, 0)
                    window_img = add_fps(window_img, fps)

                img_str = cv2.imencode('.jpg', window_img)[1].tostring()
                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(img_str)
                self.wfile.write(b"\r\n--jpgboundary\r\n")

            except KeyboardInterrupt:
                # end of the message - not sure how we ever get here, though
                print("KeyboardInterrpt in server loop - breaking the loop (server now hung?)")
                self.wfile.write(b"\r\n--jpgboundary--\r\n")
                pool.close()
                pool.join()
                break
        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""


def run(library, config_file):
    global nn, pre_process, post_process
    global config, vc, pool, camera_width, camera_height
    nn = NNLib()
    nn.load(library)
    nn.init()

    config = load_yaml(config_file)

    pre_process = build_pre_process(config.PRE_PROCESSOR)
    post_process = build_post_process(config.POST_PROCESSOR)

    camera_width = 320
    camera_height = 240
    vc = cv2.VideoCapture(0)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, camera_width)
    vc.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, camera_height)
    vc.set(cv2.cv.CV_CAP_PROP_FPS, 1)

    pool = Pool(processes=1)

    if config.TASK == "IMAGE.CLASSIFICATION":
        run_classification(config)

    if config.TASK == "IMAGE.OBJECT_DETECTION":
        run_object_detection(config)

    if config.TASK == "IMAGE.SEMANTIC_SEGMENTATION":
        try:
            server = ThreadedHTTPServer(('', 80), CamHandler)
            print("server starting")
            server.serve_forever()
        except KeyboardInterrupt:
            # ctrl-c comes here but need another to end all.  Probably should have terminated thread here, too.
            print("KeyboardInterrpt in server - ending server")
            # capture.release()
            pool.close()
            pool.join()
            server.socket.close()


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-l",
    "--library",
    type=click.Path(exists=True),
    help=u"Shared library filename",
    default="../models/lib/lib_fpga.so",
)
@click.option(
    "-c",
    "--config_file",
    type=click.Path(exists=True),
    help=u"Config file Path",
    default="../models/meta.yaml",
)
def main(library, config_file):
    run(library, config_file)


if __name__ == "__main__":
    main()
