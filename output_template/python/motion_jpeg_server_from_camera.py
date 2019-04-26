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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from multiprocessing import Pool
import os
import signal
import sys
from SocketServer import ThreadingMixIn

import click

from lmnet.nnlib import NNLib
from lmnet.utils.config import (
    load_yaml,
    build_pre_process,
    build_post_process,
)
from lmnet.utils.demo import (
    VideoStream,
    run_inference,
)
from lmnet.visualize import (
    draw_fps,
    visualize_classification,
    visualize_object_detection,
    visualize_semantic_segmentation,
)


# global variable for multi process or multi thread.
nn = None
pre_process = None
post_process = None
stream = None
config = None
pool = None

# camera settings.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 10
CAMERA_SOURCE = 0


class MotionJpegHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global config, stream, pool
        result = None
        fps = 0.0
        fps_only_network = 0.0

        camera_img = stream.read()
        pool_result = pool.apply_async(_run_inference, (camera_img, ))

        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=jpgboundary')
        self.end_headers()

        while True:
            try:
                pool_result.wait()

                if pool_result.ready():
                    window_img = camera_img
                    result, fps, fps_only_network = pool_result.get()

                    camera_img = stream.read()
                    pool_result = pool.apply_async(_run_inference, (camera_img, ))

                else:
                    result = None

                if result is not None:
                    result = result[0]
                    if config.TASK == "IMAGE.CLASSIFICATION":
                        image = visualize_classification(window_img, result, config)

                    if config.TASK == "IMAGE.OBJECT_DETECTION":
                        image = visualize_object_detection(window_img, result, config)

                    if config.TASK == "IMAGE.SEMANTIC_SEGMENTATION":
                        image = visualize_semantic_segmentation(window_img, result, config)

                    draw_fps(image, fps, fps_only_network)
                    tmp = BytesIO()
                    image.save(tmp, "JPEG")

                self.send_header('Content-type', 'image/jpeg')
                self.end_headers()
                self.wfile.write(tmp.getvalue())

            finally:
                self.wfile.write(b"\r\n--jpgboundary\r\n")

        return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _run_inference(inputs):
    global nn, pre_process, post_process
    res, fps, fps_only_network = run_inference(inputs, nn, pre_process, post_process)
    return res, fps, fps_only_network


def _init_worker():
    global nn
    # ignore SIGINT in pooled process.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # init
    nn.init()


def run(model, config_file, port=80):
    global nn, pre_process, post_process, config, stream, pool

    filename, file_extension = os.path.splitext(model)
    supported_files = ['.so', '.pb']

    if file_extension not in supported_files:
        raise Exception("""
            Unknown file type. Got %s%s.
            Please check the model file (-m).
            Only .pb (protocol buffer) or .so (shared object) file is supported.
            """ % (filename, file_extension))

    if file_extension == '.so':  # Shared library
        nn = NNLib()
        nn.load(model)

    elif file_extension == '.pb':  # Protocol Buffer file
        # only load tensorflow if user wants to use GPU
        from lmnet.tensorflow_graph_runner import TensorflowGraphRunner
        nn = TensorflowGraphRunner(model)

    nn = NNLib()
    nn.load(model)

    stream = VideoStream(CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)

    config = load_yaml(config_file)

    pre_process = build_pre_process(config.PRE_PROCESSOR)
    post_process = build_post_process(config.POST_PROCESSOR)

    pool = Pool(processes=1, initializer=_init_worker)

    try:
        server = ThreadedHTTPServer(('', port), MotionJpegHandler)
        print("server starting")
        server.serve_forever()
    except KeyboardInterrupt as e:
        print("KeyboardInterrpt in server - ending server")
        stream.release()
        pool.terminate()
        pool.join()
        server.socket.close()
        server.shutdown()

    return


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-m",
    "-l",
    "--model",
    type=click.Path(exists=True),
    help=u"""
        Inference Model filename
        (-l is deprecated please use -m instead)
    """,
    default="../models/lib/lib_fpga.so",
)
@click.option(
    "-c",
    "--config_file",
    type=click.Path(exists=True),
    help=u"Config file Path",
    default="../models/meta.yaml",
)
@click.option(
    "-p",
    "--port",
    default=80,
    help="Port number of motion jpege server."
)
def main(model, config_file, port):
    """Serve motion jpeg server from video camera source.

    1. Run inference from video camera source image.
    2. Visualize (decorate) input image from inference result.
    3. Response decorated image as motion jpeg.
    """

    run(model, config_file, port)


def _check_deprecated_arguments():
    argument_list = sys.argv
    if '-l' in argument_list:
        print("Deprecated warning: -l is deprecated please use -m instead")


if __name__ == "__main__":
    main()
