/* Copyright 2019 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================*/

#include <cstdlib>
#include <iostream>
#include <utility>

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#ifdef USE_OPENCV
#include "blueoil_opencv.hpp"
#endif
#include "blueoil_data_processor.hpp"
#include "test_util.hpp"

float test_input[3][8][8] =
  { {  // Red
     {255, 0, 0, 0, 0, 0, 0, 0},
     {0, 255, 0, 0, 0, 0, 0, 0},
     {0, 0, 100, 0, 0, 0, 0, 0},
     {0, 0, 0, 100, 0, 0, 0, 0},
     {0, 0, 0, 0, 100, 0, 0, 0},
     {0, 0, 0, 0, 0, 100, 0, 0},
     {0, 0, 0, 0, 0, 0, 255, 0},
     {0, 0, 0, 0, 0, 0, 0, 255}
     },
    {  // Green
     {0, 0, 0, 0, 0, 0, 0, 255},
     {0, 0, 0, 0, 0, 0, 255, 0},
     {0, 0, 0, 0, 0, 100, 0, 0},
     {0, 0, 0, 0, 100, 0, 0, 0},
     {0, 0, 0, 100, 0, 0, 0, 0},
     {0, 0, 100, 0, 0, 0, 0, 0},
     {0, 255, 0, 0, 0, 0, 0, 0},
     {255, 0, 0, 0, 0, 0, 0, 0}
    },
    {  // Blue
     {  0,   0,   0, 255, 255,   0,   0,   0},
     {  0,   0,   0, 255, 255,   0,   0,   0},
     {  0,   0,   0, 255, 255,   0,   0,   0},
     {255, 255, 255, 100, 100, 255, 255, 255},
     {255, 255, 255, 100, 100, 255, 255, 255},
     {  0,   0,   0, 255, 255,   0,   0,   0},
     {  0,   0,   0, 255, 255,   0,   0,   0},
     {  0,   0,   0, 255, 255,   0,   0,   0}
    } };

// python pillow(PIL) NN-resize output array
float test_expect[3][4][4] =
  { {  // Red
     {255,   0,   0,   0},
     {  0, 100,   0,   0},
     {  0,   0, 100,   0},
     {  0,   0,   0, 255}
     },
    {  // Green
     {  0,  0,  0,  0},
     {  0,  0,  0,  0},
     {  0,  0,  0,  0},
     {  0,  0,  0,  0}
    },
    {  // Blue
     {  0, 255,   0,   0},
     {255, 100, 255, 255},
     {  0, 255,   0,   0},
     {  0, 255,   0,   0}
    } };

// from test_preprocessor.py
float data_processor_input[1][1][3] = {1, 1, 1};

float divideby255_expect[1][1][3] = {0.00392157, 0.00392157, 0.00392157};

float perimagestandardization_expect[1][1][3] = {0.0000, 0.0000, 0.0000};

// python lmnet post processor output array
// rounded to 4-digit after the decimal point
float yolov2test_expect[1][16][6]=
  { {
     {12.9296, 13.0093,  6.9977,  7.1238, 0.0, 0.2522},
     { 3.5483,  3.4398, 27.7531, 28.2531, 0.0, 0.2676},
     {45.9175, 13.9759,  8.9852,  9.1471, 0.0, 0.2829},
     {33.5592,  1.3744, 35.6357, 36.2777, 0.0, 0.2979},
     {14.5502, 46.5784, 11.5373, 11.7452, 0.0, 0.3126},
     {-1.6491, 30.0659, 45.7571, 46.5815, 0.0, 0.3268},
     {46.6965, 46.6847, 14.8142, 15.0811, 0.0, 0.3404},
     {25.5605, 25.1468, 58.7533, 59.8119, 0.0, 0.3534},
     {12.9296, 13.0093,  6.9977,  7.1238, 1.0, 0.2567},
     { 3.5483,  3.4398, 27.7531, 28.2531, 1.0, 0.2725},
     {45.9175, 13.9759,  8.9852,  9.1471, 1.0, 0.288 },
     {33.5592,  1.3744, 35.6357, 36.2777, 1.0, 0.3033},
     {14.5502, 46.5784, 11.5373, 11.7452, 1.0, 0.3182},
     {-1.6491, 30.0659, 45.7571, 46.5815, 1.0, 0.3327},
     {46.6965, 46.6847, 14.8142, 15.0811, 1.0, 0.3465},
     {25.5605, 25.1468, 58.7533, 59.8119, 1.0, 0.3598}
     } };

float excludelowscorebox_input[1][8][6] =
  { {
     { 0, 0, 0, 0, 0, 0.1},
     { 0, 0, 0, 1, 1, 0.8},
     { 0, 0, 1, 0, 2, 0.4},
     { 0, 0, 1, 1, 3, 0.9},
     { 0, 1, 0, 0, 4, 1.0},
     { 0, 1, 0, 1, 5, 0.3},
     { 0, 1, 1, 0, 6, 0.2},
     { 0, 1, 1, 1, 7, 0.9}
     } };

float excludelowscorebox_expect[1][4][6] =
  { {
     { 0, 0, 0, 1, 1, 0.8},
     { 0, 0, 1, 1, 3, 0.9},
     { 0, 1, 0, 0, 4, 1.0},
     { 0, 1, 1, 1, 7, 0.9}
     } };

float nms_input[1][8][6] =
  { {
     {0.1, 0.1, 0.6, 0.6, 0, 0.1},
     {0.2, 0.2, 0.6, 0.6, 0, 0.3},
     {0.4, 0.4, 0.6, 0.6, 0, 0.5},
     {0, 0, 1, 1, 1, 0.7},
     {0, 0, 1, 1, 1, 0.8},
     {0, 0, 1, 1, 1, 0.6},
     {0, 0, 1, 1, 2, 0.4},
     {0, 0, 1, 1, 3, 0.2}  // wrong class id
     } };

float nms_expect[1][4][6] =
  { {
     {0.4, 0.4, 0.6, 0.6, 0.,  0.5},
     {0.2, 0.2, 0.6, 0.6, 0.,  0.3},
     {0.,  0.,  1.,  1.,  1.,  0.8},
     {0.,  0.,  1.,  1.,  2.,  0.4}
     } };

int test_data_processor_resize() {
  // CHW (3-channel, height, width)
  int width = 4, height = 4;
  const std::pair<int, int>& image_size = std::make_pair(width, height);
  blueoil::Tensor input({3, 8, 8}, reinterpret_cast<float *>(test_input));
  blueoil::Tensor expect({3, 4, 4}, reinterpret_cast<float *>(test_expect));
  input = blueoil::util::Tensor_CHW_to_HWC(input);
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);
  blueoil::Tensor output = blueoil::data_processor::Resize(input,
                                                           image_size);
  if (!output.allclose(expect)) {
    std::cerr << "test_data_processor_resize: output != expect" << std::endl;
    output = blueoil::util::Tensor_HWC_to_CHW(output);
    expect = blueoil::util::Tensor_HWC_to_CHW(expect);
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int test_data_processor_divideby255() {
  blueoil::Tensor input({1, 1, 3}, reinterpret_cast<float *>(data_processor_input));
  blueoil::Tensor expect({1, 1, 3}, reinterpret_cast<float *>(divideby255_expect));
  input = blueoil::util::Tensor_CHW_to_HWC(input);
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);

  blueoil::Tensor output = blueoil::data_processor::DivideBy255(input);
  if (!output.allclose(expect, 0, 0.0001)) {
    std::cerr << "test_data_processor_divideby255: output != expect" << std::endl;
    output = blueoil::util::Tensor_HWC_to_CHW(output);
    expect = blueoil::util::Tensor_HWC_to_CHW(expect);
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int test_data_processor_perimagestandardization() {
  blueoil::Tensor input({1, 1, 3}, reinterpret_cast<float *>(data_processor_input));
  blueoil::Tensor expect({1, 1, 3}, reinterpret_cast<float *>(perimagestandardization_expect));
  input = blueoil::util::Tensor_CHW_to_HWC(input);
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);

  blueoil::Tensor output = blueoil::data_processor::PerImageStandardization(input);
  if (!output.allclose(expect, 0, 0.0001)) {
    std::cerr << "test_data_processor_perimagestandardization: output != expect" << std::endl;
    output = blueoil::util::Tensor_HWC_to_CHW(output);
    expect = blueoil::util::Tensor_HWC_to_CHW(expect);
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int test_data_processor_formatyolov2() {
  int width = 64, height = 64;
  int batch_size = 1;  // support 1 only
  int num_cell_y = 2, num_cell_x = 2;

  blueoil::data_processor::FormatYoloV2Parameters params;
  params.anchors = {{0.2, 0.2}, {0.7, 0.7}};
  params.boxes_per_cell = 2;  // len(anchors)
  params.data_format = "NHWC";
  params.image_size = std::make_pair(width, height);
  params.num_classes = 2;

  blueoil::Tensor input({batch_size, num_cell_y, num_cell_x, (params.boxes_per_cell+5) * params.boxes_per_cell});
  for (int i = 0; i < input.size(); i++) {
    input.data()[i] = static_cast<float>(i) / static_cast<float>(input.size());
  }
  blueoil::Tensor output = blueoil::data_processor::FormatYoloV2(input,
                                                                 params);
  blueoil::Tensor expect({1, 16, 6}, reinterpret_cast<float *>(yolov2test_expect));
  if (!output.allclose(expect, 0, 0.0001)) {
    std::cerr << "test_data_processor_formatyolov2: output != expect" << std::endl;
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int test_data_processor_excludelowscorebox() {
  int batch_size = 1;  // support 1 only
  float threshold = 0.5;
  blueoil::Tensor input({batch_size, 8, 6}, reinterpret_cast<float *>(excludelowscorebox_input));
  blueoil::Tensor output = blueoil::data_processor::ExcludeLowScoreBox(input,
                                                                       threshold);
  blueoil::Tensor expect({1, 4, 6}, reinterpret_cast<float *>(excludelowscorebox_expect));
  if (!output.allclose(expect, 0, 0.0001)) {
    std::cerr << "test_data_processor_formatyolov2: output != expect" << std::endl;
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int test_data_processor_nms() {
  int batch_size = 1;  // support 1 only
  blueoil::data_processor::NMSParameters params;
  params.classes = {"orange", "apple", "grape"};
  params.iou_threshold = 0.7;
  params.max_output_size = 2;
  params.per_class = true;
  blueoil::Tensor input({batch_size, 8, 6}, reinterpret_cast<float *>(nms_input));
  blueoil::Tensor output = blueoil::data_processor::NMS(input,
                                                        params);
  blueoil::Tensor expect({batch_size, 4, 6}, reinterpret_cast<float *>(nms_expect));
  if (!output.allclose(expect)) {
    std::cerr << "test_data_nms: output != expect" << std::endl;
    output.dump();
    expect.dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int main(void) {
  int status_code = 0;
  std::cerr << "test_data_processor_resize" << std::endl;
  status_code = test_data_processor_resize();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::cerr << "test_data_processor_divideby255" << std::endl;
  status_code = test_data_processor_divideby255();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::cerr << "test_data_processor_perimagestandardization" << std::endl;
  status_code = test_data_processor_perimagestandardization();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::cerr << "test_data_processor_formatyolov2" << std::endl;
  status_code = test_data_processor_formatyolov2();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::cerr << "test_data_processor_excludelowscorebox" << std::endl;
  status_code = test_data_processor_excludelowscorebox();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::cerr << "test_data_processor_nms" << std::endl;
  status_code = test_data_processor_nms();
  if (status_code != EXIT_SUCCESS) {
    std::exit(status_code);
  }
  std::exit(EXIT_SUCCESS);
}
