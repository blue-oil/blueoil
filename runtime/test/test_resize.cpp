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

#include "blueoil.hpp"
#include "blueoil_image.hpp"
#ifdef USE_OPENCV
#include "blueoil_opencv.hpp"
#endif
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

float test_expect_bilinear[3][4][4] =
  { {  // Red
     {95, 19,  0,  0},
     {19, 34, 10,  0},
     { 0, 10, 34, 19},
     { 0,  0, 19, 95}
     },
    {  // Green
     { 0,  0, 19, 95},
     { 0, 10, 34, 19},
     {19, 34, 10,  0},
     {95, 19,  0,  0}
    },
    {  // Blue
     {  0, 128, 128,   0},
     {128, 153, 153, 128},
     {128, 153, 153, 128},
     {  0, 128, 128,   0},
    } };

int test_resize() {
  // CHW (3-channel, height, width)
  int width = 4, height = 4;
  blueoil::Tensor input({3, 8, 8}, reinterpret_cast<float *>(test_input));
  blueoil::Tensor expect({3, 4, 4}, reinterpret_cast<float *>(test_expect));
  blueoil::Tensor expect_bilinear({3, 4, 4}, reinterpret_cast<float *>(test_expect_bilinear));
  input = blueoil::util::Tensor_CHW_to_HWC(input);
  expect = blueoil::util::Tensor_CHW_to_HWC(expect);
  expect_bilinear = blueoil::util::Tensor_CHW_to_HWC(expect_bilinear);
  blueoil::Tensor output = blueoil::image::Resize(input, width, height,
                                                  blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR);
  if (!output.allclose(expect)) {
    std::cerr << "test_resize: output != expect (nearest-neighbor)" << std::endl;
    blueoil::util::Tensor_HWC_to_CHW(output).dump();
    blueoil::util::Tensor_HWC_to_CHW(expect).dump();
    return EXIT_FAILURE;
  }
  blueoil::Tensor output_bilinear = blueoil::image::Resize(input, width, height,
                                                           blueoil::image::RESIZE_FILTER_BI_LINEAR);
  if (!output_bilinear.allclose(expect_bilinear, 0.0, 1.0)) {
    std::cerr << "test_resize: output_bilinear != expect_bilinear" << std::endl;
    blueoil::util::Tensor_HWC_to_CHW(output_bilinear).dump();
    blueoil::util::Tensor_HWC_to_CHW(expect_bilinear).dump();
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

int command_resize(int argc, char **argv) {
#ifdef USE_OPENCV
  char *infile = argv[1];
  int width = atoi(argv[2]);
  int height = atoi(argv[3]);
  blueoil::image::ResizeFilter filter = blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR;
  if ((width <= 0) || (height <= 0)) {
    std::cerr << "width <= 0 || height <= 0" << std::endl;
    return EXIT_FAILURE;
  }
  if (5 < argc) {
    int f = atoi(argv[5]);
    if ((f != blueoil::image::RESIZE_FILTER_NEAREST_NEIGHBOR) &&
        (f != blueoil::image::RESIZE_FILTER_BI_LINEAR)) {
      std::cerr << "unknown filter:" << f << std::endl;
      return EXIT_FAILURE;
    }
    filter = static_cast<blueoil::image::ResizeFilter>(f);
  }
  char *outfile = argv[4];
  std::cout << "infile:" << infile << " width:" << width <<
    " height:" << height << " outfile:" << outfile << std::endl;

  blueoil::Tensor input = blueoil::Tensor_loadImage(infile);

  blueoil::Tensor output = blueoil::image::Resize(input, width, height,
                                                  filter);
  cv::Mat img2 = blueoil::opencv::Tensor_toCVMat(output);
  cv::imwrite(outfile, img2);
#endif  // USE_OPENCV
  return EXIT_SUCCESS;
}

int main(int argc, char **argv) {
  int status_code = 0;
  if ((argc == 5) || (argc == 6)) {
    status_code = command_resize(argc, argv);
    std::exit(status_code);
  }
  if (argc == 1) {
    status_code = test_resize();
    if (status_code != EXIT_FAILURE) {
      std::exit(status_code);
    }
    std::exit(EXIT_SUCCESS);
  }
  std::cerr <<
    "Usage: " << argv[0] << " # unit test. no news is good news" << std::endl <<
    "Usage: " << argv[0] << " <input image> <dstWidth> <dstHeight> <output  image> [<filter(1:nn,2:bl)>]" << std::endl;
  std::exit(EXIT_SUCCESS);
}
