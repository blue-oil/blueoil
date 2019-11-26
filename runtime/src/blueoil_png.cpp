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

#include <png.h>
#include <cassert>
#include <string>
#include <sstream>
#include "blueoil.hpp"
#include "blueoil_png.hpp"


namespace blueoil {
namespace png {

Tensor Tensor_fromPNGFile(const std::string filename) {
  std::stringstream ss;
  png_uint_32 png_width, png_height;
  int bpp, color_type, interlace_method;
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    ss << "can't open file::" << filename;
    throw std::runtime_error(ss.str());
  }
  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
                                               NULL, NULL, NULL);
  if (!png_ptr) {
    throw std::runtime_error("failed to png_create_read_struct");
  }
  png_infop png_info_ptr = png_create_info_struct(png_ptr);
  if (!png_info_ptr) {
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    throw std::runtime_error("failed to png_create_info_struct");
  }
  if (setjmp(png_jmpbuf(png_ptr))) {
    png_destroy_read_struct(&png_ptr, &png_info_ptr, NULL);
    fclose(fp);
    throw std::runtime_error("failed to png read");
  }
  // read header
  png_init_io(png_ptr, fp);
  png_read_info(png_ptr, png_info_ptr);

  // force any PNG to PNG24(RGB format)
  png_set_strip_alpha(png_ptr);
  png_set_expand(png_ptr);
  png_set_palette_to_rgb(png_ptr);
  png_set_expand_gray_1_2_4_to_8(png_ptr);
  png_set_gray_to_rgb(png_ptr);
  png_read_update_info(png_ptr, png_info_ptr);
  png_get_IHDR(png_ptr, png_info_ptr, &png_width, &png_height, &bpp,
               &color_type, &interlace_method, NULL, NULL);
  if (interlace_method != PNG_INTERLACE_NONE) {
    png_set_interlace_handling(png_ptr);
  }
  assert(bpp == 8);
  assert(color_type == 2);  // PNG24
  blueoil::Tensor tensor({static_cast<int>(png_height),
          static_cast<int>(png_width), 3});
  png_bytepp image_data = (png_bytepp) malloc(3 * png_height * sizeof(png_bytep));
  png_bytep image_data_rows = (png_bytep) malloc(3 * png_width * png_height);
  for (png_uint_32 y = 0 ; y < png_height ; y++) {
    image_data[y] = image_data_rows + 3 * y * png_width;
  }
  png_read_image(png_ptr, image_data);
  for (png_uint_32 y = 0 ; y < png_height ; y++) {
    png_bytep image_row = image_data[y];
    float *tensor_row = tensor.dataAsArray({static_cast<int>(y), 0, 0});
    for (png_uint_32 i = 0 ; i < 3 * png_width ; i++) {
      tensor_row[i] = image_row[i];
    }
  }
  // terminate
  free(image_data_rows);
  free(image_data);
  fclose(fp);
  png_destroy_read_struct(&png_ptr, &png_info_ptr, NULL);
  return tensor;
}

}  // namespace png
}  // namespace blueoil
