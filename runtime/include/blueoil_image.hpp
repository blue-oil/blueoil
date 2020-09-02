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

#ifndef RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_

#include <string>
#include "blueoil.hpp"

namespace blueoil {
namespace image {

enum ResizeFilter {
                   RESIZE_FILTER_NEAREST_NEIGHBOR = 1,
                   RESIZE_FILTER_BI_LINEAR = 2,
};

Tensor LoadImage(const std::string filename);

Tensor Resize(const Tensor& image, const int height, const int width,
              const enum ResizeFilter filter);

}  // namespace image
}  // namespace blueoil

#endif  // RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_
