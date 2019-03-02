#ifndef RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_
#define RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_

#include "blueoil.hpp"

namespace blueoil {
namespace image {

enum ResizeFilter {
                   RESIZE_FILTER_NEAREST_NEIGHBOR = 1,
                   RESIZE_FILTER_BI_LINEAR = 2,
};

Tensor Resize(const Tensor& image, const int width, const int height,
              const enum ResizeFilter filter);

} // namespace image
} // namespace blueoil

#endif  // RUNTIME_INCLUDE_BLUEOIL_IMAGE_HPP_
