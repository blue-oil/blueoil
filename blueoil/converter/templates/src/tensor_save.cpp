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
==============================================================================*/

#include "tensor_save.h"

#include <memory>

#include "c2numpy.h"

namespace {

void save_float32_data(const std::string &name, uint32_t size, uint32_t suffix, float *data, float scale) {
  c2numpy_writer writer;
  c2numpy_init(&writer, name.c_str(), suffix, 1<<31);
  c2numpy_addcolumn(&writer, "data", C2NUMPY_FLOAT32);
  c2numpy_addcolumn(&writer, "scale", C2NUMPY_FLOAT32);

  for(int i = 0; i < size; i++) {
    c2numpy_float32(&writer, data[i]);
    c2numpy_float32(&writer, scale);
  }
  c2numpy_close(&writer);
}

} // unnamed namespace

void save_tensor(TensorView<T_FLOAT, MemoryLayout::NHWC>& tensor, const std::string& name, int32_t suffix) {
  save_float32_data(name, tensor.size(), suffix, tensor.data(), 1.0f);
}

void save_tensor(TensorView<T_FLOAT, MemoryLayout::C>& tensor, const std::string& name, int32_t suffix) {
  save_float32_data(name, tensor.size(), suffix, tensor.data(), 1.0f);
}

void save_tensor(TensorView<QUANTIZED_PACKED, MemoryLayout::ChHWBCl>& tensor, const std::string& name, int32_t suffix) {
  const auto& shape = tensor.get_shape();
  const auto depth = shape[0];
  const auto area = shape[1] * shape[2];
  const auto chunks = depth * area;
  const auto pack_size = shape[4];
  const auto elements = chunks * pack_size;
  const auto bits = shape[3];
  const auto tmp = std::make_unique<float[]>(elements);
  const auto& buf = tensor.data();
  for (std::size_t i = 0; i < area; ++i) {
    for (std::size_t d = 0; d < depth; ++d) {
      for (std::size_t z = 0; z < pack_size; ++z) {
        float val = 0.0f;
        for (std::size_t bit = 0; bit < bits; ++bit) {
          val += static_cast<float>(((buf[d * area * bits + i * bits + bit].Raw() >> z) & 1) << bit);
        }
        tmp[i * depth * pack_size + d * pack_size + z] = val;
      }
    }
  }
  save_float32_data(name, elements, suffix, tmp.get(), 1.0);
}

void save_tensor(TensorView<QUANTIZED_PACKED, MemoryLayout::HWChBCl>& tensor, const std::string& name, int32_t suffix) {
  const auto& shape = tensor.get_shape();
  const auto chunks = shape[0] * shape[1] * shape[2];
  const auto pack_size = shape[4];
  const auto elements = chunks * pack_size;
  const auto bits = shape[3];
  const auto tmp = std::make_unique<float[]>(elements);
  const auto& buf = tensor.data();
  for (std::size_t i = 0; i < chunks; ++i) {
    for (std::size_t z = 0; z < pack_size; ++z) {
      float val = 0.0f;
      for (std::size_t bit = 0; bit < bits; ++bit) {
        val += static_cast<float>(((buf[i * bits + bit].Raw() >> z) & 1) << bit);
      }
      tmp[i * pack_size + z] = val;
    }
  }
  save_float32_data(name, elements, suffix, tmp.get(), 1.0);
}
