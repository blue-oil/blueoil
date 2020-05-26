/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

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

#ifndef TYPES_H
#define TYPES_H

#include <climits>
#include <type_traits>
#include "func/impl/pop_count.h"

#ifdef __cpp_lib_byte
#include <cstddef>
using BYTE = std::byte;
#else
enum class byte : unsigned char {};
using BYTE = byte;
#endif

typedef uint32_t T_UINT;
typedef int32_t  T_INT;
typedef float    T_FLOAT;
typedef uint8_t  T_UINT8;
typedef int8_t   T_INT8;
typedef uint16_t  T_UINT16;
typedef int16_t   T_INT16;

#define QUANTIZED_NOT_PACKED uint8_t

template <typename pack_type>
class QuantizedPacked {
 public:
  using T = pack_type;
  using base_t = std::remove_cv_t<T>;
  static constexpr std::size_t BitCount = sizeof(pack_type) * CHAR_BIT;
  QuantizedPacked() = default;
  explicit QuantizedPacked(const T val) : val(val) {}
  explicit operator base_t() const { return val; }
  template <typename U, std::enable_if_t<std::is_same<base_t, std::remove_cv_t<U>>::value, int> = 0>
  QuantizedPacked<T>& operator|=(const QuantizedPacked<U>& that) {
    val |= that.val;
    return *this;
  }
  base_t Raw() const { return val; }
 private:
  T val;
} __attribute__ ((packed));

template <typename T1, typename T2,
    std::enable_if_t<std::is_same<std::remove_cv_t<T1>, std::remove_cv_t<T2>>::value, int> = 0>
inline auto operator^(const QuantizedPacked<T1>& lhs, const QuantizedPacked<T2>& rhs) {
  using packed_t = QuantizedPacked<std::remove_cv_t<T1>>;
  return packed_t(lhs.Raw() ^ rhs.Raw());
}
template <typename pack_type>
inline auto operator~(const QuantizedPacked<pack_type>& x) {
  return QuantizedPacked<std::remove_cv_t<pack_type>>(~x.Raw());
}
template <typename pack_type>
inline int pop_count(const QuantizedPacked<pack_type>& x) {
  return dlk::impl::pop_count(x.Raw());
}

template <typename T>
struct Base {
  using type = T;
};

template <typename T>
struct Base<QuantizedPacked<T>> {
  using type = T;
};

using BIN_CONV_OUTPUT = T_INT16;

#endif // TYPES_H
