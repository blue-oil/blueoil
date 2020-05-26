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

#include "matrix/multiplication.h"
#include <memory>
#include "global.h"
#include "matrix/row_major_to_col_major.h"

#ifdef USE_NEON
  #include <arm_neon.h>
#elif defined USE_AVX
  #include <x86intrin.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif

namespace dlk {

namespace details {

constexpr std::size_t ceil_mod(const std::size_t n, const std::size_t mod) {
  return n + (mod - n%mod) % mod;
}

void matrix_multiplication_col3(
  MatrixView<float, MatrixOrder::RowMajor>& A,
  MatrixView<float, MatrixOrder::ColMajor>& B,
  MatrixView<float, MatrixOrder::ColMajor>& C,
  BYTE *temporary_buf) {
#ifdef USE_NEON
  auto A_colm = row_major_to_col_major(A, reinterpret_cast<float*>(temporary_buf));
#pragma omp parallel for
  for (std::size_t i = 0; i < B.cols(); ++i) {
    float32x4_t rhs0 = vdupq_n_f32((float)(*B.data(0, i)));
    float32x4_t rhs1 = vdupq_n_f32((float)(*B.data(1, i)));
    float32x4_t rhs2 = vdupq_n_f32((float)(*B.data(2, i)));

    assert(A.rows() % 4 == 0);
    for (std::size_t j = 0; j + 3 < A.rows(); j += 4) {
      float32x4_t lhs0 = vld1q_f32(A_colm.data(j, 0));
      float32x4_t lhs1 = vld1q_f32(A_colm.data(j, 1));
      float32x4_t lhs2 = vld1q_f32(A_colm.data(j, 2));

      float32x4_t r;
      r = vmulq_f32(lhs0, rhs0);
      r = vmlaq_f32(r, lhs1, rhs1);
      r = vmlaq_f32(r, lhs2, rhs2);
      vst1q_f32(C.data(j, i), r);
    }
  }
#endif
}

void matrix_multiplication_impl(
   MatrixView<float, MatrixOrder::RowMajor>& A,
   MatrixView<float, MatrixOrder::ColMajor>& B,
   MatrixView<float, MatrixOrder::ColMajor>& C,
   BYTE *temporary_buf) {
#ifdef USE_NEON
  constexpr std::size_t regblock_n = 8;
  static_assert(regblock_n <= MAX_UNROLL, "MAX_UNROLL is too small");
  constexpr std::size_t regblock_m = 4;
  const auto B_col_blocks = (B.cols() + regblock_m - 1) / regblock_m;
  float *B_buf_ptr = reinterpret_cast<float*>(temporary_buf);
  const auto B_buf_size = B_col_blocks * regblock_m * B.rows();
  for (std::size_t j = 0; j < B.cols(); j += regblock_m) {
    if (j + regblock_m <= B.cols()) {
      std::size_t k = 0;
      for (; k + regblock_m <= B.rows(); k += regblock_m) {
        const auto im0 = vld1q_f32(B.data(k, j + 0));
        const auto im1 = vld1q_f32(B.data(k, j + 1));
        const auto im2 = vld1q_f32(B.data(k, j + 2));
        const auto im3 = vld1q_f32(B.data(k, j + 3));
        const auto pm01 = vtrnq_f32(im0, im1);
        const auto pm23 = vtrnq_f32(im2, im3);
        const auto om0 = vcombine_f32(vget_low_f32(pm01.val[0]), vget_low_f32(pm23.val[0]));
        const auto om1 = vcombine_f32(vget_low_f32(pm01.val[1]), vget_low_f32(pm23.val[1]));
        const auto om2 = vcombine_f32(vget_high_f32(pm01.val[0]), vget_high_f32(pm23.val[0]));
        const auto om3 = vcombine_f32(vget_high_f32(pm01.val[1]), vget_high_f32(pm23.val[1]));
        vst1q_f32(B_buf_ptr, om0);
        B_buf_ptr += 4;
        vst1q_f32(B_buf_ptr, om1);
        B_buf_ptr += 4;
        vst1q_f32(B_buf_ptr, om2);
        B_buf_ptr += 4;
        vst1q_f32(B_buf_ptr, om3);
        B_buf_ptr += 4;
      }
      for (; k < B.rows(); ++k) {
        for (std::size_t j2 = 0; j2 < regblock_m; ++j2) {
          if (j + j2 >= B.cols()) {
            *B_buf_ptr++ = 0;
          } else {
            *B_buf_ptr++ = B(k, j + j2);
          }
        }
      }
    } else {
      for (std::size_t k = 0; k < B.rows(); ++k) {
        for (std::size_t j2 = 0; j2 < regblock_m; ++j2) {
          if (j + j2 >= B.cols()) {
            *B_buf_ptr++ = 0;
          } else {
            *B_buf_ptr++ = B(k, j + j2);
          }
        }
      }
    }
  }
#pragma omp parallel for
  for (std::size_t i = 0; i < A.rows(); i += regblock_n) {
    constexpr std::size_t A_buf_size = regblock_n * MAX_IN_C;
#ifdef _OPENMP
    std::size_t offset = A_buf_size * omp_get_thread_num();
#else
    std::size_t offset = 0;
#endif
    float *A_buf = reinterpret_cast<float*>(temporary_buf) + B_buf_size + offset;
    for (std::size_t k = 0; k < A.cols(); ++k) {
      for (std::size_t i2 = 0; i2 < regblock_n; ++i2) {
        if (i + i2 >= A.rows()) {
          A_buf[k * regblock_n + i2] = 0;
        } else {
          A_buf[k * regblock_n + i2] = A(i + i2, k);
        }
      }
    }
    float *B_buf_ptr = reinterpret_cast<float*>(temporary_buf);
    for (std::size_t j = 0; j < B.cols(); j += regblock_m) {
      if (A.rows() - i >= regblock_n && B.cols() - j >= regblock_m) {
        float *A_buf_ptr = A_buf;
        auto accum00 = vdupq_n_f32(0);
        auto accum01 = vdupq_n_f32(0);
        auto accum10 = vdupq_n_f32(0);
        auto accum11 = vdupq_n_f32(0);
        auto accum20 = vdupq_n_f32(0);
        auto accum21 = vdupq_n_f32(0);
        auto accum30 = vdupq_n_f32(0);
        auto accum31 = vdupq_n_f32(0);
        for (std::size_t k = 0; k < A.cols(); ++k) {
          const auto a0 = vld1q_f32(A_buf_ptr);
          A_buf_ptr += 4;
          const auto a1 = vld1q_f32(A_buf_ptr);
          A_buf_ptr += 4;
          const auto b = vld1q_f32(B_buf_ptr);
          B_buf_ptr += 4;
          const auto bl = vget_low_f32(b);
          const auto bh = vget_high_f32(b);
          accum00 = vmlaq_lane_f32(accum00, a0, bl, 0);
          accum01 = vmlaq_lane_f32(accum01, a1, bl, 0);
          accum10 = vmlaq_lane_f32(accum10, a0, bl, 1);
          accum11 = vmlaq_lane_f32(accum11, a1, bl, 1);
          accum20 = vmlaq_lane_f32(accum20, a0, bh, 0);
          accum21 = vmlaq_lane_f32(accum21, a1, bh, 0);
          accum30 = vmlaq_lane_f32(accum30, a0, bh, 1);
          accum31 = vmlaq_lane_f32(accum31, a1, bh, 1);
        }
        vst1q_f32(C.data(i + 0, j + 0), accum00);
        vst1q_f32(C.data(i + 4, j + 0), accum01);
        vst1q_f32(C.data(i + 0, j + 1), accum10);
        vst1q_f32(C.data(i + 4, j + 1), accum11);
        vst1q_f32(C.data(i + 0, j + 2), accum20);
        vst1q_f32(C.data(i + 4, j + 2), accum21);
        vst1q_f32(C.data(i + 0, j + 3), accum30);
        vst1q_f32(C.data(i + 4, j + 3), accum31);
      } else if (A.rows() - i >= regblock_n) {
        const auto j2max = std::min(regblock_m, B.cols() - j);
        for (std::size_t j2 = 0; j2 < j2max; ++j2) {
          float *A_buf_ptr = A_buf;
          auto accum0 = vdupq_n_f32(0.0f);
          auto accum1 = vdupq_n_f32(0.0f);
          for (std::size_t k = 0; k < A.cols(); ++k) {
            const auto a0 = vld1q_f32(A_buf_ptr);
            A_buf_ptr += 4;
            const auto a1 = vld1q_f32(A_buf_ptr);
            A_buf_ptr += 4;
            const auto b = vdupq_n_f32(B(k, j + j2));
            accum0 = vmlaq_f32(accum0, a0, b);
            accum1 = vmlaq_f32(accum1, a1, b);
          }
          vst1q_f32(C.data(i + 0, j + j2), accum0);
          vst1q_f32(C.data(i + 4, j + j2), accum1);
        }
      } else if (B.cols() - j >= regblock_m) {
        const auto i2max = std::min(regblock_n, A.rows() - i);
        for (std::size_t i2 = 0; i2 < i2max; ++i2) {
          B_buf_ptr = reinterpret_cast<float*>(temporary_buf) + j * A.cols();
          auto accum0 = vdupq_n_f32(0.0f);
          auto accum1 = vdupq_n_f32(0.0f);
          auto accum2 = vdupq_n_f32(0.0f);
          auto accum3 = vdupq_n_f32(0.0f);
          std::size_t k;
          for (k = 0; k+3 < A.cols(); k += 4) {
            const auto a = vld1q_f32(A.data(i + i2, k));
            const auto b = vld4q_f32(B_buf_ptr);
            B_buf_ptr += 16;
            accum0 = vmlaq_f32(accum0, a, b.val[0]);
            accum1 = vmlaq_f32(accum1, a, b.val[1]);
            accum2 = vmlaq_f32(accum2, a, b.val[2]);
            accum3 = vmlaq_f32(accum3, a, b.val[3]);
          }
          const auto res0 = vpadd_f32(vget_low_f32(accum0), vget_high_f32(accum0));
          const auto res1 = vpadd_f32(vget_low_f32(accum1), vget_high_f32(accum1));
          const auto res2 = vpadd_f32(vget_low_f32(accum2), vget_high_f32(accum2));
          const auto res3 = vpadd_f32(vget_low_f32(accum3), vget_high_f32(accum3));
          const auto res01 = vpadd_f32(res0, res1);
          const auto res23 = vpadd_f32(res2, res3);
          auto res = vcombine_f32(res01, res23);
          for (; k < A.cols(); ++k) {
            const auto a = vdupq_n_f32(A(i + i2, k));
            const auto b = vld1q_f32(B_buf_ptr);
            B_buf_ptr += 4;
            res = vmlaq_f32(res, a, b);
          }
          float res_ary[4];
          vst1q_f32(res_ary, res);
          C(i + i2, j + 0) = res_ary[0];
          C(i + i2, j + 1) = res_ary[1];
          C(i + i2, j + 2) = res_ary[2];
          C(i + i2, j + 3) = res_ary[3];
        }
      } else {
        const auto i2max = std::min(regblock_n, A.rows() - i);
        const auto j2max = std::min(regblock_m, B.cols() - j);
        for (std::size_t i2 = 0; i2 < i2max; ++i2) {
          for (std::size_t j2 = 0; j2 < j2max; ++j2) {
            auto accum = vdupq_n_f32(0.0f);
            std::size_t k;
            for (k = 0; k+3 < A.cols(); k += 4) {
              accum = vmlaq_f32(accum, vld1q_f32(A.data(i + i2, k)), vld1q_f32(B.data(k, j + j2)));
            }
            float accum_ary[4];
            vst1q_f32(accum_ary, accum);
            float res = accum_ary[0] + accum_ary[1] + accum_ary[2] + accum_ary[3];
            for (; k < A.cols(); ++k) {
              res += A(i + i2, k) * B(k, j + j2);
            }
            C(i + i2, j + j2) = res;
          }
        }
      }
    }
  }
#elif defined USE_AVX
  constexpr std::size_t align = 32;
  constexpr std::size_t regblock_n = 16;
  static_assert(regblock_n <= MAX_UNROLL, "MAX_UNROLL is too small");
  constexpr std::size_t regblock_m = 4;
  const auto kmax = ceil_mod(A.cols(), regblock_m);
  const auto jmax = ceil_mod(B.cols(), regblock_m);
  auto B_buf_aligned = reinterpret_cast<void*>(temporary_buf);
  std::size_t B_buf_size = kmax * jmax;
  auto space = B_buf_size * sizeof(float) + align;
  std::align(align, kmax * jmax, B_buf_aligned, space);
  float *B_buf_ptr = reinterpret_cast<float*>(B_buf_aligned);
  for (std::size_t j = 0; j < jmax; j += regblock_m) {
    if (B.cols() - j >= regblock_m) {
      std::size_t k = 0;
      for (; k + regblock_m <= B.rows(); k += regblock_m) {
        const auto im0 = _mm_loadu_ps(B.data(k, j + 0));
        const auto im1 = _mm_loadu_ps(B.data(k, j + 1));
        const auto im2 = _mm_loadu_ps(B.data(k, j + 2));
        const auto im3 = _mm_loadu_ps(B.data(k, j + 3));
        const auto pm0 = _mm_shuffle_ps(im0, im1, 0x44);
        const auto pm1 = _mm_shuffle_ps(im2, im3, 0x44);
        const auto pm2 = _mm_shuffle_ps(im0, im1, 0xEE);
        const auto pm3 = _mm_shuffle_ps(im2, im3, 0xEE);
        const auto qm0 = _mm_shuffle_ps(pm0, pm1, 0x88);
        const auto qm1 = _mm_shuffle_ps(pm0, pm1, 0xDD);
        const auto qm2 = _mm_shuffle_ps(pm2, pm3, 0x88);
        const auto qm3 = _mm_shuffle_ps(pm2, pm3, 0xDD);
        _mm_store_ps(B_buf_ptr, qm0);
        B_buf_ptr += 4;
        _mm_store_ps(B_buf_ptr, qm1);
        B_buf_ptr += 4;
        _mm_store_ps(B_buf_ptr, qm2);
        B_buf_ptr += 4;
        _mm_store_ps(B_buf_ptr, qm3);
        B_buf_ptr += 4;
      }
      for (; k < kmax; ++k) {
        for (std::size_t j2 = 0; j2 < regblock_m; ++j2) {
          if (j + j2 >= B.cols() || k >= B.rows()) {
            *B_buf_ptr++ = 0;
          } else {
            *B_buf_ptr++ = B(k, j + j2);
          }
        }
      }
    } else {
      for (std::size_t k = 0; k < kmax; ++k) {
        for (std::size_t j2 = 0; j2 < regblock_m; ++j2) {
          if (j + j2 >= B.cols() || k >= B.rows()) {
            *B_buf_ptr++ = 0;
          } else {
            *B_buf_ptr++ = B(k, j + j2);
          }
        }
      }
    }
  }
  constexpr std::size_t A_buf_size = ceil_mod(regblock_n * ceil_mod(MAX_IN_C, regblock_m), align);
  void *A_buf_aligned = reinterpret_cast<void*>(reinterpret_cast<float*>(B_buf_aligned) + B_buf_size);
#ifdef _OPENMP
  space = omp_get_max_threads() * A_buf_size + align;
#else
  space = A_buf_size + align;
#endif
  std::align(align, omp_get_max_threads() * A_buf_size, A_buf_aligned, space);
#pragma omp parallel for
  for (std::size_t i = 0; i < A.rows(); i += regblock_n) {
#ifdef _OPENMP
    std::size_t offset = A_buf_size * omp_get_thread_num();
#else
    std::size_t offset = 0;
#endif
    float *A_buf = reinterpret_cast<float*>(A_buf_aligned) + offset;
    for (std::size_t k = 0; k < kmax; ++k) {
      for (std::size_t i2 = 0; i2 < regblock_n; ++i2) {
        if (i + i2 >= A.rows() || k >= A.cols()) {
          A_buf[k * regblock_n + i2] = 0;
        } else {
          A_buf[k * regblock_n + i2] = A(i + i2, k);
        }
      }
    }
    float *B_buf_ptr = reinterpret_cast<float*>(B_buf_aligned);
    std::size_t j = 0;
    for (; j < jmax; j += regblock_m) {
      float *A_buf_ptr = A_buf;
      auto accum00 = _mm256_setzero_ps();
      auto accum01 = _mm256_setzero_ps();
      auto accum10 = _mm256_setzero_ps();
      auto accum11 = _mm256_setzero_ps();
      auto accum20 = _mm256_setzero_ps();
      auto accum21 = _mm256_setzero_ps();
      auto accum30 = _mm256_setzero_ps();
      auto accum31 = _mm256_setzero_ps();
      for (std::size_t k = 0; k < kmax; ++k) {
        const auto a0 = _mm256_load_ps(A_buf_ptr);
        A_buf_ptr += 8;
        const auto a1 = _mm256_load_ps(A_buf_ptr);
        A_buf_ptr += 8;
        const auto b0 = _mm256_set1_ps(*B_buf_ptr++);
        accum00 = _mm256_fmadd_ps(a0, b0, accum00);
        accum01 = _mm256_fmadd_ps(a1, b0, accum01);
        const auto b1 = _mm256_set1_ps(*B_buf_ptr++);
        accum10 = _mm256_fmadd_ps(a0, b1, accum10);
        accum11 = _mm256_fmadd_ps(a1, b1, accum11);
        const auto b2 = _mm256_set1_ps(*B_buf_ptr++);
        accum20 = _mm256_fmadd_ps(a0, b2, accum20);
        accum21 = _mm256_fmadd_ps(a1, b2, accum21);
        const auto b3 = _mm256_set1_ps(*B_buf_ptr++);
        accum30 = _mm256_fmadd_ps(a0, b3, accum30);
        accum31 = _mm256_fmadd_ps(a1, b3, accum31);
      }
      const auto lane_ids = _mm256_setr_epi32(
          0, 1, 2, 3, 4, 5, 6, 7
      );
      const auto idx0 = lane_ids + _mm256_set1_epi32(i+0);
      const auto idx1 = lane_ids + _mm256_set1_epi32(i+8);
      const auto vn = _mm256_set1_epi32(A.rows());
      const auto mask0 = _mm256_cmpgt_epi32(vn, idx0);
      const auto mask1 = _mm256_cmpgt_epi32(vn, idx1);
      _mm256_maskstore_ps(C.data(i + 0, j + 0), mask0, accum00);
      _mm256_maskstore_ps(C.data(i + 8, j + 0), mask1, accum01);
      if (j + 1 >= B.cols()) continue;
      _mm256_maskstore_ps(C.data(i + 0, j + 1), mask0, accum10);
      _mm256_maskstore_ps(C.data(i + 8, j + 1), mask1, accum11);
      if (j + 2 >= B.cols()) continue;
      _mm256_maskstore_ps(C.data(i + 0, j + 2), mask0, accum20);
      _mm256_maskstore_ps(C.data(i + 8, j + 2), mask1, accum21);
      if (j + 3 >= B.cols()) continue;
      _mm256_maskstore_ps(C.data(i + 0, j + 3), mask0, accum30);
      _mm256_maskstore_ps(C.data(i + 8, j + 3), mask1, accum31);
    }
  }
#endif
}

} // namespace details

} // namespace dlk
