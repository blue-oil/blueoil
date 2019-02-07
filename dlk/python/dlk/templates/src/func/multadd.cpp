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

#include "func/multadd.h"
#include "time_measurement.h"
#include "global.h"

void func_MultAdd_depthwise(T_FLOAT input[], T_FLOAT scale[], T_FLOAT add[], T_FLOAT output[], T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("MultAdd");

#if defined(USE_NEON)
  const T_UINT size = out_height * out_width;  
#pragma omp parallel for
  for (T_UINT f = 0; f < size; f++) {

    T_FLOAT *in_temp = &input[f * out_depth];
    T_FLOAT *out_temp = &output[f * out_depth];

    T_UINT d = 0;
    for (; d < out_depth; d += 4) {
      asm volatile("vldmia %0, {d16,d17}    \t\n" // q8(d16,d17) scale
		   "vldmia %1, {d18,d19}    \t\n" // q9(d18,d19) add
		   "vldmia %2, {d20,d21}    \t\n" // q10(d20,d21) input
		   "vmla.f32 q9, q10, q8    \t\n"
		   "vstmia %3, {d18,d19}    \t\n"
		   :
		   : "r"(&scale[d]), "r"(&add[d]), "r"(in_temp), "r"(out_temp)
		   : "memory", "q8", "q9", "q10");
      in_temp += 4;
      out_temp += 4;
    }

    for (; d < out_depth; d++) {
      *out_temp++ = *in_temp++ * scale[d] + add[d];
    }
  }
#elif defined(USE_ASIMD)
  const T_UINT size = out_height * out_width;
#pragma omp parallel for
  for (T_UINT f = 0; f < size; f++) {

    T_FLOAT *in_temp = &input[f * out_depth];
    T_FLOAT *out_temp = &output[f * out_depth];

    T_UINT d = 0;
    for (; d < out_depth; d += 4) {
      asm volatile("ldr q6, [%0]    \t\n" // q6(d12,d13) scale
		   "ldr q7, [%1]    \t\n" // q7(d14,d15) add
		   "ldr q8, [%2]    \t\n" // q8(d16,d17) input
		                                         "fmla v7.4s, v8.4s, v6.4s    \t\n"
		                                         "str q7, [%3]     \t\n"
		   :
		   : "r"(&scale[d]), "r"(&add[d]), "r"(in_temp), "r"(out_temp)
		   : "memory", "v6", "v7", "v8");
      in_temp += 4;
      out_temp += 4;
    }

    for (; d < out_depth; d++) {
      *out_temp++ = *in_temp++ * scale[d] + add[d];
    }
  }
#else
  T_UINT index = 0;
  for (T_UINT f = 0; f < out_height * out_width; f++)
    for (T_UINT d = 0; d < out_depth; d++) {
      output[index] = input[index] * scale[d] + add[d];
      index++;
    }
#endif

  Measurement::Stop();
}
