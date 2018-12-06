#include "global.h"
#include "func/pad.h"
#include "time_measurement.h"

void func_Pad(T_FLOAT input[], int32_t padding[], T_FLOAT output[],
              T_UINT in_depth, T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
#ifndef RUN_AS_HLS
  Measurement::Start("Pad");
#endif

  T_UINT elements = out_height * out_width * out_depth;
  T_UINT out_idx = 0;
  T_UINT in_idx = 0;

  T_UINT prep = padding[6];
  T_UINT posp = padding[7];

  for (T_UINT i = 0; i < elements; i+=(prep + in_depth + posp)) {
    out_idx = i;
    for (T_UINT pre = 0; pre < prep; pre++) {
      output[out_idx] = 0;
      out_idx++;
    }
    for (T_UINT j = 0; j < in_depth; j++) {
      output[out_idx] = input[in_idx];
      out_idx++;
      in_idx++;
    }
    for (T_UINT pos = 0; pos < posp; pos++) {
      output[out_idx] = 0;
      out_idx++;
    }
  }

#ifndef RUN_AS_HLS
  Measurement::Stop();
#endif
}