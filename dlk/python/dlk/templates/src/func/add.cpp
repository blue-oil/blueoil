#include "global.h"
#include "func/add.h"
#include "time_measurement.h"

void func_Add(T_FLOAT input[], T_FLOAT term, T_FLOAT output[],
              T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  Measurement::Start("Add");

  T_UINT elements = out_height * out_width * out_depth;

  for (T_UINT i = 0; i < elements; i++)
    output[i] = input[i] + term;

  Measurement::Stop();
}

void func_Add(T_FLOAT input[], T_FLOAT term[], T_FLOAT output[],
              T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
#ifndef RUN_AS_HLS
  Measurement::Start("Add");
#endif

  T_UINT elements = out_height * out_width * out_depth;

  for (T_UINT i = 0; i < elements; i++)
    output[i] = input[i] + term[i];

#ifndef RUN_AS_HLS
  Measurement::Stop();
#endif
}

void func_Add(T_FLOAT input[], T_FLOAT term, T_FLOAT output[],
              T_UINT out_depth) {
  func_Add(input, term, output, 1, 1, out_depth);
}

void func_Add(T_FLOAT term, T_FLOAT input[], T_FLOAT output[],
              T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  func_Add(input, term, output, out_height, out_width, out_depth);
}

void func_Add_depthwise(T_FLOAT input[], T_FLOAT terms[], T_FLOAT output[],
                        T_UINT out_height, T_UINT out_width, T_UINT out_depth) {
  T_UINT index = 0;
  for(T_UINT h = 0; h < out_height; h++)
  for(T_UINT w = 0; w < out_width; w++)
  for(T_UINT kz = 0; kz < out_depth; kz++)
    {
      output[index] = input[index] + terms[kz];
      index++;
    }
}

void func_Add_depthwise(T_FLOAT input[], T_FLOAT terms[], T_FLOAT output[],
                        T_UINT out_depth) {
  T_UINT index = 0;
  for(T_UINT kz = 0; kz < out_depth; kz++)
    {
      output[index] = input[index] + terms[kz];
      index++;
    }
}

void func_Add(T_FLOAT input[], T_FLOAT term[], T_FLOAT output[],
              T_UINT out_depth) {
  Measurement::Start("Add");

  for (T_UINT i = 0; i < out_depth; i++)
    output[i] = input[i] + term[i];

  Measurement::Stop();
}
