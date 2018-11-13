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

#pragma once
#include "HLS/ac_int.h"
#include "HLS/hls.h"
#include "common/global.h"

using namespace ihc;
using namespace std;
using std::cout;
using std::endl;

typedef uint32 T_q_hls;
typedef T_q_hls T_in_hls;
typedef T_q_hls T_k_hls;
typedef int16 T_out_hls;
typedef uint8 T_byte;

typedef T_in_hls T_A_hls;
typedef T_out_hls T_Y_hls;
typedef T_k_hls T_B_hls;

#define NBITS_BW_IN 256
#define NBITS_BW_OUT 256
#define NBITS_BW_K 256

#define NBITS_BW_IN_HALF 128
#define NBITS_BW_OUT_HALF 128
#define NBITS_BW_K_HALF 128
