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

#include "intel_hls/config.h"

#define T_buf int16

hls_avalon_slave_component void intel_hls_qconv_kn2row_tiling_impl(
    hls_avalon_slave_register_argument
        ihc::mm_master<T_in_hls, ihc::aspace<1>, ihc::awidth<32>,
                       ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> > &in_data,
    hls_avalon_slave_register_argument
        ihc::mm_master<T_out_hls, ihc::aspace<2>, ihc::awidth<32>,
                       ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> > &out_data,
    hls_avalon_slave_register_argument
        ihc::mm_master<T_k_hls, ihc::aspace<3>, ihc::awidth<32>,
                       ihc::dwidth<128>, ihc::latency<0>, ihc::maxburst<32>,
                       ihc::align<16>, ihc::waitrequest<true> > &k_data,
    hls_avalon_slave_register_argument ihc::mm_master<
        T_out_hls, ihc::aspace<4>, ihc::awidth<32>, ihc::dwidth<128>,
        ihc::latency<0>, ihc::maxburst<32>, ihc::align<16>,
        ihc::waitrequest<true> > &out_data_partial,
    hls_avalon_slave_register_argument int32 in_w,
    hls_avalon_slave_register_argument int32 in_h,
    hls_avalon_slave_register_argument int32 in_c_by_word,
    hls_avalon_slave_register_argument int32 out_w,
    hls_avalon_slave_register_argument int32 out_h,
    hls_avalon_slave_register_argument int32 out_c,
    hls_avalon_slave_register_argument int32 k_w,
    hls_avalon_slave_register_argument int32 k_h,
    hls_avalon_slave_register_argument int32 pad,
    hls_avalon_slave_register_argument int32 use_threshold) {}