#include "common/global.h"
#include "cpp/utils.h"

namespace cpp {
namespace p = conv_kn2row_params;

/// estimated size buffer size
/// tiled input:          H(20) * W(20) * C(1024) * B(2)  = 819200 bit
/// partial tiled output: H(16) * W(16) * C(16)   * B(16) = 65536 bit
/// partial kernel:       H(3)  * W(3)  * C(1024) * B(16) = 147456 bit
///                                                 total:  1032192 bit
void conv_kn2row_tiling_impl(T_in in_data[], T_out out_data[], T_k k_data[],
                             T_out threshold_data[], const unsigned in_w,
                             const unsigned in_h, const unsigned in_c, const unsigned out_w,
                             const unsigned out_h, const unsigned out_c, const unsigned k_w,
                             const unsigned k_h, const unsigned pad, const unsigned stride) {
  /// just alias for better understanding
  static const unsigned out_c_low = p::num_pe;
  assert((out_c % out_c_low) == 0);
  assert(in_c <= p::max_in_c);
  assert(in_c >= p::min_in_c);

  unsigned idx_in = 0;
  unsigned idx_out = 0;
  unsigned idx_t = 0;

  int in_tile_h = p::tile_h + (k_h - 1) * 2;
  int in_tile_w = p::tile_w + (k_w - 1) * 2;

  std::cout << "conv_kn2row_tiling_impl called!!!" << std::endl;

  for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {

    T_out threshold_buf[out_c_low][p::num_thresholds];

    if (threshold_data != NULL) {
      for (unsigned oc = 0; oc < out_c_low; oc++) {
        for (unsigned i = 0; i < p::num_thresholds; i++) {
          threshold_buf[oc][i] = threshold_data[oc_high + oc + i];
        }
      }
    }

    for (int ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
      for (int iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {

        T_out in_buf[in_tile_h][in_tile_w][in_c];
        T_out out_buf[p::tile_w][p::tile_w][out_c_low];
        T_out k_buf[in_c][out_c_low];

        std::cout << "ih_high = " << ih_high << ", iw_high = " << iw_high
                  << std::endl;

        /// preload input
        for (int ih_low = 0; ih_low < in_tile_h; ++ih_low) {
          for (int iw_low = 0; iw_low < in_tile_w; ++iw_low) {
            /// index must care the padding, so we skip the padding part that
            /// doesn't exist in actuall memory.
            int ih = (ih_low + ih_high - pad);
            int iw = (iw_low + iw_high - pad);
            bool input_on = (ih > 0) && (iw > 0) && (ih < in_h) && (iw < in_w);

            for (int ic = 0; ic < in_c; ic++) {
              int idx_in = ih * in_w * in_c + iw * in_c + ic;
              in_buf[ih_low][iw_low][ic] =
                  (input_on) ? in_data[idx_in] : T_in(0);
            }
          }
        }

        /// main convolution loop
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {

            /// preload kernel
            for (int ic = 0; ic < in_c; ic++) {
              for (int oc = 0; oc < out_c_low; oc++) {
                int idx_k = kh * k_w * in_c * out_c + kw * in_c * out_c +
                            ic * out_c + oc + oc_high;
                k_buf[ic][oc] = k_data[idx_k];
              }
            }

            for (int ih = 0; ih < in_tile_h; ++ih) {
              for (int iw = 0; iw < in_tile_w; ++iw) {

                int oh = ih - kh;
                int ow = iw - kw;
                bool first_output = (kh == 0) && (kw == 0);
                bool output_on = (oh > 0) && (ow > 0) && (oh < p::tile_h) &&
                                 (ow < p::tile_w);

                for (int ic = 0; ic < in_c; ic++) {
                  T_in in_elem = in_buf[ih][iw][ic];

                  for (int oc = 0; oc < out_c_low; oc++) {
                    T_q k_elem = k_buf[ic][oc];
                    T_out acc_tmp = in_elem * k_elem;

                    if (output_on) {
                      T_out out_pre = out_buf[oh][ow][oc];
                      out_buf[oh][ow][oc] =
                          (first_output) ? acc_tmp : (acc_tmp + out_pre);
                    }
                  } // for LOOP_CONV_INPUT
                }
              }
            }
          }
        }

        /// export data in output buffer step
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
            for (int oc = 0; oc < out_c_low; oc++) {

              T_out out = out_buf[oh][ow][oc];
              T_out tmp;

              if (threshold_data != NULL) {
                T_out ts0 = threshold_buf[oc][0];
                T_out ts1 = threshold_buf[oc][1];
                T_out ts2 = threshold_buf[oc][2];
                T_out flag = threshold_buf[oc][3];

                if (flag == 1) /// increasing function
                {
                  if (out < ts0)
                    tmp = 0;
                  else if (out < ts1)
                    tmp = 1;
                  else if (out < ts2)
                    tmp = 2;
                  else
                    tmp = 3;
                } else if (flag == -1) /// decreasing function
                {
                  if (out > ts2)
                    tmp = 0;
                  else if (out > ts1)
                    tmp = 1;
                  else if (out > ts0)
                    tmp = 2;
                  else
                    tmp = 3;
                } else {
                  /// max value of 2 bits
                  T_out k = 3 * 3 * out_c * 3;
                  tmp = flag - k;
                }
              } else {
                tmp = out;
              }

              /// export out data to actual memory space.
              unsigned oh_ = ih_high + oh;
              unsigned ow_ = iw_high + ow;
              unsigned oc_ = oc_high + oc;

              bool output_on = ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
              if (output_on) {
                // printf("oh: %d, ow: %d, oc: %d\n", oh_, ow_, oc_);
                out_data[oh_ * out_w * out_c + ow_ * out_c + oc_] = tmp;
              }
            }
          }
        }
      }
    }
  }
}

void qconv_kn2row_tiling_impl(T_q in_data[], T_out out_data[], T_q k_data[],
                              T_out partial_out_data[], T_out threshold_data[],
                              unsigned in_w, unsigned in_h, unsigned in_c,
                              unsigned out_w, unsigned out_h, unsigned out_c,
                              unsigned k_w, unsigned k_h, unsigned pad) {}

} // namespace cpp