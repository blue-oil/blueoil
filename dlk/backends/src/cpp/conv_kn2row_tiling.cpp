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
                             const unsigned in_h, const unsigned in_c,
                             const unsigned out_w, const unsigned out_h,
                             const unsigned out_c, const unsigned k_w,
                             const unsigned k_h, const unsigned pad,
                             const unsigned stride) {
  /// just alias for better understanding
  static const unsigned out_c_low = p::num_pe;
  assert((out_c % out_c_low) == 0);
  assert(in_c <= p::max_in_c);
  assert(in_c >= p::min_in_c);
  assert(k_h <= p::max_k_h);
  assert(k_h >= p::min_k_h);
  assert(k_w <= p::max_k_w);
  assert(k_w >= p::min_k_w);

  for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
    T_out threshold_buf[out_c_low][p::num_thresholds];

    if (threshold_data != NULL) {
      for (unsigned oc = 0; oc < out_c_low; oc++) {
        for (unsigned i = 0; i < p::num_thresholds; i++) {
          unsigned idx_th = (oc_high + oc) * p::num_thresholds + i;
          threshold_buf[oc][i] = threshold_data[idx_th];
        }
      }
    }

    for (int ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
      for (int iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {
        T_in in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c];
        T_out out_buf[p::tile_w][p::tile_w][out_c_low];
        T_k k_buf[p::max_in_c][out_c_low];

        // std::cout << "ih_high = " << ih_high << ", iw_high = " << iw_high
        //           << std::endl;

        /// preload input
        for (int ih_low = 0; ih_low < p::in_tile_h; ++ih_low) {
          for (int iw_low = 0; iw_low < p::in_tile_w; ++iw_low) {
            /// index must care the padding, so we skip the padding part that
            /// doesn't exist in actuall memory.
            int ih = (ih_low + ih_high - pad);
            int iw = (iw_low + iw_high - pad);
            bool input_on =
                (ih >= 0) && (iw >= 0) && (ih < in_h) && (iw < in_w);

            for (int ic = 0; ic < in_c; ic++) {
              int idx_in = ih * in_w * in_c + iw * in_c + ic;
              in_buf[ih_low][iw_low][ic] =
                  (input_on) ? in_data[idx_in] : T_in(0);
              // printf("in_buf[%d][%d][%d] = %d\n", ih_low, iw_low, ic,
              //        int(in_buf[ih_low][iw_low][ic]));
            }
          }
        }

        /// initialize output_buf
        /// TODO: this could be done at the same time in the accumuratoin
        /// step.
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
            for (int oc = 0; oc < out_c_low; ++oc) {
              out_buf[oh][ow][oc] = 0;
            }
          }
        }

        /// main convolution loop
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {
            /// preload kernel
            for (int ic = 0; ic < in_c; ic++) {
              for (int oc = 0; oc < out_c_low; oc++) {
                /// currently kernel oerder is NoHWCNi, which means the
                /// outermost dimension "N" is split into 2 high and low parts.
                int idx_k = (kh * k_w * in_c * out_c_low) +
                            (kw * in_c * out_c_low) + (ic * out_c_low) + oc +
                            (oc_high * k_h * k_w * in_c);
                k_buf[ic][oc] = k_data[idx_k];
              }
            }

            for (int ih = 0; ih < p::in_tile_h; ++ih) {
              for (int iw = 0; iw < p::in_tile_w; ++iw) {
                int oh = ih - kh;
                int ow = iw - kw;
                bool output_on = (oh >= 0) && (ow >= 0) && (oh < p::tile_h) &&
                                 (ow < p::tile_w);

                for (int ic = 0; ic < in_c; ic++) {
                  T_in in_elem = in_buf[ih][iw][ic];

                  for (int oc = 0; oc < out_c_low; oc++) {
                    T_k k_elem = k_buf[ic][oc];
                    T_out acc_tmp = in_elem * k_elem;

                    if (output_on) {
                      out_buf[oh][ow][oc] += acc_tmp;
                      // printf("in_elem: %d, k_elem: %d\n",
                      //        in_elem, k_elem);
                      // printf("tmp: %d, out_buf[%d][%d][%d] = %d\n",
                      //        int(acc_tmp), oh, ow, oc,
                      //        int(out_buf[oh][ow][oc]));
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

              bool output_on =
                  ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
              if (output_on) {
                int idx_out = oh_ * out_w * out_c + ow_ * out_c + oc_;
                // printf("oh: %d, ow: %d, oc: %d, idx: %d, tmp: %d\n", oh_,
                // ow_,
                //        oc_, idx_out, tmp);
                out_data[idx_out] = tmp;
              }
            }
          }
        }
      }
    }
  }
}

void qconv_kn2row_tiling_impl(T_q in_data[], T_out out_data[], T_q k_data[],
                              T_out threshold_data[], const unsigned in_w,
                              const unsigned in_h, const unsigned in_c_by_word,
                              const unsigned in_b, const unsigned out_w,
                              const unsigned out_h, const unsigned out_c,
                              const unsigned k_w, const unsigned k_h,
                              const unsigned pad) {
  /// just alias for better understanding
  static const unsigned out_c_low = p::num_pe;
  assert((out_c % out_c_low) == 0);
  assert(in_c_by_word <= p::max_in_c_by_word);
  assert(in_c_by_word >= p::min_in_c_by_word);
  assert(in_b <= p::max_in_b);
  assert(in_b >= p::min_in_b);
  assert(k_h <= p::max_k_h);
  assert(k_h >= p::min_k_h);
  assert(k_w <= p::max_k_w);
  assert(k_w >= p::min_k_w);

  for (int oc_high = 0; oc_high < out_c; oc_high += out_c_low) {
    T_out threshold_buf[out_c_low][p::num_thresholds];

    if (threshold_data != NULL) {
      for (unsigned oc = 0; oc < out_c_low; oc++) {
        for (unsigned i = 0; i < p::num_thresholds; i++) {
          unsigned idx_th = (oc_high + oc) * p::num_thresholds + i;
          threshold_buf[oc][i] = threshold_data[idx_th];
        }
      }
    }

    for (int ih_high = 0; ih_high < in_h + 2 * pad; ih_high += p::tile_h) {
      for (int iw_high = 0; iw_high < in_w + 2 * pad; iw_high += p::tile_w) {
        T_q in_buf[p::in_tile_h][p::in_tile_w][p::max_in_c_by_word]
                  [p::max_in_b];
        T_out out_buf[p::tile_w][p::tile_w][out_c_low];
        T_q k_buf[p::max_in_c_by_word][out_c_low];

        // std::cout << "ih_high = " << ih_high << ", iw_high = " << iw_high
        //           << std::endl;

        /// preload input
        for (int ih_low = 0; ih_low < p::in_tile_h; ++ih_low) {
          for (int iw_low = 0; iw_low < p::in_tile_w; ++iw_low) {
            /// index must care the padding, so we skip the padding part that
            /// doesn't exist in actuall memory.
            int ih = (ih_low + ih_high - pad);
            int iw = (iw_low + iw_high - pad);
            bool input_on =
                (ih >= 0) && (iw >= 0) && (ih < in_h) && (iw < in_w);

            for (int ic = 0; ic < in_c_by_word; ic++) {
              for (int ib = 0; ib < in_b; ib++) {
                int idx_in = ih * in_w * in_c_by_word * in_b +
                             iw * in_c_by_word * in_b + ic * in_b + ib;
                in_buf[ih_low][iw_low][ic][ib] =
                    (input_on) ? in_data[idx_in] : T_q(0);
                // printf("in_buf[%d][%d][%d] = %d\n", ih_low, iw_low, ic,
                //        int(in_buf[ih_low][iw_low][ic]));
              }
            }
          }
        }

        /// initialize output_buf
        /// TODO: this could be done at the same time in the accumuratoin
        /// step.
        for (int oh = 0; oh < p::tile_h; ++oh) {
          for (int ow = 0; ow < p::tile_w; ++ow) {
            for (int oc = 0; oc < out_c_low; ++oc) {
              out_buf[oh][ow][oc] = 0;
            }
          }
        }

        /// main convolution loop
        for (int kh = 0; kh < k_h; ++kh) {
          for (int kw = 0; kw < k_w; ++kw) {
            /// preload kernel
            for (int ic = 0; ic < in_c_by_word; ic++) {
              for (int ib = 0; ib < in_b; ib++) {
                for (int oc = 0; oc < out_c_low; oc++) {
                  /// currently kernel oerder is NoHWCNi, which means the
                  /// outermost dimension "N" is split into 2 high and low
                  /// parts. we should be carefull when compute the index.
                  int idx_k = (kh * k_w * in_c_by_word * out_c_low) +
                              (kw * in_c_by_word * out_c_low) +
                              (ic * out_c_low) + oc +
                              (oc_high * k_h * k_w * in_c_by_word);
                  k_buf[ic][oc] = k_data[idx_k];
                }
              }
            }

            for (int ih = 0; ih < p::in_tile_h; ++ih) {
              for (int iw = 0; iw < p::in_tile_w; ++iw) {
                int oh = ih - kh;
                int ow = iw - kw;
                bool output_on = (oh >= 0) && (ow >= 0) && (oh < p::tile_h) &&
                                 (ow < p::tile_w);

                for (int ic = 0; ic < in_c_by_word; ic++) {
                  T_q in_elems[p::max_in_b];
                  for (int ib = 0; ib < in_b; ib++) {
                    in_elems[ib] = in_buf[ih][iw][ic][ib];
                  }

                  for (int oc = 0; oc < out_c_low; oc++) {
                    T_q k_elem = k_buf[ic][oc];
                    T_out acc_tmp = PE(k_elem, in_elems[0], in_elems[1]);

                    if (output_on) {
                      out_buf[oh][ow][oc] += acc_tmp;
                      // printf("in_elem: %d, k_elem: %d\n",
                      //        in_elem, k_elem);
                      // printf("tmp: %d, out_buf[%d][%d][%d] = %d\n",
                      //        int(acc_tmp), oh, ow, oc,
                      //        int(out_buf[oh][ow][oc]));
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

              bool output_on =
                  ((oh_ < out_h) && (ow_ < out_w) && (oc_ < out_c));
              if (output_on) {
                int idx_out = oh_ * out_w * out_c + ow_ * out_c + oc_;
                // printf("oh: %d, ow: %d, oc: %d, idx: %d, tmp: %d\n", oh_,
                // ow_,
                //        oc_, idx_out, tmp);
                out_data[idx_out] = tmp;
              }
            }
          }
        }
      }
    }
  }
}

} // namespace cpp