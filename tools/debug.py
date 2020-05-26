# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import numpy as np


def derive_total_pads(i, o, k, s):
    return (i + k - ((o - 1) // s)) // 2


def derive_pads(in_, out, ker, st):
    pads = derive_total_pads(in_, out, ker, st)
    pad = pads // 2

    if pads % 2 == 0:  # pads is even
        return (pad, pad)
    else:  # pads is even
        return (pad, pad + 1)  # left pad has additional 1


# im2col
# This implementation is not completely generalized
# x: np.ndarray[N, H, W, D]
# kernel_shape: list[H, W]
# strides: list[H, W]
# padding: str 'SAME' or 'VALID'
def im2col(x, kernel_shape=[3, 3], padding='SAME', strides=[1, 1, 1]):
    if x.shape[0] is not 1:
        print('input batch size must be 1')
        raise NotImplemented

    in_h = x.shape[1]
    in_w = x.shape[2]
    in_d = x.shape[3]

    ker_h = kernel_shape[0]
    ker_w = kernel_shape[1]
    ker_d = in_d

    st_h = strides[0]
    st_w = strides[1]

    if padding is 'SAME':
        out_h = in_h
        out_w = in_w
        pads_h = derive_pads(in_h, out_h, ker_h, st_h)
        pads_w = derive_pads(in_w, out_w, ker_w, st_w)
    elif padding is 'VALID':
        out_h = (in_h - ker_h) // st_h + 1
        out_w = (in_w - ker_w) // st_w + 1
        pads_h = (0, 0)
        pads_w = (0, 0)
    else:
        print('padding type can accept only "SAME" or "VALID"')
        raise NotImplemented

    n_row = out_h * out_w
    n_col = ker_h * ker_w * in_d

    inp = np.pad(x, ((0, 0), pads_h, pads_w, (0, 0)), 'constant')
    out = np.empty((out_h, out_w, n_col), dtype=x.dtype)

    # this is slightly generalized but not perfect
    # for example, im2col concatinates 3 dims from the innermost dim
    # by full of the depth and part of the weight and height.
    for hi in range(0, out_h, st_h):
        for wj in range(0, out_w, st_h):
            out[hi, wj, :] = inp[0, hi:hi + ker_h, wj:wj + ker_w, :].flatten()

    return out.reshape((n_row, n_col))


# QtizedMatrix
# This implementation is not completely generalized because,
# In theory we can extend this into multi dimentional tensor.
class QuantizedMatrix(object):
    ARRANGEMENT = ['Sequential', 'BitInterleaving', 'WordInterleaving']

    def __init__(self, x, nbits=1, nbits_per_word=32, arrangement='WordInterleaving'):
        if len(x.shape) is not 2:
            print('given tensor should be 2 dimentional')
            raise NotImplemented

        if x.dtype != np.uint32:
            print('given tensor type should be np.uint32')
            raise NotImplemented

        self.orig = x
        self.nbits = nbits
        self.nbits_per_word = nbits_per_word
        self.arrangement = arrangement

        self.row = x.shape[0]
        self.col = self.derive_num_qwords(x.shape[1])

        self.qmat = np.empty([self.row, self.col], dtype=np.uint32)
        self.make_packed_matrix(self.qmat)

    def derive_num_qwords(self, n):
        if self.arrangement == 'Sequential':
            # called "alternating fll channels" in the wiki
            return ((n + (self.nbits_per_word - 1)) // self.nbits_per_word) * self.nbits
        if self.arrangement == 'BitInterleaving':
            # called "sequential" in the wiki
            return ((n * self.nbits) + (self.nbits_per_word - 1)) // self.nbits_per_word
        elif self.arrangement == 'WordInterleaving':
            # called "alternating channels" in the wiki
            return ((n + (self.nbits_per_word - 1)) // self.nbits_per_word) * self.nbits
        else:
            raise NotImplemented

    def derive_full_bit_qwords(self, n):
        if self.arrangement == 'Sequential':
            # called "alternating fll channels" in the wiki
            return ((n + (self.nbits_per_word - 1)) // self.nbits_per_word) * self.nbits
        if self.arrangement == 'BitInterleaving':
            # called "sequential" in the wiki
            return ((n * self.nbits) + (self.nbits_per_word - 1)) // self.nbits_per_word
        elif self.arrangement == 'WordInterleaving':
            # called "alternating channels" in the wiki
            return ((n + (self.nbits_per_word - 1)) // self.nbits_per_word) * self.nbits
        else:
            raise NotImplemented

    def make_packed_matrix(self, qmat):
        if self.arrangement == 'Sequential':
            raise NotImplemented
        if self.arrangement == 'BitInterleaving':
            raise NotImplemented
        elif self.arrangement == 'WordInterleaving':
            qmat = np.zeros([self.row, self.col], dtype=np.uint32)

            orig_col = self.orig.shape[1]
            full_bit_col = orig_col // self.nbits_per_word
            residual_bit_col = orig_col % self.nbits_per_word

            def pack_bits(x, num_data, row, col_offset):
                out = np.zeros([self.nbits], dtype=np.uint32)

                for bi in reversed(range(0, num_data)):
                    data = x[row, col_offset + bi]
                    for bc in range(0, self.nbits):
                        bit = data & 0x1
                        out[bc] |= (bit << bi)
                        data = data >> 1
                return out

            for hi in range(0, self.row):
                in_col_offset = 0
                out_col_offset = 0

                for c in range(0, full_bit_col):
                    out = pack_bits(self.orig, self.nbits_per_word, hi, in_col_offset)
                    for x in range(self.nbits):
                        print(f"{c*self.nbits+x}: {format(out[x], '032b')}")

                    import ipdb
                    ipdb.set_trace()

                    qmat[hi, out_col_offset:out_col_offset + self.nbits] = out
                    in_col_offset += self.nbits_per_word
                    out_col_offset += self.nbits

                if residual_bit_col != 0:
                    out = pack_bits(self.orig, residual_bit_col, hi, in_col_offset)
                    qmat[hi, out_col_offset:out_col_offset + self.nbits] = out
        else:
            raise NotImplemented

    @property
    def data(self):
        return self.qmat


def print_qmatrix(x):
    binary_mat = np.array([bin(v) for v in x.flatten()]).reshape(x.shape)
    print(binary_mat)


def sample_input():
    in_n = 1
    in_h = 4
    in_w = 4
    in_d = 4
    in_shape = [in_n, in_h, in_w, in_d]

    ker_h = 3
    ker_w = 3

    test_seq = np.array(np.arange(in_d), ndmin=2)
    ones = np.array(np.ones(in_h * in_w), ndmin=2).T
    test_mat = ones * test_seq
    x = test_mat.reshape(in_shape)

    out = im2col(x, [ker_h, ker_w])
    out = out.astype(np.uint32)
    qm = QuantizedMatrix(out, 2)

    return out


def sample_kernel():
    ker_n = 16
    ker_h = 3
    ker_w = 3
    ker_d = 8
    ker_shape = [ker_n, ker_h, ker_w, ker_d]

    filter_h = 3
    filter_w = 3

    test_seq = np.array(np.arange(ker_d * ker_h * ker_w), ndmin=2)
    ones = np.array(np.ones(ker_n), ndmin=2).T
    test_mat = ones * test_seq

    #  out = im2col(x, [ker_h, ker_w])
    x = test_mat.astype(np.uint32)
    qm = QuantizedMatrix(x, 2)

    import ipdb
    ipdb.set_trace()

    return out
