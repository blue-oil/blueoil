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

/*
 * const unsigned char *indata, unsigned long *outdata, int num_data
 * num_data: must be multiples of 8
 */

	.thumb
	.thumb_func
	.syntax unified
	.global pack2bits
pack2bits:
	pld		[r0]
	vmov.i8		d16, #1
	vmov.i8		d17, #2
1:
	vld1.8		{d0-d3}, [r0]!
	vld1.8		{d4-d7}, [r0]!

	vtrn.8		d0,d1			@ transpose 8x8
	vtrn.8		d2,d3
	vtrn.8		d4,d5
	vtrn.8		d6,d7
	vtrn.16		d0,d2
	vtrn.16		d1,d3
	vtrn.16		d4,d6
	vtrn.16		d5,d7
	vtrn.32		d0,d4
	vtrn.32		d1,d5
	vtrn.32		d2,d6
	vtrn.32		d3,d7
	pld		[r0]

	// bit 0 & 1
	vand		d20, d0, d16
	vand		d19, d0, d17
	vand		d18, d1, d16
	vshr.u8		d21, d19, #1
	vshl.u8		d18, d18, #1
	vand		d19, d1, d17
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 2
	vand		d18, d2, d16
	vand		d19, d2, d17
	vshl.u8		d18, d18, #2
	vshl.u8		d19, d19, #1
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 3
	vand		d18, d3, d16
	vand		d19, d3, d17
	vshl.u8		d18, d18, #3
	vshl.u8		d19, d19, #2
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 4
	vand		d18, d4, d16
	vand		d19, d4, d17
	vshl.u8		d18, d18, #4
	vshl.u8		d19, d19, #3
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 5
	vand		d18, d5, d16
	vand		d19, d5, d17
	vshl.u8		d18, d18, #5
	vshl.u8		d19, d19, #4
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 6
	vand		d18, d6, d16
	vand		d19, d6, d17
	vshl.u8		d18, d18, #6
	vshl.u8		d19, d19, #5
	vorr		d20, d20, d18
	vorr		d21, d21, d19
	// bit 7
	vand		d18, d7, d16
	vand		d19, d7, d17
	vshl.u8		d18, d18, #7
	vshl.u8		d19, d19, #6
	vorr		d20, d20, d18
	vorr		d21, d21, d19

	vst2.32		{d20-d21}, [r1]!
	subs		r2, #64
	bgt		1b

	bx		lr
