        .text
	.globl  pack2bits_v8
	.p2align        2
	.type   pack2bits_v8,@function
pack2bits_v8:
	// %bb.0:                               // %entry
	cmp     w2, #1                  // =1
	b.lt    .LBB0_3
	// %bb.1:                               // %for.body.preheader
	mov     x8, xzr
	sxtw    x9, w2
	add     x10, x1, #4             // =4
	add     x11, x0, #16            // =16
.LBB0_2:                                // %for.body
	add     x13, x11, x8
	ldp     w14, w12, [x13, #-16]
	ldp     w16, w15, [x13]
	add     x8, x8, #32             // =32
	cmp     x8, x9
	and     w17, w14, #0x1
	ubfx    w0, w14, #8, #1
	and     w18, w16, #0x1
	ubfx    w3, w16, #8, #1
	ubfx    w1, w14, #16, #1
	bfi     w18, w3, #1, #1
	bfi     w17, w0, #1, #1
	ubfx    w0, w16, #16, #1
	ubfx    w3, w14, #9, #1
	bfi     w18, w0, #2, #1
	bfi     w17, w1, #2, #1
	ubfx    w1, w16, #24, #1
	bfi     w18, w1, #3, #1
	ubfx    w1, w16, #9, #1
        lsl     w3, w3, #1
	ubfx    w2, w14, #24, #1
	ubfx    w0, w14, #17, #1
	lsl     w1, w1, #1
	bfxil   w3, w14, #1, #1
	bfi     w17, w2, #3, #1
	ubfx    w2, w16, #17, #1
	bfxil   w1, w16, #1, #1
	bfi     w3, w0, #2, #1
	and     w0, w15, #0x1
	ubfx    w14, w14, #25, #1
	bfi     w1, w2, #2, #1
	and     w2, w12, #0x1
	bfi     w18, w0, #4, #1
	bfi     w3, w14, #3, #1
	ubfx    w14, w15, #8, #1
	bfi     w17, w2, #4, #1
	ubfx    w16, w16, #25, #1
	orr     w14, w18, w14, lsl #5
	ubfx    w18, w12, #8, #1
	bfi     w1, w16, #3, #1
        orr     w17, w17, w18, lsl #5
	ubfx    w18, w15, #1, #1
	bfi     w1, w18, #4, #1
	ubfx    w18, w12, #1, #1
	bfi     w3, w18, #4, #1
	ubfx    w18, w15, #16, #1
	orr     w14, w14, w18, lsl #6
	ubfx    w18, w12, #16, #1
	orr     w17, w17, w18, lsl #6
	ubfx    w18, w15, #9, #1
	orr     w18, w1, w18, lsl #5
	ubfx    w1, w12, #9, #1
	ldp     w0, w2, [x13, #-8]
	ldp     w16, w13, [x13, #8]
	orr     w1, w3, w1, lsl #5
	ubfx    w3, w15, #24, #1
	orr     w14, w14, w3, lsl #7
	ubfx    w3, w12, #24, #1
	orr     w17, w17, w3, lsl #7
	ubfx    w3, w15, #17, #1
	orr     w18, w18, w3, lsl #6
	ubfx    w3, w12, #17, #1
	orr     w1, w1, w3, lsl #6
	and     w3, w16, #0x1
	ubfx    w15, w15, #25, #1
	orr     w14, w14, w3, lsl #8
	and     w3, w0, #0x1
	orr     w15, w18, w15, lsl #7
	ubfx    w18, w16, #8, #1
	orr     w17, w17, w3, lsl #8
        orr     w14, w14, w18, lsl #9
	ubfx    w18, w0, #8, #1
	ubfx    w12, w12, #25, #1
	orr     w17, w17, w18, lsl #9
	ubfx    w18, w16, #1, #1
	orr     w12, w1, w12, lsl #7
	orr     w15, w15, w18, lsl #8
	ubfx    w18, w0, #1, #1
	orr     w12, w12, w18, lsl #8
	ubfx    w18, w16, #16, #1
	orr     w14, w14, w18, lsl #10
	ubfx    w18, w0, #16, #1
	orr     w17, w17, w18, lsl #10
	ubfx    w18, w16, #9, #1
	orr     w15, w15, w18, lsl #9
	ubfx    w18, w0, #9, #1
	orr     w12, w12, w18, lsl #9
	ubfx    w18, w16, #24, #1
	orr     w14, w14, w18, lsl #11
	ubfx    w18, w0, #24, #1
	orr     w17, w17, w18, lsl #11
	ubfx    w18, w16, #17, #1
	orr     w15, w15, w18, lsl #10
	ubfx    w18, w0, #17, #1
	ubfx    w16, w16, #25, #1
	orr     w12, w12, w18, lsl #10
	and     w18, w13, #0x1
	orr     w15, w15, w16, lsl #11
	ubfx    w16, w0, #25, #1
	orr     w14, w14, w18, lsl #12
	and     w18, w2, #0x1
	orr     w12, w12, w16, lsl #11
	ubfx    w16, w13, #8, #1
	orr     w17, w17, w18, lsl #12
        orr     w14, w14, w16, lsl #13
	ubfx    w16, w2, #8, #1
	orr     w16, w17, w16, lsl #13
	ubfx    w17, w13, #1, #1
	orr     w15, w15, w17, lsl #12
	ubfx    w17, w2, #1, #1
	orr     w12, w12, w17, lsl #12
	ubfx    w17, w13, #16, #1
	orr     w14, w14, w17, lsl #14
	ubfx    w17, w2, #16, #1
	orr     w16, w16, w17, lsl #14
	ubfx    w17, w13, #9, #1
	orr     w15, w15, w17, lsl #13
	ubfx    w17, w2, #9, #1
	orr     w12, w12, w17, lsl #13
	ubfx    w17, w13, #24, #1
	orr     w14, w14, w17, lsl #15
	ubfx    w17, w2, #24, #1
	orr     w16, w16, w17, lsl #15
	ubfx    w17, w13, #17, #1
	orr     w15, w15, w17, lsl #14
	ubfx    w17, w2, #17, #1
	orr     w12, w12, w17, lsl #14
	sturh   w14, [x10, #-2]
	ubfx    w13, w13, #25, #1
	ubfx    w14, w2, #25, #1
	orr     w13, w15, w13, lsl #15
	orr     w12, w12, w14, lsl #15
	sturh   w16, [x10, #-4]
	strh    w13, [x10, #2]
	strh    w12, [x10], #8
	b.lt    .LBB0_2
.LBB0_3:                                // %for.cond.cleanup
	ret
.Lfunc_end0:
	.size   pack2bits_v8, .Lfunc_end0-pack2bits_v8

	.section        ".note.GNU-stack","",@progbits
