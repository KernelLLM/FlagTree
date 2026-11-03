"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""

import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice


@triton.jit
def _hc_head_fused_kernel(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    T,
    H: tl.constexpr,
    rms_eps,
    hc_eps,
    residual_stride_t,
    fn_stride_m,
    out_stride_t,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    if pid_t >= T:
        return
    x_base = pid_t * residual_stride_t
    sqr_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc0 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc1 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc2 = tl.zeros([BLOCK_H], dtype=tl.float32)
    mix_acc3 = tl.zeros([BLOCK_H], dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        r0 = tl.load(residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        r1 = tl.load(residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        sqr_acc += r0 * r0 + r1 * r1
        fn00 = tl.load(fn_ptr + 0 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
        fn01 = tl.load(fn_ptr + 0 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
        mix_acc0 += r0 * fn00 + r1 * fn01
        fn10 = tl.load(fn_ptr + 1 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0)
        fn11 = tl.load(fn_ptr + 1 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0)
        mix_acc1 += r0 * fn10 + r1 * fn11
        if HC > 2:
            r2 = tl.load(
                residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            r3 = tl.load(
                residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            sqr_acc += r2 * r2 + r3 * r3
            mix_acc0 += r2 * tl.load(
                fn_ptr + 0 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
            )
            mix_acc0 += r3 * tl.load(
                fn_ptr + 0 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
            )
            mix_acc1 += r2 * tl.load(
                fn_ptr + 1 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
            )
            mix_acc1 += r3 * tl.load(
                fn_ptr + 1 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
            )
            fn20 = tl.load(
                fn_ptr + 2 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0
            )
            fn21 = tl.load(
                fn_ptr + 2 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0
            )
            fn22 = tl.load(
                fn_ptr + 2 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
            )
            fn23 = tl.load(
                fn_ptr + 2 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
            )
            mix_acc2 += r0 * fn20 + r1 * fn21 + r2 * fn22 + r3 * fn23
            fn30 = tl.load(
                fn_ptr + 3 * fn_stride_m + 0 * H + h_off, mask=h_mask, other=0.0
            )
            fn31 = tl.load(
                fn_ptr + 3 * fn_stride_m + 1 * H + h_off, mask=h_mask, other=0.0
            )
            fn32 = tl.load(
                fn_ptr + 3 * fn_stride_m + 2 * H + h_off, mask=h_mask, other=0.0
            )
            fn33 = tl.load(
                fn_ptr + 3 * fn_stride_m + 3 * H + h_off, mask=h_mask, other=0.0
            )
            mix_acc3 += r0 * fn30 + r1 * fn31 + r2 * fn32 + r3 * fn33
    K = HC * H
    sqr_total = tl.sum(sqr_acc)
    rsqrt_val = tl.math.rsqrt(sqr_total / K + rms_eps)
    hc_scale = tl.load(hc_scale_ptr)
    mix0 = tl.sum(mix_acc0)
    mix1 = tl.sum(mix_acc1)
    hc_base0 = tl.load(hc_base_ptr + 0)
    hc_base1 = tl.load(hc_base_ptr + 1)
    pre_mix0 = tl.sigmoid(mix0 * rsqrt_val * hc_scale + hc_base0) + hc_eps
    pre_mix1 = tl.sigmoid(mix1 * rsqrt_val * hc_scale + hc_base1) + hc_eps
    if HC > 2:
        mix2 = tl.sum(mix_acc2)
        mix3 = tl.sum(mix_acc3)
        hc_base2 = tl.load(hc_base_ptr + 2)
        hc_base3 = tl.load(hc_base_ptr + 3)
        pre_mix2 = tl.sigmoid(mix2 * rsqrt_val * hc_scale + hc_base2) + hc_eps
        pre_mix3 = tl.sigmoid(mix3 * rsqrt_val * hc_scale + hc_base3) + hc_eps
    out_base = pid_t * out_stride_t
    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + tl.arange(0, BLOCK_H)
        h_mask = h_off < H
        r0 = tl.load(residual_ptr + x_base + 0 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        r1 = tl.load(residual_ptr + x_base + 1 * H + h_off, mask=h_mask, other=0.0).to(
            tl.float32
        )
        acc = pre_mix0 * r0 + pre_mix1 * r1
        if HC > 2:
            r2 = tl.load(
                residual_ptr + x_base + 2 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            r3 = tl.load(
                residual_ptr + x_base + 3 * H + h_off, mask=h_mask, other=0.0
            ).to(tl.float32)
            acc += pre_mix2 * r2 + pre_mix3 * r3
        tl.store(out_ptr + out_base + h_off, acc.to(tl.bfloat16), mask=h_mask)
