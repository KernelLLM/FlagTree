from triton.experimental.gluon.language.amd import cdna3
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _sum_in_layout(
    values, BLOCK_H: gl.constexpr, LAYOUT: gl.constexpr, REDUCE_LAYOUT: gl.constexpr
):
    S: gl.constexpr = LAYOUT.size_per_thread[0]
    if S > 1 and REDUCE_LAYOUT.size_per_thread[0] == 1:
        gl.static_assert(
            S * LAYOUT.threads_per_warp[0] * LAYOUT.warps_per_cta[0] == BLOCK_H
        )
        values = values.reshape([BLOCK_H // S, S]).trans().reshape([BLOCK_H])
    values = gl.convert_layout(values, REDUCE_LAYOUT, assert_trivial=True)
    return gl.sum(values, axis=0)


@g.jit
def hc_head(
    residual_ptr,
    fn_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    out_ptr,
    T,
    H: gl.constexpr,
    rms_eps,
    hc_eps,
    residual_stride_t,
    fn_stride_m,
    out_stride_t,
    HC: gl.constexpr,
    BLOCK_H: gl.constexpr,
    LAYOUT: gl.constexpr,
    REDUCE_LAYOUT: gl.constexpr,
):
    pid_t = gl.program_id(0)
    if pid_t >= T:
        return
    x_base = pid_t * residual_stride_t
    sqr_acc = gl.zeros([BLOCK_H], dtype=gl.float32, layout=LAYOUT)
    mix_acc0 = gl.zeros([BLOCK_H], dtype=gl.float32, layout=LAYOUT)
    mix_acc1 = gl.zeros([BLOCK_H], dtype=gl.float32, layout=LAYOUT)
    mix_acc2 = gl.zeros([BLOCK_H], dtype=gl.float32, layout=LAYOUT)
    mix_acc3 = gl.zeros([BLOCK_H], dtype=gl.float32, layout=LAYOUT)
    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + gl.arange(0, BLOCK_H, layout=LAYOUT)
        h_mask = h_off < H
        r0 = cdna3.buffer_load(
            residual_ptr + x_base + 0 * H, h_off, mask=h_mask, other=0.0
        ).to(gl.float32)
        r1 = cdna3.buffer_load(
            residual_ptr + x_base + 1 * H, h_off, mask=h_mask, other=0.0
        ).to(gl.float32)
        sqr_acc += r0 * r0 + r1 * r1
        fn00 = cdna3.buffer_load(
            fn_ptr + 0 * fn_stride_m + 0 * H, h_off, mask=h_mask, other=0.0
        )
        fn01 = cdna3.buffer_load(
            fn_ptr + 0 * fn_stride_m + 1 * H, h_off, mask=h_mask, other=0.0
        )
        mix_acc0 += r0 * fn00 + r1 * fn01
        fn10 = cdna3.buffer_load(
            fn_ptr + 1 * fn_stride_m + 0 * H, h_off, mask=h_mask, other=0.0
        )
        fn11 = cdna3.buffer_load(
            fn_ptr + 1 * fn_stride_m + 1 * H, h_off, mask=h_mask, other=0.0
        )
        mix_acc1 += r0 * fn10 + r1 * fn11
        if HC > 2:
            r2 = cdna3.buffer_load(
                residual_ptr + x_base + 2 * H, h_off, mask=h_mask, other=0.0
            ).to(gl.float32)
            r3 = cdna3.buffer_load(
                residual_ptr + x_base + 3 * H, h_off, mask=h_mask, other=0.0
            ).to(gl.float32)
            sqr_acc += r2 * r2 + r3 * r3
            mix_acc0 += r2 * cdna3.buffer_load(
                fn_ptr + 0 * fn_stride_m + 2 * H, h_off, mask=h_mask, other=0.0
            )
            mix_acc0 += r3 * cdna3.buffer_load(
                fn_ptr + 0 * fn_stride_m + 3 * H, h_off, mask=h_mask, other=0.0
            )
            mix_acc1 += r2 * cdna3.buffer_load(
                fn_ptr + 1 * fn_stride_m + 2 * H, h_off, mask=h_mask, other=0.0
            )
            mix_acc1 += r3 * cdna3.buffer_load(
                fn_ptr + 1 * fn_stride_m + 3 * H, h_off, mask=h_mask, other=0.0
            )
            fn20 = cdna3.buffer_load(
                fn_ptr + 2 * fn_stride_m + 0 * H, h_off, mask=h_mask, other=0.0
            )
            fn21 = cdna3.buffer_load(
                fn_ptr + 2 * fn_stride_m + 1 * H, h_off, mask=h_mask, other=0.0
            )
            fn22 = cdna3.buffer_load(
                fn_ptr + 2 * fn_stride_m + 2 * H, h_off, mask=h_mask, other=0.0
            )
            fn23 = cdna3.buffer_load(
                fn_ptr + 2 * fn_stride_m + 3 * H, h_off, mask=h_mask, other=0.0
            )
            mix_acc2 += r0 * fn20 + r1 * fn21 + r2 * fn22 + r3 * fn23
            fn30 = cdna3.buffer_load(
                fn_ptr + 3 * fn_stride_m + 0 * H, h_off, mask=h_mask, other=0.0
            )
            fn31 = cdna3.buffer_load(
                fn_ptr + 3 * fn_stride_m + 1 * H, h_off, mask=h_mask, other=0.0
            )
            fn32 = cdna3.buffer_load(
                fn_ptr + 3 * fn_stride_m + 2 * H, h_off, mask=h_mask, other=0.0
            )
            fn33 = cdna3.buffer_load(
                fn_ptr + 3 * fn_stride_m + 3 * H, h_off, mask=h_mask, other=0.0
            )
            mix_acc3 += r0 * fn30 + r1 * fn31 + r2 * fn32 + r3 * fn33
    K = HC * H
    sqr_total = _sum_in_layout(sqr_acc, BLOCK_H, LAYOUT, REDUCE_LAYOUT)
    rsqrt_val = gl.rsqrt(sqr_total / K + rms_eps)
    hc_scale = gl.load(hc_scale_ptr)
    mix0 = _sum_in_layout(mix_acc0, BLOCK_H, LAYOUT, REDUCE_LAYOUT)
    mix1 = _sum_in_layout(mix_acc1, BLOCK_H, LAYOUT, REDUCE_LAYOUT)
    hc_base0 = gl.load(hc_base_ptr + 0)
    hc_base1 = gl.load(hc_base_ptr + 1)
    pre_mix0 = 1 / (1 + gl.exp(-(mix0 * rsqrt_val * hc_scale + hc_base0))) + hc_eps
    pre_mix1 = 1 / (1 + gl.exp(-(mix1 * rsqrt_val * hc_scale + hc_base1))) + hc_eps
    if HC > 2:
        mix2 = _sum_in_layout(mix_acc2, BLOCK_H, LAYOUT, REDUCE_LAYOUT)
        mix3 = _sum_in_layout(mix_acc3, BLOCK_H, LAYOUT, REDUCE_LAYOUT)
        hc_base2 = gl.load(hc_base_ptr + 2)
        hc_base3 = gl.load(hc_base_ptr + 3)
        pre_mix2 = 1 / (1 + gl.exp(-(mix2 * rsqrt_val * hc_scale + hc_base2))) + hc_eps
        pre_mix3 = 1 / (1 + gl.exp(-(mix3 * rsqrt_val * hc_scale + hc_base3))) + hc_eps
    out_base = pid_t * out_stride_t
    for h_start in range(0, H, BLOCK_H):
        h_off = h_start + gl.arange(0, BLOCK_H, layout=LAYOUT)
        h_mask = h_off < H
        r0 = cdna3.buffer_load(
            residual_ptr + x_base + 0 * H, h_off, mask=h_mask, other=0.0
        ).to(gl.float32)
        r1 = cdna3.buffer_load(
            residual_ptr + x_base + 1 * H, h_off, mask=h_mask, other=0.0
        ).to(gl.float32)
        acc = pre_mix0 * r0 + pre_mix1 * r1
        if HC > 2:
            r2 = cdna3.buffer_load(
                residual_ptr + x_base + 2 * H, h_off, mask=h_mask, other=0.0
            ).to(gl.float32)
            r3 = cdna3.buffer_load(
                residual_ptr + x_base + 3 * H, h_off, mask=h_mask, other=0.0
            ).to(gl.float32)
            acc += pre_mix2 * r2 + pre_mix3 * r3
        cdna3.buffer_store(acc.to(gl.bfloat16), out_ptr + out_base, h_off, mask=h_mask)
