"""Frozen TLE device functions extracted without changing bodies.
Unchanged original source and licensing are retained under tle/.
"""

import triton
import triton.language as tl
from triton.experimental import tle
from triton.language.extra import libdevice


@triton.jit
def _ue8m0_to_f32(x):
    return tl.exp2(x.to(tl.float32) - 127.0)


@triton.jit
def _decode_e2m1(code):
    idx = code & 7
    mag = tl.where(
        idx == 0,
        0.0,
        tl.where(
            idx == 1,
            0.5,
            tl.where(
                idx == 2,
                1.0,
                tl.where(
                    idx == 3,
                    1.5,
                    tl.where(
                        idx == 4,
                        2.0,
                        tl.where(idx == 5, 3.0, tl.where(idx == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where(code & 8 != 0, -mag, mag)


@triton.jit
def _load_fp4_weight(
    packed_ptr,
    scale_ptr,
    expert,
    n_offsets,
    k_offsets,
    stride_e,
    stride_n,
    stride_kp,
    scale_stride_e,
    scale_stride_n,
    scale_stride_g,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
):
    packed_offsets = (
        expert * stride_e
        + n_offsets[:, None] * stride_n
        + k_offsets[None, :] // 2 * stride_kp
    )
    packed = tl.load(
        packed_ptr + packed_offsets,
        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
        other=0,
    ).to(tl.uint8)
    low = packed & 15
    high = packed >> 4 & 15
    code = tl.where(k_offsets[None, :] & 1 == 0, low, high)
    values = _decode_e2m1(code)
    scale_offsets = (
        expert * scale_stride_e
        + n_offsets[:, None] * scale_stride_n
        + k_offsets[None, :] // 32 * scale_stride_g
    )
    raw_scale = tl.load(
        scale_ptr + scale_offsets,
        mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
        other=0,
    )
    scale = _ue8m0_to_f32(raw_scale) if SCALE_IS_UE8M0 else raw_scale.to(tl.float32)
    return values * scale


@triton.jit
def _fp8_fp4_mega_moe_l1_kernel(
    x_ptr,
    x_scale_ptr,
    topk_idx_ptr,
    l1_w_ptr,
    l1_s_ptr,
    l1_out_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_xm,
    stride_xh,
    stride_xsm,
    stride_xsg,
    stride_topkm,
    stride_topkk,
    stride_w_e,
    stride_w_n,
    stride_w_kp,
    stride_s_e,
    stride_s_n,
    stride_s_g,
    stride_out_m,
    stride_out_k,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_topk = tl.program_id(1)
    pid_n = tl.program_id(2)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    expert = tl.load(topk_idx_ptr + pid_m * stride_topkm + pid_topk * stride_topkk)
    valid_expert = expert >= 0
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        ks = k0 + k_offsets
        x = tl.load(
            x_ptr + pid_m * stride_xm + ks * stride_xh, mask=ks < H, other=0.0
        ).to(tl.float32)
        x_scale = tl.load(
            x_scale_ptr + pid_m * stride_xsm + ks // 32 * stride_xsg,
            mask=ks < H,
            other=0.0,
        ).to(tl.float32)
        x = x * x_scale
        w = _load_fp4_weight(
            l1_w_ptr,
            l1_s_ptr,
            expert,
            n_offsets,
            ks,
            stride_w_e,
            stride_w_n,
            stride_w_kp,
            stride_s_e,
            stride_s_n,
            stride_s_g,
            2 * I,
            H,
            SCALE_IS_UE8M0,
        )
        acc += tl.sum(w * x[None, :], axis=1)
    out_ptrs = (
        l1_out_ptr
        + pid_m * stride_out_m
        + pid_topk * stride_out_k
        + n_offsets * stride_out_n
    )
    tl.store(out_ptrs, acc, mask=valid_expert & (n_offsets < 2 * I))


@triton.jit
def _fp8_fp4_mega_moe_l2_kernel(
    topk_idx_ptr,
    topk_weights_ptr,
    l1_out_ptr,
    l2_w_ptr,
    l2_s_ptr,
    y_ptr,
    M: tl.constexpr,
    H: tl.constexpr,
    I: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_topkm,
    stride_topkk,
    stride_twm,
    stride_twk,
    stride_l1_m,
    stride_l1_k,
    stride_l1_n,
    stride_w_e,
    stride_w_h,
    stride_w_ip,
    stride_s_e,
    stride_s_h,
    stride_s_g,
    stride_ym,
    stride_yh,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
    SCALE_IS_UE8M0: tl.constexpr,
    ACTIVATION_CLAMP: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    i_offsets = tl.arange(0, BLOCK_I)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for tk in range(0, TOP_K):
        expert = tl.load(topk_idx_ptr + pid_m * stride_topkm + tk * stride_topkk)
        valid_expert = expert >= 0
        route_w = tl.load(
            topk_weights_ptr + pid_m * stride_twm + tk * stride_twk,
            mask=valid_expert,
            other=0.0,
        )
        expert_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            is_ = i0 + i_offsets
            gate = tl.load(
                l1_out_ptr + pid_m * stride_l1_m + tk * stride_l1_k + is_ * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                l1_out_ptr
                + pid_m * stride_l1_m
                + tk * stride_l1_k
                + (I + is_) * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(tl.float32)
            if ACTIVATION_CLAMP >= 0.0:
                gate = tl.minimum(tl.maximum(gate, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
                up = tl.minimum(tl.maximum(up, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
            act = gate / (1.0 + tl.exp(-gate)) * up
            w = _load_fp4_weight(
                l2_w_ptr,
                l2_s_ptr,
                expert,
                h_offsets,
                is_,
                stride_w_e,
                stride_w_h,
                stride_w_ip,
                stride_s_e,
                stride_s_h,
                stride_s_g,
                H,
                I,
                SCALE_IS_UE8M0,
            )
            expert_acc += tl.sum(w * act[None, :], axis=1)
        acc += expert_acc * route_w
    y_ptrs = y_ptr + pid_m * stride_ym + h_offsets * stride_yh
    tl.store(y_ptrs, acc, mask=h_offsets < H)
