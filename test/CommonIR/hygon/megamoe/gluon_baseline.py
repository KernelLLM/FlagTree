from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def _ue8m0_to_f32(x):
    return gl.exp2(x.to(gl.float32) - 127.0)


@g.jit
def _decode_e2m1(code):
    idx = code & 7
    mag = gl.where(
        idx == 0,
        0.0,
        gl.where(
            idx == 1,
            0.5,
            gl.where(
                idx == 2,
                1.0,
                gl.where(
                    idx == 3,
                    1.5,
                    gl.where(
                        idx == 4,
                        2.0,
                        gl.where(idx == 5, 3.0, gl.where(idx == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return gl.where(code & 8 != 0, -mag, mag)


@g.jit
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
    N: gl.constexpr,
    K: gl.constexpr,
    SCALE_IS_UE8M0: gl.constexpr,
    MATRIX_LAYOUT: gl.constexpr,
    REDUCTION_FIRST: gl.constexpr,
):
    packed_offsets = (
        expert * stride_e
        + (n_offsets[None, :] if REDUCTION_FIRST else n_offsets[:, None]) * stride_n
        + (k_offsets[:, None] if REDUCTION_FIRST else k_offsets[None, :])
        // 2
        * stride_kp
    )
    packed = gl.load(
        packed_ptr + packed_offsets,
        mask=((n_offsets[None, :] if REDUCTION_FIRST else n_offsets[:, None]) < N)
        & ((k_offsets[:, None] if REDUCTION_FIRST else k_offsets[None, :]) < K),
        other=0,
    ).to(gl.uint8)
    low = packed & 15
    high = packed >> 4 & 15
    code = gl.where(
        (k_offsets[:, None] if REDUCTION_FIRST else k_offsets[None, :]) & 1 == 0,
        low,
        high,
    )
    values = _decode_e2m1(code)
    scale_offsets = (
        expert * scale_stride_e
        + (n_offsets[None, :] if REDUCTION_FIRST else n_offsets[:, None])
        * scale_stride_n
        + (k_offsets[:, None] if REDUCTION_FIRST else k_offsets[None, :])
        // 32
        * scale_stride_g
    )
    raw_scale = gl.load(
        scale_ptr + scale_offsets,
        mask=((n_offsets[None, :] if REDUCTION_FIRST else n_offsets[:, None]) < N)
        & ((k_offsets[:, None] if REDUCTION_FIRST else k_offsets[None, :]) < K),
        other=0,
    )
    scale = _ue8m0_to_f32(raw_scale) if SCALE_IS_UE8M0 else raw_scale.to(gl.float32)
    return values * scale


@g.jit
def _fp8_fp4_mega_moe_l1_kernel(
    x_ptr,
    x_scale_ptr,
    topk_idx_ptr,
    l1_w_ptr,
    l1_s_ptr,
    l1_out_ptr,
    M: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    TOP_K: gl.constexpr,
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
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SCALE_IS_UE8M0: gl.constexpr,
    MATRIX_LAYOUT: gl.constexpr,
    REDUCTION_FIRST: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_topk = gl.program_id(1)
    pid_n = gl.program_id(2)
    n_offsets = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0 if REDUCTION_FIRST else 1, MATRIX_LAYOUT)
    )
    k_offsets = gl.arange(
        0, BLOCK_K, layout=gl.SliceLayout(1 if REDUCTION_FIRST else 0, MATRIX_LAYOUT)
    )
    expert = gl.load(topk_idx_ptr + pid_m * stride_topkm + pid_topk * stride_topkk)
    valid_expert = expert >= 0
    acc = gl.zeros(
        [BLOCK_N],
        dtype=gl.float32,
        layout=gl.SliceLayout(0 if REDUCTION_FIRST else 1, MATRIX_LAYOUT),
    )
    for k0 in range(0, H, BLOCK_K):
        ks = k0 + k_offsets
        x = gl.load(
            x_ptr + pid_m * stride_xm + ks * stride_xh, mask=ks < H, other=0.0
        ).to(gl.float32)
        x_scale = gl.load(
            x_scale_ptr + pid_m * stride_xsm + ks // 32 * stride_xsg,
            mask=ks < H,
            other=0.0,
        ).to(gl.float32)
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
            MATRIX_LAYOUT,
            REDUCTION_FIRST,
        )
        acc += gl.sum(
            w * (x[:, None] if REDUCTION_FIRST else x[None, :]),
            axis=0 if REDUCTION_FIRST else 1,
        )
    STORE_LAYOUT: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    n_store = gl.convert_layout(n_offsets, STORE_LAYOUT)
    acc_store = gl.convert_layout(acc, STORE_LAYOUT)
    out_ptrs = (
        l1_out_ptr
        + pid_m * stride_out_m
        + pid_topk * stride_out_k
        + n_store * stride_out_n
    )
    gl.store(out_ptrs, acc_store, mask=valid_expert & (n_store < 2 * I))


@g.jit
def _fp8_fp4_mega_moe_l2_kernel(
    topk_idx_ptr,
    topk_weights_ptr,
    l1_out_ptr,
    l2_w_ptr,
    l2_s_ptr,
    y_ptr,
    M: gl.constexpr,
    H: gl.constexpr,
    I: gl.constexpr,
    TOP_K: gl.constexpr,
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
    BLOCK_H: gl.constexpr,
    BLOCK_I: gl.constexpr,
    SCALE_IS_UE8M0: gl.constexpr,
    ACTIVATION_CLAMP: gl.constexpr,
    MATRIX_LAYOUT: gl.constexpr,
    REDUCTION_FIRST: gl.constexpr,
):
    pid_m = gl.program_id(0)
    pid_h = gl.program_id(1)
    h_offsets = pid_h * BLOCK_H + gl.arange(
        0, BLOCK_H, layout=gl.SliceLayout(0 if REDUCTION_FIRST else 1, MATRIX_LAYOUT)
    )
    i_offsets = gl.arange(
        0, BLOCK_I, layout=gl.SliceLayout(1 if REDUCTION_FIRST else 0, MATRIX_LAYOUT)
    )
    acc = gl.zeros(
        [BLOCK_H],
        dtype=gl.float32,
        layout=gl.SliceLayout(0 if REDUCTION_FIRST else 1, MATRIX_LAYOUT),
    )
    for tk in range(0, TOP_K):
        expert = gl.load(topk_idx_ptr + pid_m * stride_topkm + tk * stride_topkk)
        valid_expert = expert >= 0
        route_w = gl.load(
            topk_weights_ptr + pid_m * stride_twm + tk * stride_twk,
            mask=valid_expert,
            other=0.0,
        )
        expert_acc = gl.zeros(
            [BLOCK_H],
            dtype=gl.float32,
            layout=gl.SliceLayout(0 if REDUCTION_FIRST else 1, MATRIX_LAYOUT),
        )
        for i0 in range(0, I, BLOCK_I):
            is_ = i0 + i_offsets
            gate = gl.load(
                l1_out_ptr + pid_m * stride_l1_m + tk * stride_l1_k + is_ * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(gl.float32)
            up = gl.load(
                l1_out_ptr
                + pid_m * stride_l1_m
                + tk * stride_l1_k
                + (I + is_) * stride_l1_n,
                mask=(is_ < I) & valid_expert,
                other=0.0,
            ).to(gl.float32)
            if ACTIVATION_CLAMP >= 0.0:
                gate = gl.minimum(gl.maximum(gate, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
                up = gl.minimum(gl.maximum(up, -ACTIVATION_CLAMP), ACTIVATION_CLAMP)
            act = gate / (1.0 + gl.exp(-gate)) * up
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
                MATRIX_LAYOUT,
                REDUCTION_FIRST,
            )
            expert_acc += gl.sum(
                w * (act[:, None] if REDUCTION_FIRST else act[None, :]),
                axis=0 if REDUCTION_FIRST else 1,
            )
        acc += expert_acc * route_w
    STORE_LAYOUT: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    h_store = gl.convert_layout(h_offsets, STORE_LAYOUT)
    acc_store = gl.convert_layout(acc, STORE_LAYOUT)
    y_ptrs = y_ptr + pid_m * stride_ym + h_store * stride_yh
    gl.store(y_ptrs, acc_store, mask=h_store < H)
