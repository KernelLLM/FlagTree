from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna3


@g.jit
def _store_tile(
    acc,
    weights,
    bos_vec,
    eos_vec,
    logits,
    bos,
    eos,
    ck,
    stride_lm,
    stride_ln,
    i_sh,
    Q: gl.constexpr,
    H: gl.constexpr,
    BQ: gl.constexpr,
    BK: gl.constexpr,
    MMA_LAYOUT: gl.constexpr,
    REDUCE_LAYOUT: gl.constexpr,
    STORE_LAYOUT: gl.constexpr,
    USE_LDS_RING: gl.constexpr,
):
    rounded = acc.to(gl.float16).to(gl.float32)
    score = gl.maximum(rounded, 0.0) * weights[:, None].to(gl.float32)
    if not USE_LDS_RING:
        score = gl.convert_layout(
            score, gl.BlockedLayout([1, 1], [16, 4], [1, 4], [0, 1])
        )
    reduced_input = score.trans().reshape(BK, BQ, H)
    reduced_input = gl.convert_layout(reduced_input, REDUCE_LAYOUT)
    reduced = gl.sum(reduced_input, axis=2)
    if not USE_LDS_RING:
        reduced = gl.convert_layout(
            reduced,
            gl.DistributedLinearLayout(
                [[16, 0], [32, 0]],
                [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0], [0, 0]],
                [[0, 0], [0, 0]],
                [],
                [64, 1],
            ),
        )
    output = reduced.trans()
    rows = i_sh * BQ + gl.arange(0, BQ, layout=gl.SliceLayout(1, STORE_LAYOUT))
    columns = ck * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, STORE_LAYOUT))
    begin = gl.convert_layout(bos_vec, gl.SliceLayout(1, STORE_LAYOUT))
    end = gl.convert_layout(eos_vec, gl.SliceLayout(1, STORE_LAYOUT))
    mask = (
        (rows[:, None] < Q)
        & (bos + columns[None, :] < eos)
        & (begin[:, None] <= bos + columns[None, :])
        & (end[:, None] > bos + columns[None, :])
    )
    if USE_LDS_RING:
        value = gl.convert_layout(output.to(gl.float16), STORE_LAYOUT).to(gl.float32)
    else:
        value = gl.convert_layout(output, STORE_LAYOUT).to(gl.float16).to(gl.float32)
    offsets = rows[:, None] * stride_lm + (bos + columns[None, :]) * stride_ln
    cdna3.buffer_store(value, logits, offsets, mask=mask)


@g.jit
def _load_k(
    k_index,
    bos,
    eos,
    ck,
    stride_kn,
    stride_kd,
    D: gl.constexpr,
    BK: gl.constexpr,
    K_LAYOUT: gl.constexpr,
    CONTROL_LAYOUT: gl.constexpr,
    Q_LAYOUT: gl.constexpr,
    USE_LDS_RING: gl.constexpr,
    valid=True,
):
    if USE_LDS_RING:
        d = gl.arange(0, D, layout=gl.SliceLayout(1, K_LAYOUT))
        dmask = d < D
        columns = ck * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, K_LAYOUT))
    else:
        d_q = gl.convert_layout(
            gl.arange(0, D, layout=CONTROL_LAYOUT), gl.SliceLayout(0, Q_LAYOUT)
        )
        d = gl.convert_layout(d_q, gl.SliceLayout(1, K_LAYOUT))
        dmask = gl.convert_layout(d_q < D, gl.SliceLayout(1, K_LAYOUT))
        columns_io = gl.convert_layout(
            ck * BK + gl.arange(0, BK, layout=CONTROL_LAYOUT),
            gl.SliceLayout(0, gl.BlockedLayout([1, 1], [1, 64], [1, 4], [1, 0])),
        )
        columns = gl.convert_layout(columns_io, gl.SliceLayout(0, K_LAYOUT))
    offsets = d[:, None] * stride_kd + (bos + columns[None, :]) * stride_kn
    mask = dmask[:, None] & (bos + columns[None, :] < eos) & valid
    return cdna3.buffer_load(k_index, offsets, mask=mask, other=0.0)


@g.jit
def lightning(
    q_index,
    k_index,
    cu_bg_seqlens,
    cu_ed_seqlens,
    weights,
    logits,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_wh,
    stride_lm,
    stride_ln,
    Q: gl.constexpr,
    H: gl.constexpr,
    K: gl.constexpr,
    TK: gl.constexpr,
    D: gl.constexpr,
    CU: gl.constexpr,
    BQ: gl.constexpr,
    BK: gl.constexpr,
    CONTROL_LAYOUT: gl.constexpr,
    Q_LAYOUT: gl.constexpr,
    K_LAYOUT: gl.constexpr,
    WEIGHT_LAYOUT: gl.constexpr,
    MMA_LAYOUT: gl.constexpr,
    REDUCE_LAYOUT: gl.constexpr,
    STORE_LAYOUT: gl.constexpr,
    USE_LDS_RING: gl.constexpr,
):
    i_sh = gl.program_id(0)
    i_k = gl.program_id(1)
    control = i_sh * BQ + gl.arange(0, BQ, layout=CONTROL_LAYOUT)
    bos_vec = gl.load(cu_bg_seqlens + control, control < CU, 1000000000) + i_k * TK
    eos_vec = gl.load(cu_ed_seqlens + control, control < CU, -1000000000)
    eos_vec = gl.minimum(eos_vec, bos_vec + (i_k + 1) * TK)
    bos = gl.maximum(gl.min(bos_vec, axis=0), 0)
    eos = gl.minimum(gl.max(eos_vec, axis=0), K)
    length = eos - bos
    if length > 0:
        if USE_LDS_RING:
            rows = i_sh * (BQ * H) + gl.arange(
                0, BQ * H, layout=gl.SliceLayout(1, Q_LAYOUT)
            )
            d = gl.arange(0, D, layout=gl.SliceLayout(0, Q_LAYOUT))
        else:
            rows_control = i_sh * (BQ * H) + gl.arange(0, BQ * H, layout=CONTROL_LAYOUT)
            rows = gl.convert_layout(rows_control, gl.SliceLayout(1, Q_LAYOUT))
            d = gl.convert_layout(
                gl.arange(0, D, layout=CONTROL_LAYOUT), gl.SliceLayout(0, Q_LAYOUT)
            )
        q_offsets = rows[:, None] * stride_qh + d[None, :] * stride_qd
        q = cdna3.buffer_load(
            q_index,
            q_offsets,
            mask=(rows[:, None] < Q * H) & (d[None, :] < D),
            other=0.0,
        ).to(gl.float16)
        if USE_LDS_RING:
            wrows = i_sh * (BQ * H) + gl.arange(0, BQ * H, layout=WEIGHT_LAYOUT)
            wmask = wrows < Q * H
        else:
            wrows = gl.convert_layout(rows, CONTROL_LAYOUT)
            wmask = gl.convert_layout(rows < Q * H, CONTROL_LAYOUT)
            wrows = gl.convert_layout(wrows, WEIGHT_LAYOUT)
            wmask = gl.convert_layout(wmask, WEIGHT_LAYOUT)
        w = cdna3.buffer_load(weights, wrows * stride_wh, mask=wmask, other=0.0).to(
            gl.float16
        )
        w = gl.convert_layout(w, gl.SliceLayout(1, MMA_LAYOUT))
        count = gl.cdiv(length, BK)
        if USE_LDS_RING:
            q_shared = gl.allocate_shared_memory(
                gl.float16, [BQ * H, D], gl.SwizzledSharedLayout(4, 2, 8, [1, 0]), q
            )
            a = q_shared.load(gl.DotOperandLayout(0, MMA_LAYOUT, 4))
            ring = gl.allocate_shared_memory(
                gl.bfloat16, [2, D, BK], gl.SwizzledSharedLayout(4, 2, 8, [0, 1])
            )
            ring.index(0).store(
                _load_k(
                    k_index,
                    bos,
                    eos,
                    0,
                    stride_kn,
                    stride_kd,
                    D,
                    BK,
                    K_LAYOUT,
                    CONTROL_LAYOUT,
                    Q_LAYOUT,
                    USE_LDS_RING,
                )
            )
            ring.index(1).store(
                _load_k(
                    k_index,
                    bos,
                    eos,
                    1,
                    stride_kn,
                    stride_kd,
                    D,
                    BK,
                    K_LAYOUT,
                    CONTROL_LAYOUT,
                    Q_LAYOUT,
                    USE_LDS_RING,
                    count > 1,
                )
            )
            for ck in range(0, gl.maximum(count - 2, 0)):
                future = _load_k(
                    k_index,
                    bos,
                    eos,
                    ck + 2,
                    stride_kn,
                    stride_kd,
                    D,
                    BK,
                    K_LAYOUT,
                    CONTROL_LAYOUT,
                    Q_LAYOUT,
                    USE_LDS_RING,
                )
                current = ring.index(ck % 2)
                b = current.load(gl.DotOperandLayout(1, MMA_LAYOUT, 4)).to(gl.float16)
                acc = cdna3.mfma(a, b, gl.full([BQ * H, BK], 0, gl.float32, MMA_LAYOUT))
                _store_tile(
                    acc,
                    w,
                    bos_vec,
                    eos_vec,
                    logits,
                    bos,
                    eos,
                    ck,
                    stride_lm,
                    stride_ln,
                    i_sh,
                    Q,
                    H,
                    BQ,
                    BK,
                    MMA_LAYOUT,
                    REDUCE_LAYOUT,
                    STORE_LAYOUT,
                    USE_LDS_RING,
                )
                current.store(future)
            if count >= 2:
                ck = count - 2
                b = (
                    ring.index(ck % 2)
                    .load(gl.DotOperandLayout(1, MMA_LAYOUT, 4))
                    .to(gl.float16)
                )
                acc = cdna3.mfma(a, b, gl.full([BQ * H, BK], 0, gl.float32, MMA_LAYOUT))
                _store_tile(
                    acc,
                    w,
                    bos_vec,
                    eos_vec,
                    logits,
                    bos,
                    eos,
                    ck,
                    stride_lm,
                    stride_ln,
                    i_sh,
                    Q,
                    H,
                    BQ,
                    BK,
                    MMA_LAYOUT,
                    REDUCE_LAYOUT,
                    STORE_LAYOUT,
                    USE_LDS_RING,
                )
            ck = count - 1
            b = (
                ring.index(ck % 2)
                .load(gl.DotOperandLayout(1, MMA_LAYOUT, 4))
                .to(gl.float16)
            )
            acc = cdna3.mfma(a, b, gl.full([BQ * H, BK], 0, gl.float32, MMA_LAYOUT))
            _store_tile(
                acc,
                w,
                bos_vec,
                eos_vec,
                logits,
                bos,
                eos,
                ck,
                stride_lm,
                stride_ln,
                i_sh,
                Q,
                H,
                BQ,
                BK,
                MMA_LAYOUT,
                REDUCE_LAYOUT,
                STORE_LAYOUT,
                USE_LDS_RING,
            )
        else:
            a = gl.convert_layout(q, gl.DotOperandLayout(0, MMA_LAYOUT, 4))
            for ck in range(count):
                k = _load_k(
                    k_index,
                    bos,
                    eos,
                    ck,
                    stride_kn,
                    stride_kd,
                    D,
                    BK,
                    K_LAYOUT,
                    CONTROL_LAYOUT,
                    Q_LAYOUT,
                    USE_LDS_RING,
                ).to(gl.float16)
                b = gl.convert_layout(k, gl.DotOperandLayout(1, MMA_LAYOUT, 4))
                acc = cdna3.mfma(a, b, gl.full([BQ * H, BK], 0, gl.float32, MMA_LAYOUT))
                _store_tile(
                    acc,
                    w,
                    bos_vec,
                    eos_vec,
                    logits,
                    bos,
                    eos,
                    ck,
                    stride_lm,
                    stride_ln,
                    i_sh,
                    Q,
                    H,
                    BQ,
                    BK,
                    MMA_LAYOUT,
                    REDUCE_LAYOUT,
                    STORE_LAYOUT,
                    USE_LDS_RING,
                )
