from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def mhc_split_sinkhorn_kernel_hcmult_4(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    num_tokens,
    BLOCK_N: gl.constexpr,
    SINKHORN_ITERS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    pid = gl.program_id(0)
    offs = pid * BLOCK_N + gl.arange(0, BLOCK_N, layout=LAYOUT)
    mask = offs < num_tokens
    base = offs * 24
    scale_0 = gl.load(hc_scale_ptr + 0)
    scale_1 = gl.load(hc_scale_ptr + 1)
    scale_2 = gl.load(hc_scale_ptr + 2)
    m0 = gl.load(mixes_ptr + base + 0, mask=mask)
    m1 = gl.load(mixes_ptr + base + 1, mask=mask)
    m2 = gl.load(mixes_ptr + base + 2, mask=mask)
    m3 = gl.load(mixes_ptr + base + 3, mask=mask)
    m4 = gl.load(mixes_ptr + base + 4, mask=mask)
    m5 = gl.load(mixes_ptr + base + 5, mask=mask)
    m6 = gl.load(mixes_ptr + base + 6, mask=mask)
    m7 = gl.load(mixes_ptr + base + 7, mask=mask)
    b0 = gl.load(hc_base_ptr + 0)
    b1 = gl.load(hc_base_ptr + 1)
    b2 = gl.load(hc_base_ptr + 2)
    b3 = gl.load(hc_base_ptr + 3)
    b4 = gl.load(hc_base_ptr + 4)
    b5 = gl.load(hc_base_ptr + 5)
    b6 = gl.load(hc_base_ptr + 6)
    b7 = gl.load(hc_base_ptr + 7)
    pre_base = offs * 4
    gl.store(
        pre_ptr + pre_base + 0,
        1 / (1 + gl.exp(-(m0 * scale_0 + b0))) + 1e-06,
        mask=mask,
    )
    gl.store(
        pre_ptr + pre_base + 1,
        1 / (1 + gl.exp(-(m1 * scale_0 + b1))) + 1e-06,
        mask=mask,
    )
    gl.store(
        pre_ptr + pre_base + 2,
        1 / (1 + gl.exp(-(m2 * scale_0 + b2))) + 1e-06,
        mask=mask,
    )
    gl.store(
        pre_ptr + pre_base + 3,
        1 / (1 + gl.exp(-(m3 * scale_0 + b3))) + 1e-06,
        mask=mask,
    )
    post_base = offs * 4
    gl.store(
        post_ptr + post_base + 0,
        2.0 * (1 / (1 + gl.exp(-(m4 * scale_1 + b4)))),
        mask=mask,
    )
    gl.store(
        post_ptr + post_base + 1,
        2.0 * (1 / (1 + gl.exp(-(m5 * scale_1 + b5)))),
        mask=mask,
    )
    gl.store(
        post_ptr + post_base + 2,
        2.0 * (1 / (1 + gl.exp(-(m6 * scale_1 + b6)))),
        mask=mask,
    )
    gl.store(
        post_ptr + post_base + 3,
        2.0 * (1 / (1 + gl.exp(-(m7 * scale_1 + b7)))),
        mask=mask,
    )
    cb = 8
    b8 = gl.load(hc_base_ptr + cb + 0)
    b9 = gl.load(hc_base_ptr + cb + 1)
    b10 = gl.load(hc_base_ptr + cb + 2)
    b11 = gl.load(hc_base_ptr + cb + 3)
    b12 = gl.load(hc_base_ptr + cb + 4)
    b13 = gl.load(hc_base_ptr + cb + 5)
    b14 = gl.load(hc_base_ptr + cb + 6)
    b15 = gl.load(hc_base_ptr + cb + 7)
    b16 = gl.load(hc_base_ptr + cb + 8)
    b17 = gl.load(hc_base_ptr + cb + 9)
    b18 = gl.load(hc_base_ptr + cb + 10)
    b19 = gl.load(hc_base_ptr + cb + 11)
    b20 = gl.load(hc_base_ptr + cb + 12)
    b21 = gl.load(hc_base_ptr + cb + 13)
    b22 = gl.load(hc_base_ptr + cb + 14)
    b23 = gl.load(hc_base_ptr + cb + 15)
    cm_00 = gl.load(mixes_ptr + base + cb + 0, mask=mask) * scale_2 + b8
    cm_01 = gl.load(mixes_ptr + base + cb + 1, mask=mask) * scale_2 + b9
    cm_02 = gl.load(mixes_ptr + base + cb + 2, mask=mask) * scale_2 + b10
    cm_03 = gl.load(mixes_ptr + base + cb + 3, mask=mask) * scale_2 + b11
    cm_10 = gl.load(mixes_ptr + base + cb + 4, mask=mask) * scale_2 + b12
    cm_11 = gl.load(mixes_ptr + base + cb + 5, mask=mask) * scale_2 + b13
    cm_12 = gl.load(mixes_ptr + base + cb + 6, mask=mask) * scale_2 + b14
    cm_13 = gl.load(mixes_ptr + base + cb + 7, mask=mask) * scale_2 + b15
    cm_20 = gl.load(mixes_ptr + base + cb + 8, mask=mask) * scale_2 + b16
    cm_21 = gl.load(mixes_ptr + base + cb + 9, mask=mask) * scale_2 + b17
    cm_22 = gl.load(mixes_ptr + base + cb + 10, mask=mask) * scale_2 + b18
    cm_23 = gl.load(mixes_ptr + base + cb + 11, mask=mask) * scale_2 + b19
    cm_30 = gl.load(mixes_ptr + base + cb + 12, mask=mask) * scale_2 + b20
    cm_31 = gl.load(mixes_ptr + base + cb + 13, mask=mask) * scale_2 + b21
    cm_32 = gl.load(mixes_ptr + base + cb + 14, mask=mask) * scale_2 + b22
    cm_33 = gl.load(mixes_ptr + base + cb + 15, mask=mask) * scale_2 + b23
    rm = gl.maximum(gl.maximum(cm_00, cm_01), gl.maximum(cm_02, cm_03))
    cm_00 = gl.exp(cm_00 - rm)
    cm_01 = gl.exp(cm_01 - rm)
    cm_02 = gl.exp(cm_02 - rm)
    cm_03 = gl.exp(cm_03 - rm)
    inv_rs = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03)
    cm_00 = cm_00 * inv_rs + 1e-06
    cm_01 = cm_01 * inv_rs + 1e-06
    cm_02 = cm_02 * inv_rs + 1e-06
    cm_03 = cm_03 * inv_rs + 1e-06
    rm = gl.maximum(gl.maximum(cm_10, cm_11), gl.maximum(cm_12, cm_13))
    cm_10 = gl.exp(cm_10 - rm)
    cm_11 = gl.exp(cm_11 - rm)
    cm_12 = gl.exp(cm_12 - rm)
    cm_13 = gl.exp(cm_13 - rm)
    inv_rs = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13)
    cm_10 = cm_10 * inv_rs + 1e-06
    cm_11 = cm_11 * inv_rs + 1e-06
    cm_12 = cm_12 * inv_rs + 1e-06
    cm_13 = cm_13 * inv_rs + 1e-06
    rm = gl.maximum(gl.maximum(cm_20, cm_21), gl.maximum(cm_22, cm_23))
    cm_20 = gl.exp(cm_20 - rm)
    cm_21 = gl.exp(cm_21 - rm)
    cm_22 = gl.exp(cm_22 - rm)
    cm_23 = gl.exp(cm_23 - rm)
    inv_rs = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23)
    cm_20 = cm_20 * inv_rs + 1e-06
    cm_21 = cm_21 * inv_rs + 1e-06
    cm_22 = cm_22 * inv_rs + 1e-06
    cm_23 = cm_23 * inv_rs + 1e-06
    rm = gl.maximum(gl.maximum(cm_30, cm_31), gl.maximum(cm_32, cm_33))
    cm_30 = gl.exp(cm_30 - rm)
    cm_31 = gl.exp(cm_31 - rm)
    cm_32 = gl.exp(cm_32 - rm)
    cm_33 = gl.exp(cm_33 - rm)
    inv_rs = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33)
    cm_30 = cm_30 * inv_rs + 1e-06
    cm_31 = cm_31 * inv_rs + 1e-06
    cm_32 = cm_32 * inv_rs + 1e-06
    cm_33 = cm_33 * inv_rs + 1e-06
    inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + 1e-06)
    inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + 1e-06)
    inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + 1e-06)
    inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + 1e-06)
    cm_00 *= inv_cs0
    cm_10 *= inv_cs0
    cm_20 *= inv_cs0
    cm_30 *= inv_cs0
    cm_01 *= inv_cs1
    cm_11 *= inv_cs1
    cm_21 *= inv_cs1
    cm_31 *= inv_cs1
    cm_02 *= inv_cs2
    cm_12 *= inv_cs2
    cm_22 *= inv_cs2
    cm_32 *= inv_cs2
    cm_03 *= inv_cs3
    cm_13 *= inv_cs3
    cm_23 *= inv_cs3
    cm_33 *= inv_cs3
    for _ in range(SINKHORN_ITERS - 1):
        inv_rs0 = 1.0 / (cm_00 + cm_01 + cm_02 + cm_03 + 1e-06)
        inv_rs1 = 1.0 / (cm_10 + cm_11 + cm_12 + cm_13 + 1e-06)
        inv_rs2 = 1.0 / (cm_20 + cm_21 + cm_22 + cm_23 + 1e-06)
        inv_rs3 = 1.0 / (cm_30 + cm_31 + cm_32 + cm_33 + 1e-06)
        cm_00 *= inv_rs0
        cm_01 *= inv_rs0
        cm_02 *= inv_rs0
        cm_03 *= inv_rs0
        cm_10 *= inv_rs1
        cm_11 *= inv_rs1
        cm_12 *= inv_rs1
        cm_13 *= inv_rs1
        cm_20 *= inv_rs2
        cm_21 *= inv_rs2
        cm_22 *= inv_rs2
        cm_23 *= inv_rs2
        cm_30 *= inv_rs3
        cm_31 *= inv_rs3
        cm_32 *= inv_rs3
        cm_33 *= inv_rs3
        inv_cs0 = 1.0 / (cm_00 + cm_10 + cm_20 + cm_30 + 1e-06)
        inv_cs1 = 1.0 / (cm_01 + cm_11 + cm_21 + cm_31 + 1e-06)
        inv_cs2 = 1.0 / (cm_02 + cm_12 + cm_22 + cm_32 + 1e-06)
        inv_cs3 = 1.0 / (cm_03 + cm_13 + cm_23 + cm_33 + 1e-06)
        cm_00 *= inv_cs0
        cm_01 *= inv_cs1
        cm_02 *= inv_cs2
        cm_03 *= inv_cs3
        cm_10 *= inv_cs0
        cm_11 *= inv_cs1
        cm_12 *= inv_cs2
        cm_13 *= inv_cs3
        cm_20 *= inv_cs0
        cm_21 *= inv_cs1
        cm_22 *= inv_cs2
        cm_23 *= inv_cs3
        cm_30 *= inv_cs0
        cm_31 *= inv_cs1
        cm_32 *= inv_cs2
        cm_33 *= inv_cs3
    co = offs * 16
    gl.store(comb_ptr + co + 0, cm_00, mask=mask)
    gl.store(comb_ptr + co + 1, cm_01, mask=mask)
    gl.store(comb_ptr + co + 2, cm_02, mask=mask)
    gl.store(comb_ptr + co + 3, cm_03, mask=mask)
    gl.store(comb_ptr + co + 4, cm_10, mask=mask)
    gl.store(comb_ptr + co + 5, cm_11, mask=mask)
    gl.store(comb_ptr + co + 6, cm_12, mask=mask)
    gl.store(comb_ptr + co + 7, cm_13, mask=mask)
    gl.store(comb_ptr + co + 8, cm_20, mask=mask)
    gl.store(comb_ptr + co + 9, cm_21, mask=mask)
    gl.store(comb_ptr + co + 10, cm_22, mask=mask)
    gl.store(comb_ptr + co + 11, cm_23, mask=mask)
    gl.store(comb_ptr + co + 12, cm_30, mask=mask)
    gl.store(comb_ptr + co + 13, cm_31, mask=mask)
    gl.store(comb_ptr + co + 14, cm_32, mask=mask)
    gl.store(comb_ptr + co + 15, cm_33, mask=mask)


@g.jit
def mhc_split_sinkhorn_kernel_generic(
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    pre_ptr,
    post_ptr,
    comb_ptr,
    num_tokens,
    SINKHORN_ITERS: gl.constexpr,
    HC_MULT: gl.constexpr,
    MIX_HC: gl.constexpr,
):
    pid_n = gl.program_id(0)
    if pid_n >= num_tokens:
        return
    base = pid_n * MIX_HC
    pre_base = pid_n * HC_MULT
    post_base = pid_n * HC_MULT
    comb_base = pid_n * (HC_MULT * HC_MULT)
    scale_0 = gl.load(hc_scale_ptr + 0)
    scale_1 = gl.load(hc_scale_ptr + 1)
    scale_2 = gl.load(hc_scale_ptr + 2)
    for j in gl.static_range(HC_MULT):
        pre_idx = j
        post_idx = HC_MULT + j
        pre_m = gl.load(mixes_ptr + base + pre_idx)
        post_m = gl.load(mixes_ptr + base + post_idx)
        pre_b = gl.load(hc_base_ptr + pre_idx)
        post_b = gl.load(hc_base_ptr + post_idx)
        gl.store(
            pre_ptr + pre_base + j, 1 / (1 + gl.exp(-(pre_m * scale_0 + pre_b))) + 1e-06
        )
        gl.store(
            post_ptr + post_base + j,
            2.0 * (1 / (1 + gl.exp(-(post_m * scale_1 + post_b)))),
        )
    comb_offset = 2 * HC_MULT
    for row in gl.static_range(HC_MULT):
        for col in gl.static_range(HC_MULT):
            idx = comb_offset + row * HC_MULT + col
            out_idx = row * HC_MULT + col
            m = gl.load(mixes_ptr + base + idx)
            b = gl.load(hc_base_ptr + idx)
            gl.store(comb_ptr + comb_base + out_idx, m * scale_2 + b)
    for row in gl.static_range(HC_MULT):
        row_ptr0 = comb_ptr + comb_base + row * HC_MULT
        row_max = gl.load(row_ptr0)
        for col in gl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            row_max = gl.maximum(row_max, gl.load(row_ptr))
        row_sum = 0.0
        for col in gl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            v = gl.exp(gl.load(row_ptr) - row_max)
            row_sum += v
            gl.store(row_ptr, v)
        inv_row_sum = 1.0 / row_sum
        for col in gl.static_range(HC_MULT):
            row_ptr = comb_ptr + comb_base + row * HC_MULT + col
            v = gl.load(row_ptr) * inv_row_sum + 1e-06
            gl.store(row_ptr, v)
    for col in gl.static_range(HC_MULT):
        col_sum = 0.0
        for row in gl.static_range(HC_MULT):
            ptr = comb_ptr + comb_base + row * HC_MULT + col
            col_sum += gl.load(ptr)
        inv_col_sum = 1.0 / (col_sum + 1e-06)
        for row in gl.static_range(HC_MULT):
            ptr = comb_ptr + comb_base + row * HC_MULT + col
            gl.store(ptr, gl.load(ptr) * inv_col_sum)
    for _ in range(SINKHORN_ITERS - 1):
        for row in gl.static_range(HC_MULT):
            row_sum = 0.0
            for col in gl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                row_sum += gl.load(ptr)
            inv_row_sum = 1.0 / (row_sum + 1e-06)
            for col in gl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                gl.store(ptr, gl.load(ptr) * inv_row_sum)
        for col in gl.static_range(HC_MULT):
            col_sum = 0.0
            for row in gl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                col_sum += gl.load(ptr)
            inv_col_sum = 1.0 / (col_sum + 1e-06)
            for row in gl.static_range(HC_MULT):
                ptr = comb_ptr + comb_base + row * HC_MULT + col
                gl.store(ptr, gl.load(ptr) * inv_col_sum)
