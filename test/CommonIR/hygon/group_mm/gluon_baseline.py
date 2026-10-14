from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna3


@gluon.jit
def _grouped_tile(
    pid, m, n, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, GROUP_M: gl.constexpr
):
    grid_m = gl.cdiv(m, BLOCK_M)
    grid_n = gl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = gl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    tile_m = group_id * GROUP_M + pid % group_size
    tile_n = pid % width // group_size
    return tile_m, tile_n


@gluon.jit
def _load_descriptor_tile(
    base,
    rows,
    cols,
    stride_row,
    stride_col,
    row_offset,
    col_offset,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    # Descriptor coordinates/strides are i64, even when scheduling is i32.
    row = row_offset.to(gl.int64) + gl.arange(
        0, ROWS, layout=gl.SliceLayout(1, LAYOUT)
    ).to(gl.int64)
    col = col_offset.to(gl.int64) + gl.arange(
        0, COLS, layout=gl.SliceLayout(0, LAYOUT)
    ).to(gl.int64)
    pointers = (
        base
        + row[:, None] * gl.cast(stride_row, gl.int64)
        + col[None, :] * gl.cast(stride_col, gl.int64)
    )
    valid = (
        (row[:, None] >= 0)
        & (row[:, None] < gl.cast(rows, gl.int64))
        & (col[None, :] >= 0)
        & (col[None, :] < gl.cast(cols, gl.int64))
    )
    return gl.load(pointers, mask=valid, other=0)


@gluon.jit
def _store_descriptor_tile(
    base,
    values,
    rows,
    cols,
    stride_row,
    stride_col,
    row_offset,
    col_offset,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    LAYOUT: gl.constexpr,
):
    row = row_offset.to(gl.int64) + gl.arange(
        0, ROWS, layout=gl.SliceLayout(1, LAYOUT)
    ).to(gl.int64)
    col = col_offset.to(gl.int64) + gl.arange(
        0, COLS, layout=gl.SliceLayout(0, LAYOUT)
    ).to(gl.int64)
    pointers = (
        base
        + row[:, None] * gl.cast(stride_row, gl.int64)
        + col[None, :] * gl.cast(stride_col, gl.int64)
    )
    valid = (
        (row[:, None] >= 0)
        & (row[:, None] < gl.cast(rows, gl.int64))
        & (col[None, :] >= 0)
        & (col[None, :] < gl.cast(cols, gl.int64))
    )
    gl.store(pointers, gl.convert_layout(values, LAYOUT), mask=valid)


@gluon.jit
def grouped_mm_kernel(
    a_desc,
    b_desc,
    c_desc,
    C,
    offs,
    num_groups: gl.constexpr,
    M,
    N: gl.constexpr,
    K: gl.constexpr,
    stride_cm: gl.constexpr,
    stride_cn: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
):
    # Tuples are an explicit host ABI: (base, shape, strides, block_shape).
    # Shape and strides are constexpr, exactly as in the historical Gluon.
    # These are not TensorDescriptor objects or an alternative device DSL.
    A_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [2, 4], [0, 1])
    B_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [1, 8], [0, 1])
    MMA_LAYOUT: gl.constexpr = AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[2, 4],
        element_bitwidth=32,
    )
    A_SHARED: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
    B_SHARED: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, [0, 1])
    FULL_STORE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [2, 4], [0, 1])
    TAIL_STORE_LAYOUT: gl.constexpr = gl.BlockedLayout([1, 8], [2, 32], [8, 1], [1, 0])

    total_grid = gl.num_programs(axis=0)
    tile_idx = gl.program_id(axis=0)
    num_n_tiles = gl.cdiv(N, BLOCK_N)
    last_problem_end = 0
    group_start = 0

    for group_idx in range(num_groups):
        group_end = gl.load(offs + group_idx).to(gl.int32)
        group_m = group_end - group_start
        num_tiles = gl.cdiv(group_m, BLOCK_M) * num_n_tiles
        current_problem_end = last_problem_end + num_tiles

        if tile_idx >= last_problem_end and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in range(loop_count):
                tile_m, tile_n = _grouped_tile(
                    tile_idx - last_problem_end,
                    group_m,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    GROUP_M,
                )
                row_offset = group_start + tile_m * BLOCK_M
                col_offset = tile_n * BLOCK_N
                b_row_offset = group_idx * K
                accumulator = gl.full(
                    [BLOCK_M, BLOCK_N],
                    0,
                    gl.float32,
                    layout=MMA_LAYOUT,
                )

                for k_tile in range(gl.cdiv(K, BLOCK_K)):
                    a = _load_descriptor_tile(
                        a_desc[0],
                        a_desc[1][0],
                        a_desc[1][1],
                        a_desc[2][0],
                        a_desc[2][1],
                        row_offset,
                        k_tile * BLOCK_K,
                        BLOCK_M,
                        BLOCK_K,
                        A_LAYOUT,
                    )
                    b = _load_descriptor_tile(
                        b_desc[0],
                        b_desc[1][0],
                        b_desc[1][1],
                        b_desc[2][0],
                        b_desc[2][1],
                        b_row_offset + k_tile * BLOCK_K,
                        col_offset,
                        BLOCK_K,
                        BLOCK_N,
                        B_LAYOUT,
                    )
                    # Explicit LDS organizations from the historical final IR.
                    # Do not depend on helper-injected optimize-dot passes.
                    a_shared = gl.allocate_shared_memory(
                        gl.bfloat16,
                        [BLOCK_M, BLOCK_K],
                        A_SHARED,
                        a,
                    )
                    a_operand = a_shared.load(gl.DotOperandLayout(0, MMA_LAYOUT, 4))
                    b_shared = gl.allocate_shared_memory(
                        gl.bfloat16,
                        [BLOCK_K, BLOCK_N],
                        B_SHARED,
                        b,
                    )
                    b_operand = b_shared.load(gl.DotOperandLayout(1, MMA_LAYOUT, 4))
                    accumulator = cdna3.mfma(a_operand, b_operand, accumulator)

                values = accumulator.to(C.dtype.element_ty)
                if row_offset + BLOCK_M <= group_end:
                    _store_descriptor_tile(
                        c_desc[0],
                        values,
                        c_desc[1][0],
                        c_desc[1][1],
                        c_desc[2][0],
                        c_desc[2][1],
                        row_offset,
                        col_offset,
                        BLOCK_M,
                        BLOCK_N,
                        FULL_STORE_LAYOUT,
                    )
                else:
                    # Unlike descriptor coordinates, the original tail address
                    # DAG is i32. Preserve the group-end mask, not just M.
                    row = row_offset + gl.arange(
                        0,
                        BLOCK_M,
                        layout=gl.SliceLayout(1, TAIL_STORE_LAYOUT),
                    )
                    col = col_offset + gl.arange(
                        0,
                        BLOCK_N,
                        layout=gl.SliceLayout(0, TAIL_STORE_LAYOUT),
                    )
                    pointers = C + stride_cm * row[:, None] + stride_cn * col[None, :]
                    valid = (row[:, None] < group_end) & (col[None, :] < N)
                    gl.store(
                        pointers,
                        gl.convert_layout(values, TAIL_STORE_LAYOUT),
                        mask=valid,
                    )
                tile_idx += total_grid

        last_problem_end = current_problem_end
        group_start = group_end
