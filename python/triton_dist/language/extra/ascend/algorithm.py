# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
import triton
import triton.language as tl


@triton.jit
def dist_swizzle2d(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
    comm_data_split=1,
    swizzle_direction: tl.constexpr = 1,  # 0: data tiles, 1: ranks
):
    """Map a linear task id to ``(tile row, tile col, rank)``.

    ``(comm_data_split, comm_npu_split)`` decides the core splits.
    Direction 0 groups data tiles and traverses ranks. Direction 1 groups ranks
    and traverses data tiles.
    """
    data_row_loop_num = tl.cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = tl.cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num

    if swizzle_direction == 0:
        grouped_shape = data_loop_num
        traversed_shape = rank_size
        swizzle_offset = comm_data_split
    else:
        grouped_shape = rank_size
        traversed_shape = data_loop_num
        swizzle_offset = comm_npu_split

    group_size = swizzle_offset * traversed_shape
    group_idx = iter_id // group_size
    group_offset = iter_id - group_idx * group_size
    in_group_size = tl.minimum(
        swizzle_offset,
        grouped_shape - group_idx * swizzle_offset,
    )
    traversed_idx = group_offset // in_group_size
    grouped_idx = group_idx * swizzle_offset + group_offset % in_group_size

    if swizzle_direction == 0:
        data_tile_idx = grouped_idx
        rank_idx = traversed_idx
    else:
        data_tile_idx = traversed_idx
        rank_idx = grouped_idx

    # Permute and rotate the target rank
    rank_stride = rank_size // comm_npu_split
    rank_offset = rank_idx * rank_stride
    rank_idx = (rank_offset + rank_offset // rank_size + data_tile_idx) % rank_size

    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = tl.minimum(
        data_row_shape - data_row_idx * tile_row_shape, tile_row_shape
    )
    comm_col_size = tl.minimum(
        data_col_shape - data_col_idx * tile_col_shape, tile_col_shape
    )
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


@triton.jit
def gemm_swizzle2d(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
    swizzle_direction: tl.constexpr = 1,  # 0: rows, 1: columns
):
    """Map a linear task index to a tile in the 2-D GEMM output grid.

    ``swizzle_offset`` is the maximum number of tiles in one group.
    ``swizzle_direction`` selects the grouped dimension: 0 groups rows and
    traverses columns, while 1 groups columns and traverses rows. Odd groups
    traverse the other dimension in reverse.
    """

    row_loop_num = tl.cdiv(data_row_shape, tile_row_shape)
    col_loop_num = tl.cdiv(data_col_shape, tile_col_shape)

    if swizzle_direction == 0:
        grouped_shape = row_loop_num
        traversed_shape = col_loop_num
    else:
        grouped_shape = col_loop_num
        traversed_shape = row_loop_num

    group_size = swizzle_offset * traversed_shape
    group_idx = iter_id // group_size
    group_offset = iter_id - group_idx * group_size
    in_group_size = tl.minimum(
        swizzle_offset,
        grouped_shape - group_idx * swizzle_offset,
    )
    traversed_idx = group_offset // in_group_size
    grouped_idx = group_idx * swizzle_offset + group_offset % in_group_size

    if group_idx % 2 == 1:
        traversed_idx = traversed_shape - traversed_idx - 1

    if swizzle_direction == 0:
        row_idx = grouped_idx
        col_idx = traversed_idx
    else:
        row_idx = traversed_idx
        col_idx = grouped_idx

    return row_idx, col_idx


@triton.jit
def dist_swizzle2d_Nz(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
):
    """Compatibility special case of ``dist_swizzle2d`` with direction 1.

    Prefer ``dist_swizzle2d`` for new code.
    """
    return dist_swizzle2d(
        iter_id,
        rank_size,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
        comm_npu_split=comm_npu_split,
        comm_data_split=1,
        swizzle_direction=1,
    )


@triton.jit
def gemm_swizzle2d_Nz(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    """Compatibility special case of ``gemm_swizzle2d`` with direction 1.

    Prefer ``gemm_swizzle2d`` for new code.
    """
    return gemm_swizzle2d(
        iter_id,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
        swizzle_offset=swizzle_offset,
        swizzle_direction=1,
    )
