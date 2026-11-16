# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import torch

import triton
import triton.language as tl
from triton_dist.language.extra.ascend.algorithm import (
    dist_swizzle2d,
    gemm_swizzle2d,
)


@triton.jit
def _dist_swizzle_kernel(
    output,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split,
    comm_data_split,
    swizzle_direction: tl.constexpr,
):
    iter_id = tl.program_id(0)
    row_idx, col_idx, rank_idx, row_size, col_size = dist_swizzle2d(
        iter_id,
        rank_size,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
        comm_npu_split,
        comm_data_split,
        swizzle_direction,
    )
    output_offset = iter_id * 5
    tl.store(output + output_offset, row_idx)
    tl.store(output + output_offset + 1, col_idx)
    tl.store(output + output_offset + 2, rank_idx)
    tl.store(output + output_offset + 3, row_size)
    tl.store(output + output_offset + 4, col_size)


@triton.jit
def _dist_swizzle_default_kernel(
    output,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
):
    iter_id = tl.program_id(0)
    row_idx, col_idx, rank_idx, row_size, col_size = dist_swizzle2d(
        iter_id,
        rank_size,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
    )
    output_offset = iter_id * 5
    tl.store(output + output_offset, row_idx)
    tl.store(output + output_offset + 1, col_idx)
    tl.store(output + output_offset + 2, rank_idx)
    tl.store(output + output_offset + 3, row_size)
    tl.store(output + output_offset + 4, col_size)


@triton.jit
def _gemm_swizzle_kernel(
    output,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset,
    swizzle_direction: tl.constexpr,
):
    iter_id = tl.program_id(0)
    row_idx, col_idx = gemm_swizzle2d(
        iter_id,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
        swizzle_offset,
        swizzle_direction,
    )
    output_offset = iter_id * 2
    tl.store(output + output_offset, row_idx)
    tl.store(output + output_offset + 1, col_idx)


@triton.jit
def _gemm_swizzle_default_kernel(
    output,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
):
    iter_id = tl.program_id(0)
    row_idx, col_idx = gemm_swizzle2d(
        iter_id,
        data_row_shape,
        data_col_shape,
        tile_row_shape,
        tile_col_shape,
    )
    output_offset = iter_id * 2
    tl.store(output + output_offset, row_idx)
    tl.store(output + output_offset + 1, col_idx)


def _cdiv(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def _dist_reference(
    iter_id,
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
    comm_data_split=1,
    swizzle_direction=1,
):
    data_row_loop_num = _cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = _cdiv(data_col_shape, tile_col_shape)
    data_loop_num = data_row_loop_num * data_col_loop_num

    grid_shape = (data_loop_num, rank_size)
    core_split = (comm_data_split, comm_npu_split)
    grouped_dim = swizzle_direction
    traversed_dim = 1 - swizzle_direction
    swizzle_offset = core_split[grouped_dim]

    group_size = swizzle_offset * grid_shape[traversed_dim]
    group_idx = iter_id // group_size
    group_offset = iter_id - group_idx * group_size
    in_group_size = min(
        swizzle_offset,
        grid_shape[grouped_dim] - group_idx * swizzle_offset,
    )

    traversed_idx = group_offset // in_group_size
    local_grouped_idx = group_offset % in_group_size
    grouped_idx = group_idx * swizzle_offset + local_grouped_idx

    if swizzle_direction == 0:
        data_tile_idx = grouped_idx
        rank_idx = traversed_idx
    else:
        data_tile_idx = traversed_idx
        rank_idx = grouped_idx

    rank_stride = rank_size // comm_npu_split
    rank_offset = rank_idx * rank_stride
    rank_idx = (rank_offset + rank_offset // rank_size + data_tile_idx) % rank_size
    data_row_idx = data_tile_idx // data_col_loop_num
    data_col_idx = data_tile_idx % data_col_loop_num
    comm_row_size = min(
        data_row_shape - data_row_idx * tile_row_shape, tile_row_shape
    )
    comm_col_size = min(
        data_col_shape - data_col_idx * tile_col_shape, tile_col_shape
    )
    return data_row_idx, data_col_idx, rank_idx, comm_row_size, comm_col_size


def _gemm_reference(
    iter_id,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
):
    data_row_loop_num = _cdiv(data_row_shape, tile_row_shape)
    data_col_loop_num = _cdiv(data_col_shape, tile_col_shape)
    col_loop_num = _cdiv(data_col_loop_num, swizzle_offset)
    n_tile_idx = iter_id // (swizzle_offset * data_row_loop_num)
    m_n_tile_idx = iter_id % (swizzle_offset * data_row_loop_num)
    n_tile_size = swizzle_offset
    if n_tile_idx == col_loop_num - 1:
        n_tile_size = data_col_loop_num - swizzle_offset * n_tile_idx
    data_row_idx = m_n_tile_idx // n_tile_size
    data_col_idx = n_tile_idx * swizzle_offset + m_n_tile_idx % n_tile_size
    if n_tile_idx % 2 == 1:
        data_row_idx = data_row_loop_num - data_row_idx - 1
    return data_row_idx, data_col_idx


def _run_dist_swizzle(
    rank_size,
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    comm_npu_split=1,
    comm_data_split=1,
    swizzle_direction=1,
    use_default=False,
):
    data_loop_num = _cdiv(data_row_shape, tile_row_shape) * _cdiv(
        data_col_shape, tile_col_shape
    )
    iteration_count = rank_size * data_loop_num
    output = torch.empty((iteration_count, 5), dtype=torch.int32).npu()
    grid = (iteration_count, 1, 1)
    if use_default:
        _dist_swizzle_default_kernel[grid](
            output,
            rank_size,
            data_row_shape,
            data_col_shape,
            tile_row_shape,
            tile_col_shape,
        )
    else:
        _dist_swizzle_kernel[grid](
            output,
            rank_size,
            data_row_shape,
            data_col_shape,
            tile_row_shape,
            tile_col_shape,
            comm_npu_split,
            comm_data_split,
            swizzle_direction,
        )
    return [tuple(row) for row in output.cpu().tolist()]


def _run_gemm_swizzle(
    data_row_shape,
    data_col_shape,
    tile_row_shape,
    tile_col_shape,
    swizzle_offset=7,
    swizzle_direction=1,
    use_default=False,
):
    iteration_count = _cdiv(data_row_shape, tile_row_shape) * _cdiv(
        data_col_shape, tile_col_shape
    )
    output = torch.empty((iteration_count, 2), dtype=torch.int32).npu()
    grid = (iteration_count, 1, 1)
    if use_default:
        _gemm_swizzle_default_kernel[grid](
            output,
            data_row_shape,
            data_col_shape,
            tile_row_shape,
            tile_col_shape,
        )
    else:
        _gemm_swizzle_kernel[grid](
            output,
            data_row_shape,
            data_col_shape,
            tile_row_shape,
            tile_col_shape,
            swizzle_offset,
            swizzle_direction,
        )
    return [tuple(row) for row in output.cpu().tolist()]


def test_dist_swizzle_default_rank_group():
    args = (4, 40, 1024, 20, 256)
    actual = _run_dist_swizzle(*args, use_default=True)
    expected = [_dist_reference(iter_id, *args) for iter_id in range(len(actual))]

    assert actual == expected
    for data_tile_idx in range(8):
        tile_row_idx, tile_col_idx = divmod(data_tile_idx, 4)
        tile_results = [
            result
            for result in actual
            if result[0] == tile_row_idx and result[1] == tile_col_idx
        ]
        assert [result[2] for result in tile_results] == [
            (rank_idx + data_tile_idx) % 4 for rank_idx in range(4)
        ]


def test_dist_swizzle_tail_block():
    args = (4, 45, 530, 20, 256, 1)
    actual = _run_dist_swizzle(*args)
    expected = [_dist_reference(iter_id, *args) for iter_id in range(len(actual))]

    assert actual == expected
    block_sizes = {
        (row, col): (row_size, col_size)
        for row, col, _, row_size, col_size in actual
    }
    assert block_sizes == {
        (0, 0): (20, 256),
        (0, 1): (20, 256),
        (0, 2): (20, 18),
        (1, 0): (20, 256),
        (1, 1): (20, 256),
        (1, 2): (20, 18),
        (2, 0): (5, 256),
        (2, 1): (5, 256),
        (2, 2): (5, 18),
    }


def test_dist_swizzle_rank_tile_bijection():
    rank_size = 12
    args = (rank_size, 45, 530, 20, 256, 3)
    actual = _run_dist_swizzle(*args)
    actual_tiles = {(row, col, rank) for row, col, rank, _, _ in actual}
    expected_tiles = {
        (row, col, rank)
        for row in range(3)
        for col in range(3)
        for rank in range(rank_size)
    }

    assert len(actual_tiles) == len(actual)
    assert actual_tiles == expected_tiles


def test_dist_swizzle_directions():
    common_args = (4, 2, 2, 1, 1, 2, 2)

    direction_zero = _run_dist_swizzle(*common_args, swizzle_direction=0)
    assert [(row, col, rank) for row, col, rank, _, _ in direction_zero] == [
        (0, 0, 0),
        (0, 1, 1),
        (0, 0, 2),
        (0, 1, 3),
        (0, 0, 1),
        (0, 1, 2),
        (0, 0, 3),
        (0, 1, 0),
        (1, 0, 2),
        (1, 1, 3),
        (1, 0, 0),
        (1, 1, 1),
        (1, 0, 3),
        (1, 1, 0),
        (1, 0, 1),
        (1, 1, 2),
    ]

    direction_one = _run_dist_swizzle(*common_args, swizzle_direction=1)
    assert [(row, col, rank) for row, col, rank, _, _ in direction_one] == [
        (0, 0, 0),
        (0, 0, 2),
        (0, 1, 1),
        (0, 1, 3),
        (1, 0, 2),
        (1, 0, 0),
        (1, 1, 3),
        (1, 1, 1),
        (0, 0, 1),
        (0, 0, 3),
        (0, 1, 2),
        (0, 1, 0),
        (1, 0, 3),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 2),
    ]


def test_dist_swizzle_direction_zero_tail_group():
    args = (4, 1, 3, 1, 1, 2, 2, 0)
    actual = _run_dist_swizzle(*args)
    expected = [_dist_reference(iter_id, *args) for iter_id in range(len(actual))]

    assert actual == expected
    assert {(row, col, rank) for row, col, rank, _, _ in actual} == {
        (0, col, rank) for col in range(3) for rank in range(4)
    }


def test_gemm_swizzle_snake_direction():
    args = (60, 1792, 20, 128)
    actual = _run_gemm_swizzle(*args, use_default=True)
    expected = [_gemm_reference(iter_id, *args) for iter_id in range(len(actual))]

    assert actual == expected
    first_group_end = 7 * 3 - 1
    second_group_begin = first_group_end + 1
    assert actual[first_group_end] == (2, 6)
    assert actual[second_group_begin] == (2, 7)
    assert [actual[second_group_begin + offset * 7][0] for offset in range(3)] == [
        2,
        1,
        0,
    ]


def test_gemm_swizzle_directions():
    args = (3, 3, 1, 1, 2)

    direction_zero = _run_gemm_swizzle(*args, swizzle_direction=0)
    assert direction_zero == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 1),
        (2, 0),
    ]

    direction_one = _run_gemm_swizzle(*args, swizzle_direction=1)
    assert direction_one == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (1, 2),
        (0, 2),
    ]


def test_gemm_swizzle_tail_ngroup():
    args = (60, 1280, 20, 128, 7)
    actual = _run_gemm_swizzle(*args)
    expected = [_gemm_reference(iter_id, *args) for iter_id in range(len(actual))]

    assert actual == expected
    tail_group = actual[7 * 3 :]
    assert tail_group == [
        (2, 7),
        (2, 8),
        (2, 9),
        (1, 7),
        (1, 8),
        (1, 9),
        (0, 7),
        (0, 8),
        (0, 9),
    ]


def test_gemm_swizzle_tile_bijection():
    args = (65, 1290, 20, 128, 7)
    actual = _run_gemm_swizzle(*args)
    actual_tiles = set(actual)
    expected_tiles = {(row, col) for row in range(4) for col in range(11)}

    assert len(actual_tiles) == len(actual)
    assert actual_tiles == expected_tiles
