# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.cann.extension import sub_vec_id
from utils import init_aclshmem

g_ash_size = 1024 * 1024 * 1024


def torch_mype(x0, rank, ncore):
    y_ref = x0.cpu()
    for i in range(ncore):
        y_ref[i] += rank
    return y_ref


@triton.jit
def _shmemx_get_my_pe(ptr):
    subblock_idx = sub_vec_id()
    if subblock_idx == 0:
        mype = dl.rank()
        offset = tl.program_id(0)
        tmp0 = tl.load(ptr + offset, None)
        tmp2 = tmp0 + mype
        tl.store(ptr + offset, tmp2, None)


def run_test_distributed(rank, world_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    init_aclshmem(rank, world_size, g_ash_size, ash.OpEngineType.MTE)
    x0 = torch.zeros([32], dtype=torch.int64).npu()
    y_ref = torch_mype(x0, rank, 3)
    _shmemx_get_my_pe[3, 1, 1](x0)
    assert torch.equal(x0.cpu(), y_ref.cpu())
    _ = ash.aclshmem_finalize()


@pytest.mark.ci
@pytest.mark.dist
def test_mype(dist_test):
    dist_test(run_test_distributed, world_size=2)
