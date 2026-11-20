# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import pytest
import shmem as ash
import torch
import torch.distributed as dist

import triton
import triton.language as tl

from triton.language.extra.cann.extension import sub_vec_id
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend import libaclshmem_device

from shmem_api_test_utils import check_extern_lowering
from utils import init_aclshmem

G_ASH_SIZE = 1024 * 1024 * 1024

pytestmark = [pytest.mark.dist]


def _init_aclshmem(rank, world_size):
    init_aclshmem(rank, world_size, G_ASH_SIZE, ash.OpEngineType.MTE)


@triton.jit
def _team_pe_mapping_kernel(result_ptr, team, pe):
    if sub_vec_id() == 0:
        mapped_pe = libshmem_device.team_pe_mapping(team, pe)
        tl.store(result_ptr, mapped_pe)


def run_test_team_pe_mapping(rank, world_size):
    _init_aclshmem(rank, world_size)
    result = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
    try:
        result.zero_()
        dist.barrier()
        _team_pe_mapping_kernel[(1, 1, 1)](
            result, libshmem_device.ACLSHMEM_TEAM_WORLD, 0
        )
        dist.barrier()
        assert result.cpu().item() == 0
    finally:
        ash.aclshmem_free_tensor(result)
        ash.aclshmem_finalize()


def test_team_pe_mapping_runtime(dist_test):
    dist_test(run_test_team_pe_mapping, world_size=2)


@pytest.mark.ci
def test_team_pe_mapping_extern_abi(monkeypatch):
    semantic = object()
    calls = []
    core_types = []

    def extern_call(lib_name, lib_path, args, type_map, is_pure, _semantic=None):
        assert _semantic is semantic
        calls.append((lib_name, lib_path, args, type_map, is_pure))
        return "extern_result"

    def cast(value, dtype, _semantic=None):
        assert _semantic is semantic
        return ("cast", value, dtype)

    def mark_core_type(call, core_type, _semantic=None):
        assert _semantic is semantic
        core_types.append(core_type)
        return call

    monkeypatch.setattr(libaclshmem_device, "_mark_core_type", mark_core_type)
    monkeypatch.setattr(libaclshmem_device, "extern_call", extern_call)
    monkeypatch.setattr(libaclshmem_device.tl, "cast", cast)
    monkeypatch.setattr(libshmem_device._shmem_module, "_module", libaclshmem_device)
    team = tl.constexpr(0)
    pe = tl.constexpr(1)
    result = libaclshmem_device.team_pe_mapping(team, pe, _semantic=semantic)
    assert result == "extern_result"
    assert core_types == ["cube_or_vector"]
    assert calls == [
        (
            "libshmem_device",
            "",
            [("cast", team, tl.int32), ("cast", pe, tl.int32)],
            {(tl.int32, tl.int32): ("aclshmem_team_pe_mapping", tl.int32)},
            True,
        )
    ]


@pytest.mark.ci
@pytest.mark.parametrize(
    "mixed,core_type", [(False, "VECTOR"), (True, "CUBE_AND_VECTOR")]
)
def test_team_pe_mapping_extern_lowering(mixed, core_type):
    check_extern_lowering(
        "aclshmem_team_pe_mapping",
        ["i32", "i32"],
        "i32",
        core_type,
        False,
        mixed,
    )
