# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""End-to-end Ascend test for ``triton_dist.autotuner.contextual_autotune``.

The contextual autotuner wraps a function that internally launches a kernel
decorated by ``triton.autotune``. On Ascend the per-config timing is taken via
``triton.runtime.driver.active`` (backend-agnostic) and the distributed path
uses ``all_reduce(MAX)`` over a HCCL group.

We vary only ``num_warps`` (BLOCK is fixed) so the launch grid is independent
of the chosen config -> guaranteed correctness regardless of which config wins.

"""

import pytest
import torch
import triton
import triton.language as tl
from triton_dist.utils import is_ascend


pytestmark = pytest.mark.skipif(not is_ascend(), reason="Ascend only")


_BLOCK = 512


def _scale_configs():
    # only num_warps varies -> grid (cdiv(n, _BLOCK)) is config-independent
    return [triton.Config({"BLOCK": _BLOCK}, num_warps=w) for w in (4, 8)]


# pure triton: importable everywhere
@triton.autotune(configs=_scale_configs(), key=["n"])
@triton.jit
def _scale_kernel(x_ptr, y_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, alpha * x, mask=mask)


if is_ascend():
    from triton_dist.autotuner import contextual_autotune

    @contextual_autotune(is_dist=False, n_repeat=3, n_warmup=2)
    def run_scale(x, alpha):
        y = torch.empty_like(x)
        grid = (triton.cdiv(x.numel(), _BLOCK),)
        _scale_kernel[grid](x, y, alpha, x.numel())
        return y

else:
    run_scale = None


def test_contextual_autotune_ascend():
    torch.npu.set_device(0)
    n = 1 << 18
    x = torch.randn(n, device="npu", dtype=torch.float32)
    alpha = 3.0

    y = run_scale(x, alpha)
    assert torch.allclose(y, alpha * x, atol=1e-5, rtol=1e-4)

    # the contextual machinery must have selected a best config on the inner
    # triton.autotuner and cached it.
    assert _scale_kernel.best_config is not None, "best_config was not selected"

    y2 = run_scale(x, alpha)  # cache hit, no re-tune
    assert torch.allclose(y2, alpha * x, atol=1e-5, rtol=1e-4)


def _worker_distributed_contextual(rank, world_size):
    torch.npu.set_device(rank)
    n = 1 << 17
    x = torch.randn(n, device=f"npu:{rank}", dtype=torch.float32)
    alpha = 0.75

    # is_dist=True exercises the all_reduce(MAX) timing-sync path over HCCL.
    @contextual_autotune(is_dist=True, n_repeat=3, n_warmup=2)
    def _run():
        y = torch.empty_like(x)
        grid = (triton.cdiv(n, _BLOCK),)
        _scale_kernel[grid](x, y, alpha, n)
        return y

    y = _run()
    assert torch.allclose(y, alpha * x, atol=1e-5, rtol=1e-4), f"rank {rank} failed"


def test_contextual_autotune_distributed(dist_test):
    dist_test(_worker_distributed_contextual, world_size=2)
