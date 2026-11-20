# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""End-to-end Ascend tests for ``triton_dist.tune.autotune`` (function-level autotuner).

Drives the full pipeline on Ascend: config space + prune_fn, warmup + timed
runs via the backend-agnostic Event API, in-memory + on-disk JSON cache, and
distributed tuning over a HCCL ProcessGroup (all_reduce MAX of timings).

A simple elementwise ``y = alpha * x`` kernel is used so correctness is
guaranteed on every Ascend/triton-ascend revision; the autotuner machinery is
exercised identically to a real GEMM.

"""

from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl
from triton_dist.utils import (
    _npu_smi_index,
    barrier_async,
    current_device_str,
    get_aclshmem_hash,
    get_aclshmem_home,
    get_aclshmem_lib_path,
    get_aclshmem_version,
    get_current_gpu_clock_rate_in_khz,
    get_max_gpu_clock_rate_in_khz,
    get_shmem_backend,
    get_shmem_hash,
    get_shmem_version,
    get_smi_device_index,
    is_ascend,
    torch_device_api,
    torch_device_backend,
    wait_until_max_gpu_clock_or_warning,
)


pytestmark = pytest.mark.skipif(not is_ascend(), reason="Ascend only")


# --------------------------------------------------------------------------- #
# kernel + config space (pure triton, importable everywhere)
# --------------------------------------------------------------------------- #
@triton.jit
def _scale_kernel(x_ptr, y_ptr, alpha, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, alpha * x, mask=mask)


def _scale_config_space():
    return [
        {"cfg": triton.Config({"BLOCK": b}, num_warps=w)}
        for b in (256, 512, 1024, 2048)
        for w in (4, 8)
    ]


def _key_fn(x, alpha, *args, **kwargs):
    return (x.shape, x.dtype)


def _prune_fn(entry, x, alpha, *args, **kwargs):
    # prune configs whose BLOCK exceeds a (conservative) shared-mem budget
    block = entry["cfg"].all_kwargs()["BLOCK"]
    return block * x.element_size() < 32 * 1024


# --------------------------------------------------------------------------- #
# autotune-wrapped function (only defined on a real Ascend box)
# --------------------------------------------------------------------------- #
if is_ascend():
    from triton_dist.tune import autotune

    @autotune(
        config_space=_scale_config_space(),
        key_fn=_key_fn,
        prune_fn=_prune_fn,
    )
    def scale_fn(x, alpha, cfg: triton.Config):
        y = torch.empty_like(x)
        meta = cfg.all_kwargs()
        grid = (triton.cdiv(x.numel(), meta["BLOCK"]),)
        _scale_kernel[grid](x, y, alpha, x.numel(), **meta)
        return y

else:
    scale_fn = None


def _fake_autotune_fn(x, **kwargs):
    return x


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _clear_scale_fn_cache():
    import shutil

    cache_dir = Path.home() / ".triton_dist" / "autotune" / "scale_fn"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def test_autotune_single_process():
    torch.npu.set_device(0)
    _clear_scale_fn_cache()
    n = 1 << 18
    x = torch.randn(n, device="npu", dtype=torch.float32)
    alpha = 2.5

    y = scale_fn(x, alpha, autotune=True, autotune_verbose=True)
    assert torch.allclose(y, alpha * x, atol=1e-5, rtol=1e-4)

    cache_dir = Path.home() / ".triton_dist" / "autotune" / "scale_fn"
    assert cache_dir.exists(), f"cache dir {cache_dir} not created"
    assert list(cache_dir.glob("*.json")), "no cache json written"

    y2 = scale_fn(x, alpha, autotune=True)  # in-memory cache hit
    assert torch.equal(y, y2)


def test_autotune_disabled_uses_first_config():
    torch.npu.set_device(0)
    x = torch.randn(1 << 14, device="npu", dtype=torch.float32)
    y = scale_fn(x, 1.0, autotune=False)  # config_space[0]
    assert torch.allclose(y, x, atol=1e-5, rtol=1e-4)


def _worker_distributed_autotune(rank, world_size):
    import torch.distributed as dist
    from triton_dist import utils as _td_utils

    torch.npu.set_device(rank)
    # AutoTuner.tune consults triton_dist's local world size
    _td_utils._TRITON_DIST_LOCAL_WORLD_SIZE = world_size

    n = 1 << 17
    x = torch.randn(n, device=f"npu:{rank}", dtype=torch.float32)
    alpha = -1.25

    pg = dist.new_group(ranks=list(range(world_size)))
    y = scale_fn(x, alpha, autotune=True, autotune_pg=pg)

    assert torch.allclose(
        y, alpha * x, atol=1e-5, rtol=1e-4
    ), f"rank {rank} correctness failed"


def test_autotune_distributed(dist_test):
    _clear_scale_fn_cache()
    dist_test(_worker_distributed_autotune, world_size=2)


def test_deps_include_ascend_extras():
    from triton_dist.tune import get_deps

    deps = get_deps()
    assert "aclshmem.version" in deps
    assert "aclshmem.hash" in deps
    assert "torch_npu.version" in deps


# --------------------------------------------------------------------------- #
# Ascend-specific helpers in triton_dist.utils.
# --------------------------------------------------------------------------- #
def test_device_identification():
    """torch_device_backend / torch_device_api / current_device_str."""
    torch.npu.set_device(0)
    torch_device_backend.cache_clear()
    assert torch_device_backend() == "npu"
    assert torch_device_api() is torch.npu
    assert current_device_str() == f"npu:{torch.npu.current_device()}"


def test_smi_index_and_device_index():
    """_npu_smi_index and get_smi_device_index: explicit id + None fallback."""
    torch.npu.set_device(0)
    assert _npu_smi_index(3) == 3
    assert _npu_smi_index(None) == torch.npu.current_device()
    assert get_smi_device_index(7) == 7
    assert get_smi_device_index(None) == torch.npu.current_device()


def test_parse_aicore_clocks_mhz():
    from triton_dist.ascend_utils import _parse_aicore_clocks_mhz

    frequencies = """
        AI Core Current Freq                         : 800 MHZ
        AI Core Rated Freq                           : 1850 MHZ
    """
    assert _parse_aicore_clocks_mhz(frequencies) == (800, 1850)

    common = "        Aicore Freq(MHZ)       : 0 / 1800\n        Aicore Freq Rated(MHZ) : 1800\n"
    assert _parse_aicore_clocks_mhz(common) == (0, 1800)


def test_clock_rate_from_npu_smi():
    """Clock helpers query npu-smi; rated freq is positive and >= current."""
    torch.npu.set_device(0)
    get_max_gpu_clock_rate_in_khz.cache_clear()
    max_khz = get_max_gpu_clock_rate_in_khz(0)
    cur_khz = get_current_gpu_clock_rate_in_khz(0)
    assert max_khz > 0
    assert cur_khz >= 0
    assert max_khz >= cur_khz


@pytest.mark.dist
def test_autotuner_logpath_and_tune_unit(monkeypatch, tmp_path):
    from triton_dist import tune as _tune_mod
    from triton_dist.tune import AutoTuner
    import types

    fake_event = types.SimpleNamespace(
        record=lambda: None, synchronize=lambda: None, elapsed_time=lambda o: 1.0
    )
    fake_api = types.SimpleNamespace(
        Event=lambda **k: fake_event, synchronize=lambda: None
    )
    fake_pg = types.SimpleNamespace(broadcast=lambda t, s: None, rank=lambda: 0)

    at = AutoTuner(_fake_autotune_fn, [{"BLOCK": 256}], key_fn=lambda x, **kw: (x,))
    at.cache_dir = tmp_path / "autotune"
    monkeypatch.setattr(_tune_mod, "torch_device_api", lambda: fake_api)
    monkeypatch.setattr(_tune_mod, "current_device_str", lambda: "cpu")
    monkeypatch.setattr(_tune_mod, "hw_hash", lambda *a, **k: "fakehw")

    assert at(5, autotune=True) == 5
    assert "rank0" in at._get_tune_log_path((5,), fake_pg).name


def test_wait_until_max_gpu_clock_default_skip():
    """Without TRITON_DIST_WAIT_GPU_CLOCK set, the function short-circuits."""
    assert wait_until_max_gpu_clock_or_warning() is True


def test_shmem_backend_and_dispatch():
    """get_shmem_backend -> 'aclshmem'; version/hash delegate to aclshmem."""
    assert get_shmem_backend() == "aclshmem"
    assert get_shmem_version() == get_aclshmem_version()
    assert get_shmem_hash() == get_aclshmem_hash()


def test_aclshmem_discovery():
    """get_aclshmem_home / version / lib_path / hash."""
    home = get_aclshmem_home()
    assert isinstance(home, Path) and home.exists()
    assert get_aclshmem_version() == "unknown"
    path = get_aclshmem_lib_path()
    assert path is None or (isinstance(path, Path) and path.is_file())
    h = get_aclshmem_hash()
    assert h == "unknown" or all(c in "0123456789abcdef" for c in h)


def test_barrier_async_runs_allreduce():
    """barrier_async must create an NPU tensor and call pg.allreduce on it.

    Uses a single-process torch.distributed group so no spawn is needed.
    """
    torch.npu.set_device(0)
    import torch.distributed as dist

    if not dist.is_initialized():
        pytest.skip("requires an initialized torch.distributed process group")
    pg = dist.new_group(ranks=list(range(dist.get_world_size())), backend="hccl")
    barrier_async(pg)  # must not raise
    torch.npu.synchronize()
