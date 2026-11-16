# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""@prof decorator for NPU event timing in the Ascend perf scripts.

Each perf script decorates its golden and kernel launch functions with @prof,
calls them once, prints the report, then checks correctness:

    from prof import check_all_ranks_close, load_tutorial, prof, prof_report

    @prof(label="allgather_gemm")
    def run_kernel():
        tut.allgather_gemm(A, B, C, peer_mem, ...)

    @prof(label="torch_allgather_gemm")
    def run_golden():
        return tut.torch_allgather_gemm(A, B, world_size)

    C_golden = run_golden()
    run_kernel()
    prof_report(run_golden, run_kernel)
    check_all_ranks_close(C, C_golden, rtol=1e-3, atol=1e-3, context="C")

Warmup and iteration counts resolve as ASCEND_PROFILE_WARMUP / ASCEND_PROFILE_ITERS
(when set) > decorator arguments > defaults (5 / 50). The timing loop matches the
loops previously embedded in the tutorials. Every round is:

    prepare()  ->  barrier  ->  sync  ->  invoke  ->  sync  ->  barrier

and timed rounds additionally record NPU events around invoke, so prepare's
cost (state resets such as signal_mem.fill_(0) or C.zero_()) stays outside the
measured window. prepare is the decorator hook for exactly those resets.

--mode=verify executes once without NPU events; --mode=perf measures without
accuracy comparison; --mode=both (default) enables both. select_cases() applies
--case=full (default) or samples up to ten combinations for --case=random.
"""

import functools
import argparse
import importlib.util
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import sys

WARMUP_ENV = "ASCEND_PROFILE_WARMUP"
ITERS_ENV = "ASCEND_PROFILE_ITERS"
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 50

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

_TUTORIAL_DIR = Path(__file__).resolve().parents[4] / "tutorials" / "ascend"
_KERNEL_DIR = Path(__file__).resolve().parents[2] / "kernels" / "ascend"
# tutorials/ascend/utils.py holds the shared aclshmem bootstrap; tutorial
# scripts import it by plain name (their directory is sys.path[0] under
# torchrun). Make that plain-name import work for tests and by-path loads too.
sys.path.insert(0, str(_TUTORIAL_DIR))
MODE = "both"
CASE = "full"


def run_options(argv=None):
    """Parse shared options, leaving operator-specific options to the script.

    Call once at script startup to set the shared MODE and CASE variables.
    """
    global MODE, CASE

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=("verify", "perf", "both"), default="both")
    parser.add_argument("--case", choices=("random", "full"), default="full")
    options = parser.parse_known_args(argv)[0]
    MODE, CASE = options.mode, options.case
    return options


def verify_enabled():
    return MODE in ("verify", "both")


def perf_enabled():
    return MODE in ("perf", "both")


def select_cases(cases):
    """Sample at most ten cases without replacement, in original order.

    All ranks must supply the same case list. torchrun shares MASTER_PORT;
    run.sh supplies ASCEND_CASE_SEED, which also permits reproducible sweeps.
    """
    cases = list(cases)
    if CASE == "full" or len(cases) <= 10:
        return cases
    seed = os.environ.get("ASCEND_CASE_SEED", os.environ.get("MASTER_PORT", "0"))
    indices = sorted(random.Random(seed).sample(range(len(cases)), 10))
    return [cases[index] for index in indices]


class ProfRecord:
    """Timings (ms) collected for one call of a @prof-decorated function."""

    def __init__(self, label, times_ms):
        self.label = label
        self.times_ms = list(times_ms)

    @property
    def mean(self):
        return sum(self.times_ms) / len(self.times_ms)


def distributed_max_times(times_ms):
    """Return per-iteration times using the slowest rank for each round."""
    values = torch.tensor(times_ms, dtype=torch.float32)
    if not (dist.is_available() and dist.is_initialized()):
        return values
    local = values.npu()
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.stack([row.cpu() for row in gathered], dim=0).max(dim=0).values


def timing_stats(times_ms):
    """Summarize a sequence or tensor of millisecond measurements."""
    values = torch.as_tensor(times_ms, dtype=torch.float32)
    return {
        "mean": float(values.mean().item()),
        "median": float(values.median().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def load_tutorial(name):
    """Import tutorials/ascend/<name>.py by path (filenames contain hyphens)."""
    path = _TUTORIAL_DIR / f"{name}.py"
    mod_name = f"tutorials_ascend_{name}"

    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    return module


def load_kernel(name):
    """Import kernels/ascend/<name>.py from this source tree by file path."""
    path = _KERNEL_DIR / f"{name}.py"
    mod_name = f"kernels_ascend_{name}"

    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise

    return module


def _resolve_counts(warmup, iters):
    def resolve(env_name, arg_value, default):
        env_value = os.environ.get(env_name)
        if env_value is not None:
            return int(env_value)
        if arg_value is not None:
            return arg_value
        return default

    n_warmup = resolve(WARMUP_ENV, warmup, DEFAULT_WARMUP)
    n_iters = resolve(ITERS_ENV, iters, DEFAULT_ITERS)
    if n_warmup < 0 or n_iters <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    return n_warmup, n_iters


def _default_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def prof(
    func=None,
    *,
    label=None,
    warmup=None,
    iters=None,
    prepare=None,
    barrier=None,
):
    """Decorate a launch function for NPU event timing.

    Works both as @prof and @prof(...). Each call runs n_warmup untimed rounds
    and n_iters timed rounds (see module docstring for the round layout),
    returns the callable's result from the last round, and stores the
    per-iteration timings on the wrapper's .prof attribute. The record is
    overwritten on every call, so shape loops can report per shape.

    prepare(), when given, runs at the start of every round, before the
    barrier and outside the recorded event window. It is the hook for state
    resets that must precede each launch and whose cost must stay unmeasured,
    e.g. signal_mem.fill_(0) for wait/notify kernels or C.zero_() before
    accumulate-style kernels. The round layout guarantees the contract the
    signal-based kernels rely on: the previous round ends with a device sync
    and a barrier, so every rank's kernel (including its remote writes into
    peers' signal_mem) has landed before prepare runs, and prepare's own
    device work is synchronized before the launch is enqueued.
    """

    if func is None:
        return functools.partial(
            prof,
            label=label,
            warmup=warmup,
            iters=iters,
            prepare=prepare,
            barrier=barrier,
        )

    tag = label or func.__name__
    do_barrier = barrier if barrier is not None else _default_barrier

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not perf_enabled():
            if prepare is not None:
                prepare()
            do_barrier()
            torch.npu.synchronize()
            result = func(*args, **kwargs)
            torch.npu.synchronize()
            do_barrier()
            wrapper.prof = None
            return result
        n_warmup, n_iters = _resolve_counts(warmup, iters)
        result = None
        for _ in range(n_warmup):
            if prepare is not None:
                prepare()
            do_barrier()
            torch.npu.synchronize()
            result = func(*args, **kwargs)
            torch.npu.synchronize()
            do_barrier()
        starts = [torch.npu.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.npu.Event(enable_timing=True) for _ in range(n_iters)]
        for index in range(n_iters):
            if prepare is not None:
                prepare()
            do_barrier()
            torch.npu.synchronize()
            starts[index].record()
            result = func(*args, **kwargs)
            ends[index].record()
            torch.npu.synchronize()
            do_barrier()
        torch.npu.synchronize()
        wrapper.prof = ProfRecord(
            tag, [start.elapsed_time(end) for start, end in zip(starts, ends)]
        )
        return result

    wrapper.prof = None
    return wrapper


def prof_report(golden_fn, kernel_fn, context="", print_rank=0):
    """Print the EVENT_TIMING line from the latest records, on print_rank only."""
    if not perf_enabled():
        return
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != print_rank:
        return
    golden_rec, kernel_rec = golden_fn.prof, kernel_fn.prof
    if golden_rec is None or kernel_rec is None:
        raise RuntimeError("call both @prof functions before prof_report()")
    suffix = f" {context}" if context else ""
    print(
        f"EVENT_TIMING {kernel_rec.label}{suffix} golden={golden_rec.mean:.4f}ms "
        f"kernel={kernel_rec.mean:.4f}ms "
        f"speedup={golden_rec.mean / kernel_rec.mean:.3f}x",
        flush=True,
    )


def check_all_ranks_close(actual, expected, rtol, atol, context="result"):
    """assert_close on this rank, then print PASS/FAIL rank by rank across the job.

    Raises AssertionError when any rank failed, after every rank has printed.
    """
    if not verify_enabled():
        return
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    passed = torch.tensor([1], dtype=torch.int32).npu()
    error_msg = ""
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as err:
        passed[0] = 0
        error_msg = str(err)
    if world_size == 1:
        all_passed = [passed]
    else:
        all_passed = [
            torch.zeros(1, dtype=torch.int32).npu() for _ in range(world_size)
        ]
        dist.all_gather(all_passed, passed)
    for rank_id in range(world_size):
        if rank == rank_id:
            if all_passed[rank_id].item() == 1:
                print(
                    f"{GREEN}[PASS]{RESET} Rank {rank}: {context} matches golden "
                    f"within tolerances (rtol={rtol}, atol={atol}).",
                    flush=True,
                )
            else:
                print(
                    f"{RED}[FAIL]{RESET} Rank {rank}: {context} does NOT match "
                    f"golden. Details:\n{error_msg}",
                    flush=True,
                )
        if distributed:
            dist.barrier()
    if any(r.item() == 0 for r in all_passed):
        raise AssertionError(error_msg or f"{context} correctness check failed")
