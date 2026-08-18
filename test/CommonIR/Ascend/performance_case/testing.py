# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import builtins
import fnmatch
import math
import multiprocessing
import os
import shutil
from datetime import datetime, timezone
from typing import Callable, List, Optional, Union

import torch
import torch_npu

import triton.runtime as runtime
from triton.knobs import cache

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# A single timing result or one result per function.
BenchResult = Union[float, List[float]]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProfilerResultMismatchError(RuntimeError):
    """Raised when the number of profiler rows does not match expectations."""

    def __init__(self, target_kernel_name: str, expected_rows: int, actual_rows: int) -> None:
        self.target_kernel_name = target_kernel_name
        self.expected_rows = expected_rows
        self.actual_rows = actual_rows
        super().__init__("Profiler rows filtered by target kernel name do not match the expected count. "
                         f"target_kernel_name={target_kernel_name!r}, "
                         f"expected_rows={expected_rows}, actual_rows={actual_rows}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_prof_dir(prof_dir: Optional[str]) -> str:
    """Return the profiler output directory, generating a unique path when not provided."""
    if prof_dir is not None:
        return prof_dir
    process = multiprocessing.current_process()
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_path = cache.get_triton_dir("profile_results")
    return os.path.join(base_path, f"prof_{timestamp}_{process.name}-{process.pid}")


def _make_l2_cache_buffer() -> torch.Tensor:
    """Allocate and warm up the L2-cache-clearing buffer."""
    buffer = runtime.driver.active.get_empty_cache_for_benchmark().float()
    buffer.sum()
    torch.npu.synchronize()
    return buffer


def _remove_dir(path: str, keep: bool) -> None:
    """Delete *path* recursively unless *keep* is True."""
    if keep:
        return
    if os.path.exists(path):
        shutil.rmtree(path)


def _inf_result(num_funcs: int) -> BenchResult:
    """Return +inf for a single function or a list of +inf for multiple functions."""
    return float("inf") if num_funcs == 1 else [float("inf")] * num_funcs


def _find_csv(base_dir: str, use_task_time: bool) -> Optional[str]:
    """Walk *base_dir* and return the path of the first matching CSV file."""
    for root, _, files in os.walk(base_dir):
        for file in files:
            if use_task_time and fnmatch.fnmatch(file, "task_time*.csv"):
                return os.path.join(root, file)
            if not use_task_time and file == "kernel_details.csv":
                return os.path.join(root, file)
    return None


def _active_run_times_single_kernel(
    filter_df,
    col_time: str,
    num_funcs: int,
    num_warmup: int,
    num_active: int,
) -> List[float]:
    """
    Sum active-run times when exactly one kernel row is produced per function call.

    Returns a list of total active time (in microseconds) per function.
    """
    runs_per_func = num_warmup + num_active
    time_cost: List[float] = [0.0] * num_funcs
    for func_idx in range(num_funcs):
        for active_index in range(num_active):
            row = func_idx * runs_per_func + num_warmup + active_index
            time_cost[func_idx] += filter_df.iloc[row][col_time]
    return time_cost


def _active_run_times_multi_kernel(
    filter_df,
    col_time: str,
    num_funcs: int,
    num_warmup: int,
    num_active: int,
    kernels_per_run: int,
) -> List[float]:
    """
    Sum active-run times when each function call produces *kernels_per_run* rows.

    Returns a list of total active time (in microseconds) per function.
    """
    runs_per_func = num_warmup + num_active
    time_cost: List[float] = [0.0] * num_funcs
    for func_idx in range(num_funcs):
        for active_index in range(num_active):
            run_idx = func_idx * runs_per_func + num_warmup + active_index
            start_row = run_idx * kernels_per_run
            end_row = start_row + kernels_per_run
            time_cost[func_idx] += filter_df.iloc[start_row:end_row][col_time].sum()
    return time_cost


def _collect_prof_result(
    base_dir: str,
    funcs: List[Callable],
    num_warmup: int,
    num_active: int,
    target_kernel_name: Optional[str] = None,
    clear_l2_cache: bool = False,
) -> BenchResult:
    """
    Parse profiler CSV output and return the average kernel time in milliseconds.

    Two CSV sources are supported:

    * ``task_time*.csv`` (default, ``target_kernel_name=None``): records every
      kernel dispatched by each function call. A single call may launch multiple
      kernels; all of them are summed to represent one "run".

    * ``kernel_details.csv`` (``target_kernel_name`` specified): only the named
      kernel is counted — exactly one row per function call is expected.

    Args:
        base_dir: Root directory written by ``torch_npu.profiler``.
        funcs: The list of callables that were profiled.
        num_warmup: Number of leading runs to discard.
        num_active: Number of runs whose times are averaged.
        target_kernel_name: When set, only rows whose kernel name matches are
            counted, and exactly one such row per run is required.
        clear_l2_cache: When True, ``ReduceSum`` rows are filtered out because
            they belong to the cache-clearing operation, not the benchmark.

    Returns:
        A single ``float`` when ``len(funcs) == 1``, otherwise a ``list[float]``,
        each value being the average wall-time in milliseconds for that function.
        Returns ``+inf`` (or a list of ``+inf``) if no CSV file is found or the
        row count is inconsistent.
    """
    import pandas as pd

    num_funcs = len(funcs)
    use_task_time = target_kernel_name is None

    csv_path = _find_csv(base_dir, use_task_time)
    if csv_path is None:
        return _inf_result(num_funcs)

    df = pd.read_csv(csv_path)

    if use_task_time:
        # task_time*.csv starts and ends with a PROFILING_DISABLE sentinel row.
        df = df.iloc[1:-1]
        col_time = "task_time(us)"
        # Exclude ReduceSum rows that originate from L2-cache clearing.
        l2_mask = ~df["kernel_name"].str.contains(r"^ReduceSum", case=False, na=False)
    else:
        col_time = "Duration(us)"
        l2_mask = ~df["Type"].str.contains(r"^ReduceSum$", case=False, na=False)

    filter_df = df[l2_mask] if clear_l2_cache else df

    if target_kernel_name is not None:
        filter_df = filter_df[filter_df["Name"] == target_kernel_name]

    actual_rows = len(filter_df)
    total_runs = num_funcs * (num_warmup + num_active)

    if target_kernel_name is not None:
        # Exactly one kernel row per run expected.
        if actual_rows != total_runs:
            raise ProfilerResultMismatchError(target_kernel_name, total_runs, actual_rows)
        time_cost = _active_run_times_single_kernel(filter_df, col_time, num_funcs, num_warmup, num_active)
    else:
        # Each run may produce multiple kernel rows; detect the count.
        if actual_rows % total_runs != 0:
            return _inf_result(num_funcs)
        kernels_per_run = actual_rows // total_runs
        time_cost = _active_run_times_multi_kernel(filter_df, col_time, num_funcs, num_warmup, num_active,
                                                   kernels_per_run)

    # Convert from total microseconds → average milliseconds.
    avg_ms = [t / num_active / 1e3 for t in time_cost]
    return avg_ms[0] if num_funcs == 1 else avg_ms


# ---------------------------------------------------------------------------
# mspti backend
# ---------------------------------------------------------------------------

try:
    from mspti import KernelMonitor
except ImportError:
    KernelMonitor = None


def _collect_mspti_result(
    all_durations: List[int],
    num_funcs: int,
    warmup: int,
    active: int,
    target_kernel_name: Optional[str],
) -> BenchResult:
    """
    Convert the flat list of kernel durations recorded by mspti into
    per-function average times in milliseconds.

    When *target_kernel_name* is ``None`` each function call may have produced
    multiple kernel records; they are summed per run. When a specific kernel
    name was requested there is exactly one record per run.

    Args:
        all_durations: Flat list of kernel durations in nanoseconds, in the
            order they were recorded.
        num_funcs: Number of benchmarked functions.
        warmup: Number of leading runs to discard.
        active: Number of runs to average.
        target_kernel_name: Kernel filter that was applied during recording.

    Returns:
        A single ``float`` when ``num_funcs == 1``, otherwise a ``list[float]``.
        Returns ``+inf`` (or a list thereof) when insufficient data was recorded.
    """
    total = warmup + active
    total_records = len(all_durations)

    if target_kernel_name is not None:
        # One record per run expected.
        expected = num_funcs * total
        if total_records < expected:
            return _inf_result(num_funcs)
        kernels_per_run = 1
    else:
        # Detect how many kernel rows each run produced.
        total_runs = num_funcs * total
        if total_records == 0 or total_records % total_runs != 0:
            return _inf_result(num_funcs)
        kernels_per_run = total_records // total_runs

    avg_ms: List[float] = []
    for func_idx in range(num_funcs):
        total_ns = 0
        for active_index in range(active):
            run_idx = func_idx * total + warmup + active_index
            start = run_idx * kernels_per_run
            end = start + kernels_per_run
            total_ns += sum(all_durations[start:end])
        avg_ms.append(total_ns / active / 1e6)

    return avg_ms[0] if num_funcs == 1 else avg_ms


# If the CANN version is earlier than 9.1.0, libmspti.so must be set in
# LD_PRELOAD before mspti can be imported.
def do_bench_npu_mspti(
    funcs: Union[Callable, List[Callable]],
    warmup: int = 5,
    active: int = 30,
    clear_l2_cache: bool = False,
    target_kernel_name: Optional[str] = None,
) -> BenchResult:
    """
    Benchmark NPU kernels using the lightweight mspti ``KernelMonitor`` API.

    Each function in *funcs* is first called once for a JIT warmup, then run
    ``warmup + active`` times under the monitor. The ``warmup`` leading runs are
    discarded and the remaining ``active`` runs are averaged.

    Args:
        funcs: A single callable or a list of callables to benchmark.
        warmup: Number of runs to discard at the start of measurement.
        active: Number of runs whose times are averaged.
        clear_l2_cache: When True, zero the L2-cache buffer before every run.
            Kernel names containing *zero* or *zeroslike* are excluded from the
            recorded durations because they belong to the cache-clearing step.
        target_kernel_name: When set, only kernels whose name contains this
            string are recorded; exactly one such kernel per run is expected.

    Returns:
        Average kernel time in milliseconds. Returns a single ``float`` when
        *funcs* contains one entry, otherwise a ``list[float]``. Returns
        ``+inf`` (or a list thereof) if insufficient kernel records were
        collected.
    """
    if not isinstance(funcs, list):
        funcs = [funcs]

    # One JIT warmup call per function before starting the monitor.
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    buffer = _make_l2_cache_buffer() if clear_l2_cache else None

    all_durations: List[int] = []

    def _on_kernel(data) -> None:
        if clear_l2_cache and ("zero" in data.name.lower() or "zeroslike" in data.name.lower()):
            return
        if target_kernel_name is not None and target_kernel_name not in data.name:
            return
        all_durations.append(data.end - data.start)

    total = warmup + active
    monitor = KernelMonitor()
    torch.npu.synchronize()
    monitor.start(_on_kernel)
    try:
        for fn in funcs:
            for _ in builtins.range(total):
                if buffer is not None:
                    buffer.zero_()
                fn()
    finally:
        torch.npu.synchronize()
        monitor.stop()

    return _collect_mspti_result(all_durations, len(funcs), warmup, active, target_kernel_name)


# ---------------------------------------------------------------------------
# torch_npu.profiler backend
# ---------------------------------------------------------------------------


def do_bench_npu_profiler(
    funcs: Union[Callable, List[Callable]],
    warmup: int = 5,
    active: int = 30,
    clear_l2_cache: bool = False,
    prof_dir: Optional[str] = None,
    keep_res: bool = False,
    target_kernel_name: Optional[str] = None,
) -> BenchResult:
    """
    Benchmark NPU kernels using ``torch_npu.profiler``.

    Profiler traces are written to *prof_dir* (or an auto-generated path under
    the Triton cache directory). After parsing, the trace directory is deleted
    unless *keep_res* is True.

    Args:
        funcs: A single callable or a list of callables to benchmark.
        warmup: Number of runs to discard at the start of measurement.
        active: Number of runs whose times are averaged.
        clear_l2_cache: When True, the L2 cache is flushed before every run
            via a buffer read, and the resulting ``ReduceSum`` kernel rows are
            excluded from the timing.
        prof_dir: Directory to write profiler output. Auto-generated when None.
        keep_res: When True, the profiler output directory is not deleted after
            parsing (useful for post-hoc inspection).
        target_kernel_name: When set, only the named kernel's timing is
            returned (uses ``kernel_details.csv`` instead of ``task_time*.csv``).

    Returns:
        Average kernel time in milliseconds. Returns a single ``float`` when
        *funcs* contains one entry, otherwise a ``list[float]``. Returns
        ``+inf`` (or a list thereof) if the profiler output cannot be parsed.
    """
    if not isinstance(funcs, list):
        funcs = [funcs]

    # One JIT warmup call per function before starting the profiler.
    for fn in funcs:
        fn()
        torch.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    torch_path = _make_prof_dir(prof_dir)
    buffer = _make_l2_cache_buffer() if clear_l2_cache else None

    total = warmup + active
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(torch_path),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            with_modules=False,
            experimental_config=experimental_config,
    ) as prof:
        for fn in funcs:
            for _ in builtins.range(total):
                if buffer is not None:
                    buffer.sum()  # flush L2 cache via a read
                    torch.npu.synchronize()
                fn()
                torch.npu.synchronize()
        _ = prof  # suppress F841 "assigned but never used"

    del buffer  # release before parsing to avoid holding memory

    try:
        return _collect_prof_result(
            torch_path,
            funcs,
            warmup,
            active,
            target_kernel_name=target_kernel_name,
            clear_l2_cache=clear_l2_cache,
        )
    finally:
        _remove_dir(torch_path, keep=keep_res)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def do_bench_npu(
    funcs: Union[Callable, List[Callable]],
    warmup: int = 5,
    active: int = 30,
    clear_l2_cache: bool = False,
    prof_dir: Optional[str] = None,
    keep_res: bool = False,
    target_kernel_name: Optional[str] = None,
) -> BenchResult:
    """
    Benchmark one or more NPU kernel functions and return average run times.

    Tries the lightweight ``mspti`` backend first (available when the
    ``mspti`` package is importable and ``target_kernel_name`` is ``None``).
    Falls back to the ``torch_npu.profiler`` backend when mspti is
    unavailable or returns ``+inf``.

    Args:
        funcs: A single callable or a list of callables to benchmark.
        warmup: Number of leading runs to discard before measurement.
        active: Number of runs to average for the reported time.
        clear_l2_cache: Flush the L2 cache before every timed run.
        prof_dir: Profiler output directory (profiler backend only).
        keep_res: Keep the profiler output directory after parsing.
        target_kernel_name: Restrict timing to a single named kernel.

    Returns:
        Average time in milliseconds per function call. A single ``float``
        when *funcs* has one entry, otherwise a ``list[float]``.
    """
    if not isinstance(funcs, list):
        funcs = [funcs]

    use_mspti = KernelMonitor is not None and target_kernel_name is None
    if not use_mspti and KernelMonitor is None:
        print("[WARNING] mspti package not found. Falling back to torch_npu.profiler.")

    if use_mspti:
        try:
            results = do_bench_npu_mspti(funcs, warmup, active, clear_l2_cache, target_kernel_name)
            first_val = results[0] if isinstance(results, list) else results
            if not math.isinf(first_val):
                return results
        except Exception:
            pass

    return do_bench_npu_profiler(funcs, warmup, active, clear_l2_cache, prof_dir, keep_res, target_kernel_name)
