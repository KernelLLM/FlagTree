# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for alltoallv-gmm.py (MTE AllToAllV-GMM).

Correctness path lives in tutorials/ascend/alltoallv-gmm.py; this
script re-uses its kernel and reference functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    HCCL_NPU_SOCKET_PORT_RANGE=auto \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/alltoallv-gmm.py
"""

import json
import os
import statistics

import torch
import shmem as ash
import torch.distributed as dist

from prof import run_options, perf_enabled, verify_enabled, load_tutorial, prof
from utils import init_aclshmem

tut = load_tutorial("alltoallv-gmm")


def _first_warmup_time(fn, warmup: int) -> float:
    """Time the very first warmup call (includes Triton compile) with events."""
    if warmup <= 0:
        return 0.0
    warmup_start = torch.npu.Event(enable_timing=True)
    warmup_end = torch.npu.Event(enable_timing=True)
    warmup_start.record()
    fn()
    warmup_end.record()
    torch.npu.synchronize()
    return warmup_start.elapsed_time(warmup_end)


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    m = int(os.environ.get("TRITON_M", 64))
    k = int(os.environ.get("TRITON_K", 128))
    n = int(os.environ.get("TRITON_N", 128))
    experts_per_rank = int(os.environ.get("TRITON_EXPERTS_PER_RANK", 2))
    dispatch_block_m = min(int(os.environ.get("TRITON_DISPATCH_BLOCK_M", 128)), m)
    dispatch_block_k = min(int(os.environ.get("TRITON_DISPATCH_BLOCK_K", 256)), k)
    gmm_block_m = min(int(os.environ.get("TRITON_GMM_BLOCK_M", 128)), m)
    gmm_block_n = min(int(os.environ.get("TRITON_GMM_BLOCK_N", 256)), n)
    gmm_block_k = min(int(os.environ.get("TRITON_GMM_BLOCK_K", 256)), k)
    iterations = int(os.environ.get("ASCEND_PROFILE_ITERS", 10))
    validation_iterations = int(os.environ.get("TRITON_VALIDATION_ITERS", 1))
    sparse = os.environ.get("TRITON_SPARSE", "1") == "1"
    route_modes = tuple(
        value.strip()
        for value in os.environ.get("TRITON_ROUTE_MODES", "random").split(",")
        if value.strip()
    )
    if len(route_modes) != 1:
        raise ValueError(
            "TRITON_ROUTE_MODES must contain exactly one mode when running the perf script"
        )
    route_mode = route_modes[0]
    if (
        min(
            m,
            k,
            n,
            experts_per_rank,
            dispatch_block_m,
            dispatch_block_k,
            gmm_block_m,
            gmm_block_n,
            gmm_block_k,
        )
        <= 0
    ):
        raise ValueError("shape and block-size variables must be positive")
    if validation_iterations < 0:
        raise ValueError("TRITON_VALIDATION_ITERS must be non-negative")
    if verify_enabled():
        validation_iterations = max(1, validation_iterations)
    else:
        validation_iterations = 0
    if iterations <= 0:
        raise ValueError("ASCEND_PROFILE_ITERS must be positive")
    buffer_count = 2
    local_mem_size = tut.symmetric_memory_size(world_size, m, k, buffer_count)
    if rank == 0:
        print(
            f"ALLTOALLV_GMM_LOCAL_MEM_SIZE_BYTES={local_mem_size}",
            flush=True,
        )
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.MTE)

    receive_buffers = []
    try:
        seed = 20260811
        torch.manual_seed(seed + rank)
        weights = torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()
        for _ in range(buffer_count):
            receive_buffers.append(
                ash.aclshmem_create_tensor(
                    [world_size * m, k], dtype=torch.float16, device_id=local_rank
                )
            )
        output = torch.empty((world_size * m, n), dtype=torch.float16).npu()
        tail_guard = tut.OutputTailGuard.create(world_size * m)
        invocation_count = 0

        def prepare_invocation(
            epoch: int,
        ) -> tuple[
            tut.RouteSample,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
            ],
        ]:
            sample = tut.prepare_route(
                m,
                k,
                world_size,
                experts_per_rank,
                rank,
                epoch,
                route_mode,
                sparse,
            )
            metadata = tut.build_routing_metadata(
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
                dispatch_block_m,
                gmm_block_m,
            )
            return sample, metadata

        def invoke(
            activations: torch.Tensor,
            metadata: tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                int,
                torch.Tensor,
                torch.Tensor,
                int,
                int,
            ],
        ) -> torch.Tensor:
            nonlocal invocation_count
            buffer_index = invocation_count % buffer_count
            invocation_count += 1
            tut.launch_alltoallv_gmm(
                activations,
                weights,
                receive_buffers[buffer_index],
                output,
                rank,
                world_size,
                experts_per_rank,
                dispatch_block_m,
                dispatch_block_k,
                gmm_block_m,
                gmm_block_n,
                gmm_block_k,
                tail_guard,
                metadata,
            )
            return output

        if validation_iterations:
            for iteration in range(validation_iterations):
                sample, metadata = prepare_invocation(iteration)
                actual = invoke(sample.activations, metadata)
                expected = tut._reference_output(
                    sample.activations,
                    weights,
                    sample.token_counts,
                    rank,
                    world_size,
                    experts_per_rank,
                )
                torch.testing.assert_close(actual, expected, rtol=2**-8, atol=2**-8)
                dist.barrier()
                if rank == 0:
                    print(
                        f"AllToAllV-GMM MTE verification iteration {iteration + 1}/{validation_iterations} passed",
                        flush=True,
                    )

        if not perf_enabled():
            return

        sample, metadata = prepare_invocation(validation_iterations)

        @prof(label="alltoallv_gmm_mte")
        def run_kernel():
            return invoke(sample.activations, metadata)

        @prof(label="torch_npu_reference")
        def run_reference():
            return tut._reference_output(
                sample.activations,
                weights,
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
            )

        # The first warmup call carries Triton compile time; report it aside
        # from the @prof records.
        first_warmup_ms = _first_warmup_time(
            lambda: invoke(sample.activations, metadata),
            int(os.environ.get("ASCEND_PROFILE_WARMUP", 3)),
        )
        run_kernel()
        run_reference()
        triton_timings = run_kernel.prof.times_ms
        torch_npu_timings = run_reference.prof.times_ms
        rounds = tut.gather_timings(triton_timings, world_size)
        reference_rounds = tut.gather_timings(torch_npu_timings, world_size)
        if rank != 0:
            return
        round_maxima = [max(values) for values in rounds]
        reference_maxima = [max(values) for values in reference_rounds]
        triton_median = statistics.median(round_maxima)
        torch_npu_median = statistics.median(reference_maxima)
        print(
            f"ALLTOALLV_GMM_MTE_RESULT median_max={triton_median:.4f} ms rounds={round_maxima}",
            flush=True,
        )
        print(
            f"ALLTOALLV_GMM_TORCH_NPU_RESULT median_max={torch_npu_median:.4f} ms",
            flush=True,
        )
        speedup = torch_npu_median / triton_median
        print(f"ALLTOALLV_GMM_TORCH_NPU_SPEEDUP={speedup:.4f}x", flush=True)
        print(f"ALLTOALLV_GMM_FIRST_WARMUP={first_warmup_ms:.4f} ms", flush=True)
        benchmark_report = {
            "raw_timings_ms": rounds,
            "kernel": {
                "median_max_ms": triton_median,
                "round_maxima_ms": round_maxima,
            },
            "torch_npu": {
                "median_max_ms": torch_npu_median,
                "round_maxima_ms": reference_maxima,
            },
            "speedup": speedup,
            "first_warmup_ms": first_warmup_ms,
        }
        benchmark_output = os.environ.get("TRITON_BENCHMARK_JSON", "")
        if benchmark_output:
            with open(benchmark_output, "w", encoding="utf-8") as result_file:
                json.dump(benchmark_report, result_file, indent=2)
                result_file.write("\n")
    finally:
        for receive_buffer in receive_buffers:
            ash.aclshmem_free_tensor(receive_buffer)
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
