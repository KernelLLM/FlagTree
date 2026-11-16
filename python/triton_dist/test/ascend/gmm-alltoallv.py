# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""MTE GMM-AllToAllV cases driven by gmm-alltoallv-case.csv.

The correctness path lives in tutorials/ascend/gmm-alltoallv.py; this
script re-uses its kernels, route helpers, fp32 golden and the same-semantics
torch_npu reference, and adds the shared verification/performance protocol:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gmm-alltoallv.py
"""

import csv
import os
import sys
from pathlib import Path

import shmem as ash
import torch
import torch.distributed as dist
from prof import (
    load_tutorial,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from utils import init_aclshmem

tut = load_tutorial("gmm-alltoallv")


CASE_CSV = Path(__file__).resolve().with_name("gmm-alltoallv-case.csv")


def read_test_cases(path: Path) -> list[tuple[int, int, int]]:
    """Read the M, K, N shape table of this operator."""
    cases = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            cases.append((int(row["M"]), int(row["K"]), int(row["N"])))
    if not cases:
        raise ValueError(f"{path} has no cases")
    return cases


def is_out_of_memory(error: Exception) -> bool:
    """Return whether an allocation failure means the device ran out of memory."""
    text = str(error).lower()
    return "out of memory" in text


def main() -> int:
    run_options()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    store = tut.host_store()

    experts_per_rank = int(os.environ.get("TRITON_EXPERTS_PER_RANK", "2"))
    sparse = tut._env_flag("TRITON_SPARSE", "1")
    route_modes = tuple(
        value.strip()
        for value in os.environ.get("TRITON_ROUTE_MODES", "random").split(",")
        if value.strip()
    )
    if len(route_modes) != 1:
        raise ValueError("TRITON_ROUTE_MODES must contain exactly one mode")
    route_mode = route_modes[0]
    cases = select_cases(read_test_cases(CASE_CSV))
    # The ACLSHMEM heap has to cover the largest *single* case, not the product of
    # the largest M and the largest N: those come from different rows, and their
    # product can exceed the device even when every case fits comfortably.
    heap_case = max(cases, key=lambda case: case[0] * case[2])

    # One symmetric heap holds every sampled shape; each case allocates its own
    # output buffer inside it and frees it again.
    local_mem_size = tut.symmetric_memory_size(world_size, heap_case[0], heap_case[2])
    if rank == 0:
        print(f"GMM_ALLTOALLV_LOCAL_MEM_SIZE_BYTES={local_mem_size}", flush=True)
        print(
            f"GMM_ALLTOALLV_CASES mode=mte cases={len(cases)} "
            f"route={route_mode} sparse={int(sparse)} "
            f"heap_case=m{heap_case[0]}_k{heap_case[1]}_n{heap_case[2]} "
            f"timing={'event_timing' if perf_enabled() else 'off'}",
            flush=True,
        )
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.MTE)

    failures: list[str] = []
    skipped: list[str] = []

    def record_case_error(m: int, k: int, n: int, error: Exception) -> None:
        """Report one failed shape once, on the rank that hit it, and keep going.

        A shape can fail on some ranks and not on others, so the line carries the
        rank and is printed where the failure happened instead of being repeated
        by every rank at the end of the sweep.
        """
        case_id = f"m{m}_k{k}_n{n}"
        if isinstance(error, RuntimeError) and is_out_of_memory(error):
            skipped.append(f"{case_id}: {error}")
            print(f"SKIP (OOM) rank={rank} m={m} k={k} n={n}", flush=True)
            return
        failures.append(f"{case_id}: {type(error).__name__}: {error}")
        print(f"GMM_ALLTOALLV_FAILED rank={rank} m={m} k={k} n={n}: {error}", flush=True)

    seed = 20260811
    try:
        torch.manual_seed(seed + rank)
        for case_index, (m, k, n) in enumerate(cases):
            output = None
            actual = None
            expected = None
            ok = True
            if rank == 0:
                print(f"[CASE] {case_index + 1}/{len(cases)} m={m} k={k} n={n}", flush=True)
            try:
                try:
                    gmm_block_m, gmm_block_n, gmm_block_k = tut.effective_tiles(m, k, n)
                    # Every sampled shape needs its own weights: the grouped GEMM's N
                    # changes with the shape.
                    weights = torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()
                    sample = tut.prepare_route(
                        m, k, world_size, experts_per_rank, rank, case_index, route_mode, store, sparse
                    )
                    device_token_counts = sample.token_counts.npu()
                    recv = tut.build_receive_buffer(
                        sample.activations, sample.token_counts, rank, world_size, experts_per_rank
                    )
                    gmm_metadata = tut.build_gmm_metadata(
                        sample.token_counts, rank, experts_per_rank, gmm_block_m
                    )
                    scatter_metadata = tut.build_result_scatter_metadata(
                        sample.token_counts, rank, world_size, experts_per_rank
                    )
                    baseline_plan = tut.build_baseline_plan(
                        sample.token_counts,
                        rank,
                        world_size,
                        experts_per_rank,
                        scatter_metadata[1],
                        scatter_metadata[2],
                    )
                    # The fp32 golden and the all-rank weights it multiplies exist only
                    # for the correctness check; --mode=perf must not pay their memory
                    # and time.
                    expected = None
                    if verify_enabled():
                        full_weights_list = [torch.empty_like(weights) for _ in range(world_size)]
                        dist.all_gather(full_weights_list, weights)
                        full_weights = torch.cat(full_weights_list, dim=0)
                        expected = tut.golden_fp32(
                            sample.activations,
                            full_weights,
                            sample.token_counts,
                            rank,
                            world_size,
                            experts_per_rank,
                        )
                        del full_weights, full_weights_list
                    output = ash.aclshmem_create_tensor(
                        [world_size * m, n], dtype=torch.float16, device_id=local_rank
                    )
                    tail_guard = tut.OutputTailGuard.create(world_size * m)
                    launchers = tut.RoundLaunchers(compute=tut.KernelLauncher(tut._grouped_matmul_scatter_kernel))

                    # Bind the per-case tensors as defaults: the closures must not
                    # capture the loop variables themselves.
                    def invoke(
                        recv=recv,
                        weights=weights,
                        output=output,
                        token_counts=device_token_counts,
                        gmm_metadata=gmm_metadata,
                        scatter_metadata=scatter_metadata,
                        tail_guard=tail_guard,
                        launchers=launchers,
                        gmm_block_m=gmm_block_m,
                        gmm_block_n=gmm_block_n,
                        gmm_block_k=gmm_block_k,
                    ):
                        tut.launch_gmm_alltoallv(
                            recv,
                            weights,
                            output,
                            token_counts,
                            rank,
                            world_size,
                            experts_per_rank,
                            gmm_block_m,
                            gmm_block_n,
                            gmm_block_k,
                            gmm_metadata,
                            scatter_metadata,
                            tail_guard,
                            launchers,
                        )
                        return output

                    @prof(label="gmm_alltoallv_mte")
                    def run_kernel(run=invoke):
                        return run()

                    @prof(label="torch_npu_reference")
                    def run_reference(
                        recv=recv, weights=weights, output=output, baseline_plan=baseline_plan
                    ):
                        return tut.baseline_output(recv, weights, output, baseline_plan)

                    if perf_enabled():
                        run_reference()
                    actual = run_kernel()
                    if perf_enabled():
                        prof_report(run_reference, run_kernel, context=f"m{m}_k{k}_n{n}")
                except Exception as error:  # noqa: BLE001 - one shape must not stop the sweep
                    ok = False
                    record_case_error(m, k, n, error)

                if verify_enabled() and ok:
                    try:
                        torch.npu.synchronize()
                        tut.assert_precision(actual, expected, tut.precision_rtol(k))
                        if rank == 0:
                            print(f"[PASS] m{m} k{k} n{n}: output matches the fp32 golden", flush=True)
                    except Exception as error:  # noqa: BLE001 - one shape must not stop the sweep
                        record_case_error(m, k, n, error)

                # Every rank joins this rendezvous even when its own case failed, so
                # a failing shape cannot leave the others waiting in the next one.
                dist.barrier()
            except Exception as error:  # noqa: BLE001 - one shape must not stop the sweep
                record_case_error(m, k, n, error)
                try:
                    dist.barrier()
                except Exception as barrier_error:  # noqa: BLE001 - keep the remaining cases running
                    if rank == 0:
                        print(
                            f"GMM_ALLTOALLV_WARNING m={m} k={k} n={n} resync failed: "
                            f"{type(barrier_error).__name__}: {barrier_error}",
                            flush=True,
                        )
            finally:
                if output is not None:
                    ash.aclshmem_free_tensor(output)

        if rank == 0:
            print(
                f"GMM_ALLTOALLV_CASES_DONE mode=mte cases={len(cases)} "
                f"failed={len(failures)} skipped={len(skipped)}",
                flush=True,
            )
    finally:
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
