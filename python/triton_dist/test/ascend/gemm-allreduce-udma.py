# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for gemm-allreduce-udma.py (UDMA GEMM with AllReduce).

Reuse the tutorial's kernels and correctness checks with shared NPU event timing:

    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gemm-allreduce-udma.py --mode=both

Read M, K, N from the adjacent allgather-gemm-case.csv using BF16 inputs.
--case=full runs every row; --case=random samples up to ten rows.
Tuned overrides are loaded from gemm-allreduce-udma-config.csv; blank pvalue
fields use the default batching policy after applying the tile override.
"""

import csv
import math
import os
from pathlib import Path

import shmem as ash
import torch
import torch.distributed as dist
from prof import (
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from triton.runtime import driver

tut = load_tutorial("gemm-allreduce-udma")

OWNER_RSAG_MIN_OUTPUT_BYTES = 32 * 1024 * 1024
OWNER_RSAG_MAX_K = 4096
OWNER_RSAG_DEFAULT_PVALUE = 32
OWNER_RSAG_PIPELINE_MIN_K = 2048
OWNER_RSAG_PIPELINE_TARGET_BATCHES = 4


def read_test_cases(csv_path):
    """Read M, K, N shapes in CSV order, matching the AllGather-GEMM sweep."""
    with open(csv_path, newline="", encoding="utf-8") as file:
        return [
            (int(row["M"]), int(row["K"]), int(row["N"]))
            for row in csv.DictReader(file)
        ]


def read_tuned_configs(csv_path):
    """Load per-algorithm tile overrides; a blank pvalue uses default batching."""
    configs = {}
    with open(csv_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {
            "M", "K", "N", "algorithm", "BLOCK_SIZE_M", "BLOCK_SIZE_N",
            "BLOCK_SIZE_K", "pvalue",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path}: missing tuning configuration columns")
        for line, row in enumerate(reader, start=2):
            try:
                shape = tuple(int(row[name]) for name in ("M", "K", "N"))
                tiles = tuple(
                    int(row[name])
                    for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")
                )
                algorithm = row["algorithm"].strip()
                pvalue_text = row["pvalue"].strip()
                pvalue = int(pvalue_text) if pvalue_text else None
                if algorithm not in ("one-shot", "owner-rsag"):
                    raise ValueError("unknown algorithm")
                if min(*shape, *tiles) <= 0 or (pvalue is not None and pvalue <= 0):
                    raise ValueError("dimensions and configuration values must be positive")
                key = (algorithm, *shape)
                if key in configs:
                    raise ValueError("duplicate algorithm and shape")
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"{csv_path}:{line}: {error}") from error
            configs[key] = (*tiles, pvalue)
    return configs


def _adjust_oneshot_configuration(M, K, N, v_core, tuned_configs):
    """Select the default tile sizes and communication batch size."""
    BLOCK_SIZE_M = min(128, M)
    if N >= 256:
        BLOCK_SIZE_N = 256
    elif N >= 128:
        BLOCK_SIZE_N = 128
    else:
        # Preserve narrow-N shapes without introducing mostly-padding tiles.
        BLOCK_SIZE_N = N

    if K >= 32768:
        BLOCK_SIZE_K = 128
    else:
        BLOCK_SIZE_K = min(256, K)

    # Preserve the previously measured narrow-K special case.
    if M == 65536 and K == 128:
        BLOCK_SIZE_M = min(64, M)
        BLOCK_SIZE_N = min(64, N)

    tuned_tile = tuned_configs.get(("one-shot", M, K, N))
    if tuned_tile is not None:
        BLOCK_SIZE_M = min(tuned_tile[0], M)
        BLOCK_SIZE_N = min(tuned_tile[1], N)
        BLOCK_SIZE_K = min(tuned_tile[2], K)

    # Derive batching only after the GEMM tile has reached its final value.
    total_tiles = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    if tuned_tile is not None and tuned_tile[3] is not None:
        pvalue = tuned_tile[3]
    elif M == 65536 and K == 128:
        pvalue = 32
    elif K >= 32768 or K <= 128:
        pvalue = 8
    else:
        # With at most two waves of GEMM tiles, p=1 preserved the useful extra
        # communication batch measured for 512x8192x4096.
        pvalue = 1 if total_tiles <= 2 * v_core else 2

    REDUCE_BLOCK_SIZE_M = min(128, BLOCK_SIZE_M)
    REDUCE_BLOCK_SIZE_N = min(128, BLOCK_SIZE_N)
    if BLOCK_SIZE_M % REDUCE_BLOCK_SIZE_M != 0:
        raise ValueError(
            "BLOCK_SIZE_M must be divisible by REDUCE_BLOCK_SIZE_M, got "
            f"{BLOCK_SIZE_M} and {REDUCE_BLOCK_SIZE_M}"
        )
    if BLOCK_SIZE_N % REDUCE_BLOCK_SIZE_N != 0:
        raise ValueError(
            "BLOCK_SIZE_N must be divisible by REDUCE_BLOCK_SIZE_N, got "
            f"{BLOCK_SIZE_N} and {REDUCE_BLOCK_SIZE_N}"
        )
    return (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
        pvalue,
    )


def _adjust_owner_rsag_configuration(M, K, N, v_core, tuned_configs):
    """Select the default tile sizes and communication batch size."""
    BLOCK_SIZE_M = min(128, M)
    if N >= 256:
        BLOCK_SIZE_N = 256
    elif N >= 128:
        BLOCK_SIZE_N = 128
    else:
        # Preserve narrow-N shapes without introducing mostly-padding tiles.
        BLOCK_SIZE_N = N

    if K >= 32768:
        BLOCK_SIZE_K = 128
    else:
        BLOCK_SIZE_K = min(256, K)

    # Preserve the previously measured narrow-K special case.
    if M == 65536 and K == 128:
        BLOCK_SIZE_M = min(64, M)
        BLOCK_SIZE_N = min(64, N)

    measured_config = tuned_configs.get(("owner-rsag", M, K, N))
    if measured_config is not None:
        BLOCK_SIZE_M = min(measured_config[0], M)
        BLOCK_SIZE_N = min(measured_config[1], N)
        BLOCK_SIZE_K = min(measured_config[2], K)

    # Derive batching only after the GEMM tile has reached its final value.
    total_tiles = math.ceil(M / BLOCK_SIZE_M) * math.ceil(N / BLOCK_SIZE_N)
    output_bytes = M * N * 2
    if measured_config is not None and measured_config[3] is not None:
        pvalue = measured_config[3]
    elif K <= OWNER_RSAG_MAX_K and output_bytes >= OWNER_RSAG_MIN_OUTPUT_BYTES:
        # Owner RSAG benefits from fewer, larger contiguous UDMA transfers.
        # With 32 AICs and four ranks, p=32 gives about 256 tiles per owner.
        pvalue = OWNER_RSAG_DEFAULT_PVALUE
    elif (
        OWNER_RSAG_PIPELINE_MIN_K <= K <= OWNER_RSAG_MAX_K
        and output_bytes >= tut.HYBRID_RSAG_MIN_OUTPUT_BYTES
    ):
        # For sub-32 MB, high-K outputs, retain about four batches so the
        # asynchronous RS put for batch b can overlap GEMM batch b+1.  This
        # yields p=1/2/4 for the measured 56-512 tile range instead of the
        # discontinuous p=1/2 fallback immediately below 32 MB.
        pvalue = max(
            1,
            math.ceil(total_tiles / (OWNER_RSAG_PIPELINE_TARGET_BATCHES * v_core)),
        )
    elif K >= 32768 or K <= 128:
        pvalue = 8
    else:
        # With at most two waves of GEMM tiles, p=1 preserved the useful extra
        # communication batch measured for 512x8192x4096.
        pvalue = 1 if total_tiles <= 2 * v_core else 2

    REDUCE_BLOCK_SIZE_M = min(128, BLOCK_SIZE_M)
    REDUCE_BLOCK_SIZE_N = min(128, BLOCK_SIZE_N)
    if BLOCK_SIZE_M % REDUCE_BLOCK_SIZE_M != 0:
        raise ValueError(
            "BLOCK_SIZE_M must be divisible by REDUCE_BLOCK_SIZE_M, got "
            f"{BLOCK_SIZE_M} and {REDUCE_BLOCK_SIZE_M}"
        )
    if BLOCK_SIZE_N % REDUCE_BLOCK_SIZE_N != 0:
        raise ValueError(
            "BLOCK_SIZE_N must be divisible by REDUCE_BLOCK_SIZE_N, got "
            f"{BLOCK_SIZE_N} and {REDUCE_BLOCK_SIZE_N}"
        )
    return (
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        REDUCE_BLOCK_SIZE_M,
        REDUCE_BLOCK_SIZE_N,
        pvalue,
    )


def adjust_configuration(M, K, N, v_core, tuned_configs):
    """Use each collective's own existing configuration policy."""
    if tut.use_owner_rsag(M, K, N):
        return _adjust_owner_rsag_configuration(M, K, N, v_core, tuned_configs)
    return _adjust_oneshot_configuration(M, K, N, v_core, tuned_configs)


def run_test_case(M, K, N, pe, world_size, local_pe, ncore, tuned_configs):
    dtype = torch.bfloat16
    config = adjust_configuration(M, K, N, ncore, tuned_configs)
    block_m, block_n, block_k, reduce_m, reduce_n, pvalue = config
    peer_mem_size = tut.peer_memory_size(M, K, N, world_size, ncore, config)
    peer_mem = None
    try:
        peer_mem = ash.aclshmem_create_tensor(
            [peer_mem_size], dtype=dtype, device_id=local_pe
        )
        A_cpu, B_cpu = tut.generate_matrices_for_rank(pe, M, N, K, dtype)
        A_rank, B_rank = A_cpu.npu(), B_cpu.npu()
        C = torch.zeros([M, N], dtype=dtype).npu()

        @prof(label="gemm_allreduce_udma")
        def run_kernel():
            tut.gemm_allreduce(
                A_rank,
                B_rank,
                C,
                peer_mem,
                pe,
                world_size,
                pvalue,
                block_m,
                block_n,
                block_k,
                reduce_m,
                reduce_n,
            )

        @prof(label="torch_gemm_allreduce")
        def run_golden():
            return tut.torch_gemm_allreduce(A_rank, B_rank)

        C_golden = run_golden()
        run_kernel()
        prof_report(
            run_golden,
            run_kernel,
            context=f"M={M} K={K} N={N} algorithm={tut.collective_name(M, K, N)}",
        )
        if verify_enabled():
            tut.check_accuracy(C_golden, C, pe, world_size, M, K, N)
    finally:
        if peer_mem is not None:
            ash.aclshmem_free_tensor(peer_mem)


def main():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    local_pe = int(os.environ["LOCAL_RANK"])
    ncore = driver.active.utils.get_aicore_num()
    test_cases = read_test_cases(Path(__file__).with_name("allgather-gemm-case.csv"))
    tuned_configs = read_tuned_configs(
        Path(__file__).with_name("gemm-allreduce-udma-config.csv")
    )
    selected_cases = select_cases(test_cases)
    if pe == 0:
        print(
            f"[INFO] Selected test cases: {len(selected_cases)}/{len(test_cases)}",
            flush=True,
        )

    # The bf16 peer arena is the only symmetric allocation; size the heap for
    # the worst (case, tuned config) pair the sweep runs.
    max_required = 0
    for M, K, N in selected_cases:
        config = adjust_configuration(M, K, N, ncore, tuned_configs)
        required = tut.peer_memory_size(M, K, N, world_size, ncore, config) * 2
        max_required = max(max_required, required)
    tut.shmem_init(pe, world_size, tut._ash_pool_bytes(max_required))
    try:
        for M, K, N in selected_cases:
            run_test_case(M, K, N, pe, world_size, local_pe, ncore, tuned_configs)
    finally:
        _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    run_options()
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl")
    pe = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {pe} of {world_size} initialized")
    dist.barrier()
    try:
        main()
        if pe == 0:
            print(f"[INFO] Perf run passed successfully for {world_size} ranks!")
    finally:
        dist.destroy_process_group()
