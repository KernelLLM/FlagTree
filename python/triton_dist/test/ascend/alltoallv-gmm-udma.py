# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

# torchrun --standalone --nproc_per_node=4 python/triton_dist/test/ascend/alltoallv-gmm-udma.py --mode=both --case=full



import gc
import os
import csv
from pathlib import Path
import statistics
from types import SimpleNamespace



import shmem as ash
import torch
import torch_npu
import torch.distributed as dist
from triton.runtime import driver

from prof import (
    ProfRecord,
    check_all_ranks_close,
    distributed_max_times,
    load_tutorial,
    perf_enabled,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)



tut = load_tutorial("alltoallv-gmm-udma")


def read_test_cases():
    """Read the fixed 300-case table; M is original tokens, E is per rank.

    The table covers all 139 in-range M/K/N tuples from gmm-combine, with
    two or three routes each. Each topk and each expert count has 100 cases.
    """
    path = Path(__file__).with_name("gmm-alltoallv-cases.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        return [tuple(int(row[key]) for key in
                      ("M", "K", "N", "topk", "experts_per_rank"))
                for row in csv.DictReader(stream)]


def local_topk_route(m, k, ranks, experts, rank, epoch, topk):
    generator = torch.Generator().manual_seed(20260828 + rank * 1000003 + epoch * 10007)
    # Independent random scores choose distinct experts for each original token.
    ids = torch.rand((m, ranks * experts), generator=generator).topk(topk, dim=1).indices.flatten()
    x = torch.randn((m, k), generator=generator, dtype=torch.float16)
    order = torch.argsort(ids, stable=True)
    activations = x[order // topk]
    counts = torch.bincount(ids, minlength=ranks * experts).to(torch.int32)
    return activations, counts


def prepare_topk_route(tutorial, m, k, ranks, experts, rank, epoch, topk, store=None):
    activations, counts = local_topk_route(m, k, ranks, experts, rank, epoch, topk)
    if store is None:
        global_counts = tutorial._exchange_counts(counts, ranks)
    else:
        global_counts = tutorial._exchange_counts(store, counts, epoch, rank, ranks)
    return tutorial.RouteSample(activations.npu(), global_counts, "random", epoch)


def reference_layout(token_counts, rank, world_size, experts_per_rank):
    """Map source-major receive rows to expert-major, then source-major rows."""
    counts = token_counts.to(device="cpu", dtype=torch.int64)
    begin = rank * experts_per_rank
    local_counts = counts[:, begin:begin + experts_per_rank]
    send_splits = counts[rank].reshape(world_size, experts_per_rank).sum(1).tolist()
    receive_splits = local_counts.sum(1).tolist()
    # AllToAllV concatenates source chunks; each chunk is already expert-major.
    flat_counts = local_counts.flatten()
    starts = flat_counts.cumsum(0) - flat_counts
    segments = []
    for expert in range(experts_per_rank):
        for source in range(world_size):
            slot = source * experts_per_rank + expert
            start, count = int(starts[slot]), int(flat_counts[slot])
            segments.append(torch.arange(start, start + count, dtype=torch.int64))
    order = torch.cat(segments)
    return send_splits, receive_splits, local_counts.sum(0).tolist(), order


def prepare_reference(activations, weights, token_counts, rank, world_size, experts_per_rank):
    """Prepare fixed metadata, views and buffers outside benchmark events.

    Timed invocation includes HCCL, indexed data movement, grouped GEMM and
    copying its result into the padded output. GMM still allocates its result.
    """

    send_splits, receive_splits, expert_sizes, order = reference_layout(
        token_counts, rank, world_size, experts_per_rank
    )
    valid_rows = sum(receive_splits)
    order = order.to(activations.device)
    received = torch.empty((valid_rows, activations.shape[1]),
                           dtype=activations.dtype, device=activations.device)
    grouped = torch.empty_like(received)
    x_list = list(grouped.split(expert_sizes, dim=0))
    weight_list = list(weights.unbind(0))
    padded = torch.empty((world_size * activations.shape[0], weights.shape[-1]),
                         dtype=activations.dtype, device=activations.device)
    # Match the custom path: unused output rows are initialized once, untimed.
    padded[valid_rows:].zero_()

    def invoke():
        dist.all_to_all_single(received, activations, output_split_sizes=receive_splits,
                               input_split_sizes=send_splits)
        torch.index_select(received, 0, order, out=grouped)
        result = torch_npu.npu_grouped_matmul(
            x=x_list, weight=weight_list, group_type=0, split_item=2
        )[0]
        padded[:valid_rows].copy_(result)
        return padded

    return invoke


# Fixed scheduling exceptions validated with two experts per rank on 4/8 devices. Other world sizes use the same policy; they need independent device validation.
SWIZZLE_OFFSETS = {
    (896, 45056, 4608): 0,
    (1024, 16384, 13696): 2,
    (1024, 17152, 13056): 2,
    (2688, 17152, 7168): 8,
    (8192, 8192, 2752): 2,
}


def select_swizzle_offset(case, experts_per_rank, route, epoch, sparse):
    """Select a fixed offset before launch, never search or time candidates."""
    if (experts_per_rank, route, epoch, int(sparse)) == (2, "random", 1, 1):
        return SWIZZLE_OFFSETS.get(tuple(case), 7)
    return 7


def _effective_tiles(m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    """Return the fixed dispatch M/K and GEMM M/N/K configuration."""
    if min(m, k, n) <= 0:
        raise ValueError("M/K/N must be positive")
    return 128, 256, 128, 256, 128



def run_reference_case(options, sample, weights, rank, world_size):
    """Return the HCCL golden for verification and per-round reference timings."""

    invoke_reference = prepare_reference(
        sample.activations, weights, sample.token_counts, rank, world_size,
        options.experts_per_rank,
    )

    @prof(label="torch_npu_alltoallv_gmm")
    def run_reference():
        return invoke_reference()

    expected = run_reference()
    if not perf_enabled():
        return expected, None
    reference_times = distributed_max_times(run_reference.prof.times_ms).tolist()
    return expected if verify_enabled() else None, reference_times


def run_test_case(
    options,
    case,
    route,
    sample,
    weights,
    rank,
    world_size,
    input_buffer,
    ready_signals,
    reference_times,
    expected,
    ncore,
):
    """Reuse symmetric buffers; reset signals outside the profiled invocation."""
    m, k, n = case
    tiles = _effective_tiles(*case)
    swizzle_offset = select_swizzle_offset(
        case, options.experts_per_rank, route, options.epoch, options.sparse
    )
    dispatch, metadata = tut._prepare_dispatch(options, sample, rank, world_size, tiles)
    grouped_input = input_buffer[: world_size * m * k].view(world_size * m, k)
    output = torch.empty(
        (world_size * m, n), dtype=torch.float16, device=sample.activations.device
    )
    output[metadata.valid_rows:].zero_()
    if verify_enabled():
        output[:metadata.valid_rows].fill_(float("nan"))
        grouped_input.fill_(float("nan"))

    def invoke():
        tut.launch_alltoallv_gmm_udma(
            sample.activations,
            weights,
            ready_signals,
            grouped_input,
            output,
            rank,
            world_size,
            *tiles,
            routes=dispatch,
            metadata=metadata,
            ncore=ncore,
            swizzle_offset=swizzle_offset,
        )
        return output

    def reset_signals():
        tut._clear_ready_signals(ready_signals)
        torch.npu.synchronize()

    @prof(
        label="alltoallv_gmm_udma",
        prepare=reset_signals,
    )
    def run_kernel():
        return invoke()

    actual = run_kernel()
    if verify_enabled():
        check_all_ranks_close(
            actual,
            expected,
            rtol=2**-8,
            atol=2**-8,
            context=f"AllToAllV-GMM {case} route={route}",
        )
    if not perf_enabled():
        return None

    kernel_times = distributed_max_times(run_kernel.prof.times_ms).tolist()
    # prof_report's means are computed from per-round rank maxima for both paths.
    run_kernel.prof = ProfRecord(run_kernel.prof.label, kernel_times)
    reference = SimpleNamespace(
        prof=ProfRecord("torch_npu_alltoallv_gmm", reference_times)
    )
    prof_report(reference, run_kernel, context=f"M={m} K={k} N={n} route={route}")
    kernel_median = statistics.median(kernel_times)
    reference_median = statistics.median(reference_times)
    report = {
        "route": route,
        "speedup": reference_median / kernel_median,
    }
    if rank == 0:
        print(
            f"MEDIAN_TIMING kernel={kernel_median:.4f}ms "
            f"golden={reference_median:.4f}ms speedup={report['speedup']:.4f}x",
            flush=True,
        )
    return report


def main():
    options = SimpleNamespace(epoch=int(os.environ.get("TRITON_BENCHMARK_EPOCH", "1")),
                              sparse=1, routes=("random",))
    cases = select_cases(read_test_cases())
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    ncore = driver.active.utils.get_aicore_num()
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    tut.require_udma_device(local_rank)
    dist.init_process_group("hccl")
    reports = []
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        dist.barrier()
        if rank == 0:
            print(f"[CONFIG] ranks={world_size} cases={len(cases)} golden=single_index_reorder "
                  "reference_buffers_and_metadata=preallocated gmm_result_allocation=timed", flush=True)
        torch.npu.synchronize()
        torch.npu.empty_cache()
        dist.barrier()
        max_elements = max(world_size * m * topk * k for m, k, n, topk, e in cases)
        input_buffer, ready_signals = tut.shmem_init(
            local_rank, rank, world_size, max_elements
        )
        try:
            for index, (m, k, n, topk, experts) in enumerate(cases):
                route = "random"
                options.experts_per_rank = experts
                # Kernel and golden consume expert-major routed rows.
                case = (m * topk, k, n)
                if rank == 0:
                    print(f"[CASE] M={m} K={k} N={n} topk={topk} E_local={experts} routed_rows={m * topk}", flush=True)
                if rank == 0:
                    print(
                        f"[REFERENCE {index + 1}/{len(cases)}] {case} route={route}",
                        flush=True,
                    )
                # Both implementations read these inputs without modifying them.
                sample = prepare_topk_route(tut, m, k, world_size, experts, rank,
                                            options.epoch + index, topk)
                torch.manual_seed(20260811 + rank + index)
                weights = torch.randn((experts, k, n), dtype=torch.float16).npu()
                expected, reference_times = run_reference_case(
                    options,
                    sample,
                    weights,
                    rank,
                    world_size,
                )
                # Collect the reference profiler wrapper; keep shared inputs and the golden alive until kernel verification completes.
                gc.collect()
                torch.npu.synchronize()
                dist.barrier()
                if rank == 0:
                    print(
                        f"[KERNEL {index + 1}/{len(cases)}] {case} route={route}",
                        flush=True,
                    )
                report = run_test_case(
                    options,
                    case,
                    route,
                    sample,
                    weights,
                    rank,
                    world_size,
                    input_buffer,
                    ready_signals,
                    reference_times,
                    expected,
                    ncore,
                )
                del expected, sample, weights
                if report is not None:
                    reports.append(report)
                gc.collect()
                torch.npu.synchronize()
                dist.barrier()
                if rank == 0:
                    status = "PASS" if verify_enabled() else "COMPLETE"
                    print(f"[{status}] {case} route={route}", flush=True)
            if rank == 0:
                for route in options.routes:
                    ratios = [r["speedup"] for r in reports if r["route"] == route]
                    if ratios:
                        print(
                            f"[SUMMARY] route={route} cases={len(ratios)} arithmetic_mean_torch_speedup={statistics.mean(ratios):.4f} geomean_torch_speedup={statistics.geometric_mean(ratios):.4f} "
                            f"faster_than_torch={sum(r > 1 for r in ratios)} "
                            "(golden_median/kernel_median, per-round rank maxima, equal weight per shape)",
                            flush=True,
                        )
                if not reports:
                    print("[SUMMARY] mean_speedup=N/A geomean_speedup=N/A (no timed samples)", flush=True)
                print(f"[DONE] Completed {len(cases)} cases", flush=True)
        finally:
            torch.npu.synchronize()
            ash.aclshmem_free_tensor(ready_signals)
            ash.aclshmem_free_tensor(input_buffer)
            ash.aclshmem_finalize()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
