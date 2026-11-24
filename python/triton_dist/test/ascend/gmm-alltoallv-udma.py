# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""UDMA GMM-AllToAllV cases from the fixed gmm-alltoallv-cases.csv table.

The correctness path lives in tutorials/ascend/gmm-alltoallv-udma.py;
this script re-uses its one-shot kernel, route helpers, fp32 golden and the
same-semantics torch_npu reference.

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/gmm-alltoallv-udma.py
"""

import itertools
import json
import os
import csv
from pathlib import Path
import statistics
import sys


import shmem as ash
import torch
import torch_npu
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


tut = load_tutorial("gmm-alltoallv-udma")


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


def packing_order(plan):
    """Gather expert-major GMM rows into destination-major send order."""
    segments = [torch.arange(start, start + count, dtype=torch.int64)
                for slices in plan.payload_slices for start, count in slices]
    order = torch.cat(segments)
    # The current output contract keeps routes in global expert order.
    # Incoming writer/local-expert order is exactly that same contiguous order.
    cursor = 0
    for destination, source, count in plan.placement:
        if destination != cursor or source != cursor:
            raise ValueError('reference requires contiguous incoming/output placement')
        cursor += count
    if cursor != sum(plan.receive_split_sizes):
        raise ValueError('reference placement does not cover incoming rows')
    return order


def prepare_reference(receive_buffer, weights, output, plan):
    """Prepare views, indices and communication buffers outside events.

    GMM result allocation, send packing, HCCL and the output copy remain timed.
    The caller provides a zero-initialized output tail, matching the fixed route.
    """

    order = packing_order(plan).to(receive_buffer.device)
    x_list = list(receive_buffer.split(plan.gemm_sizes, dim=0))
    weight_list = list(weights.unbind(0))
    input_splits = list(plan.input_split_sizes)
    receive_splits = list(plan.receive_split_sizes)
    packed = torch.empty((sum(input_splits), output.shape[1]),
                         dtype=output.dtype, device=output.device)
    incoming = torch.empty((sum(receive_splits), output.shape[1]),
                           dtype=output.dtype, device=output.device)
    valid_output = output[:incoming.shape[0]]

    def invoke():
        local_result = torch_npu.npu_grouped_matmul(
            x=x_list, weight=weight_list, group_type=0, split_item=2
        )[0]
        torch.index_select(local_result, 0, order, out=packed)
        # All ranks participate, including ranks with empty local splits.
        dist.all_to_all_single(incoming, packed, output_split_sizes=receive_splits,
                               input_split_sizes=input_splits)
        valid_output.copy_(incoming)
        return output

    return invoke


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


def build_receive_buffer(activations, token_counts, rank, world_size, experts_per_rank):
    # Input rows are sorted by global expert, so each destination is contiguous.
    send_splits, receive_splits, _, order = reference_layout(
        token_counts, rank, world_size, experts_per_rank
    )
    received = torch.empty((sum(receive_splits), activations.shape[1]),
                           dtype=activations.dtype, device=activations.device)
    dist.all_to_all_single(received, activations, output_split_sizes=receive_splits,
                           input_split_sizes=send_splits)
    # Preserve the original expert-major/source-major order, including empty experts.
    return torch.index_select(received, 0, order.to(activations.device))

SEED = 20260811


def weight_seed(case_index: int, owner_rank: int) -> int:
    """Seed of one rank's weights for one case; identical on every rank and pass."""
    return SEED + owner_rank * 1000003 + case_index * 10007


def is_out_of_memory(error: Exception) -> bool:
    """Return whether an allocation failure means the device ran out of memory."""
    text = str(error).lower()
    return "out of memory" in text


def case_fence(
    store,
    tag: str,
    rank: int,
    world_size: int,
    ok: bool,
) -> list[dict]:
    """Publish one rank's outcome for a case, then read every peer's.

    A rank that failed early must not skip this rendezvous: the store-based
    validation of the case waits on keys which only a participating rank writes.
    Carrying the outcome through the store instead of a collective keeps that
    rendezvous independent of the barrier the rounds themselves already use.
    """
    key_prefix = f"gmm_alltoallv_udma/{tag}"
    payload = {"ok": bool(ok)}
    store.set(f"{key_prefix}/{rank}", json.dumps(payload))
    return [json.loads(store.get(f"{key_prefix}/{peer}").decode("utf-8")) for peer in range(world_size)]


def main() -> int:
    run_options()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    tut.require_udma_device(local_rank)
    dist.init_process_group("hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # Create HCCL before ACLSHMEM; input preparation completes before UDMA launches.
    dist.barrier(device_ids=[local_rank])
    store = tut.host_store()

    route_mode = "random"
    cases = select_cases(read_test_cases())
    # The ACLSHMEM heap has to cover the largest *single* case, not the product of
    # the largest M and the largest N: those come from different rows, and their
    # product can exceed the device even when every case fits comfortably.
    heap_case = max(cases, key=lambda case: case[0] * case[3] * case[2])
    heap_m, heap_n = heap_case[0] * heap_case[3], heap_case[2]

    def generate_weights(case_index: int, owner_rank: int, k: int, n: int) -> torch.Tensor:
        torch.manual_seed(weight_seed(case_index, owner_rank))
        return torch.randn((experts_per_rank, k, n), dtype=torch.float16).npu()

    def build_case(case_index: int, m: int, k: int, n: int) -> dict:
        """Prepare reproducible inputs and expert-local rows outside events."""
        gmm_block_m, gmm_block_n, gmm_block_k = tut._effective_tiles(m, k, n)
        weights = generate_weights(case_index, rank, k, n)
        sample = prepare_topk_route(
            tut, m // topk, k, world_size, experts_per_rank, rank, case_index, topk, store
        )
        recv = build_receive_buffer(
            sample.activations, sample.token_counts, rank, world_size, experts_per_rank
        )
        torch.npu.synchronize()
        metadata = tut._build_oneshot_metadata(
            sample.token_counts, rank, world_size, experts_per_rank, gmm_block_m
        )
        ws_offsets, dst_offsets = tut._baseline_offsets(
            sample.token_counts, rank, world_size, experts_per_rank
        )
        baseline_plan = tut.build_baseline_plan(
            sample.token_counts, rank, world_size, experts_per_rank, ws_offsets, dst_offsets
        )
        # The fp32 golden and the all-rank weights it multiplies exist only for the
        # correctness check; --mode=perf must not pay their memory and time.
        expected = None
        if verify_enabled():
            full_weights = torch.cat(
                [generate_weights(case_index, owner, k, n) for owner in range(world_size)], dim=0
            )
            expected = tut.golden_fp32(
                sample.activations,
                full_weights,
                sample.token_counts,
                rank,
                world_size,
                experts_per_rank,
            )
            del full_weights
        return {
            "weights": weights,
            "recv": recv,
            "metadata": metadata,
            "baseline_plan": baseline_plan,
            "expected": expected,
            "gmm_block_m": gmm_block_m,
            "gmm_block_n": gmm_block_n,
            "gmm_block_k": gmm_block_k,
        }

    speedups: list[float] = []
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

    if rank == 0:
        print(
            f"GMM_ALLTOALLV_CASES mode=udma cases={len(cases)} "
            f"route={route_mode} reference=indexed "
            f"heap_case=m{heap_case[0]}_k{heap_case[1]}_n{heap_case[2]} "
            f"timing={'event_timing' if perf_enabled() else 'off'} "
            f"reference={'after_init' if perf_enabled() else 'off'}",
            flush=True,
        )
    # One symmetric heap holds every sampled shape; each case allocates its own
    # output buffer inside it and frees it again.
    local_mem_size = tut.symmetric_memory_size(world_size, heap_m, heap_n)
    if rank == 0:
        print(f"GMM_ALLTOALLV_LOCAL_MEM_SIZE_BYTES={local_mem_size}", flush=True)
    init_aclshmem(rank, world_size, local_mem_size, ash.OpEngineType.UDMA)
    tut.host_barrier(store, "cases_oneshot_init", rank, world_size)

    # host_barrier keys persist in the TCPStore, so every round needs a fresh tag;
    # a reused key would let a fast rank read a stale "1" and pass early.
    barrier_round = itertools.count()

    def udma_barrier() -> None:
        tut.host_barrier(store, f"cases_round_{next(barrier_round)}", rank, world_size)

    # One pass inside the UDMA phase: reference, kernel, validation and the case
    # result line per shape.
    try:
        for case_index, (tokens, k, n, topk, experts_per_rank) in enumerate(cases):
            m = tokens * topk
            output = None
            actual = None
            expected = None
            ok = True
            speedup = None
            reference_actual = reference_invoke = run_reference = None
            if rank == 0:
                print(f"[CASE] {case_index + 1}/{len(cases)} M={tokens} K={k} N={n} topk={topk} E_local={experts_per_rank} routed_rows={m}", flush=True)
            try:
                try:
                    case = build_case(case_index, m, k, n)
                    output = ash.aclshmem_create_tensor(
                        [world_size * m, n], dtype=torch.float16, device_id=local_rank
                    )
                    workspace = torch.empty((world_size * m, n), dtype=torch.float16, device=case["recv"].device)
                    baseline_buffer = torch.zeros((world_size * m, n), dtype=torch.float16, device=case["recv"].device)
                    tail_guard = tut.OutputTailGuard.create(world_size * m)
                    tail_guard.prepare(output, case["metadata"].valid_rows)
                    torch.npu.synchronize()
                    launchers = tut.RoundLaunchers(
                        oneshot=tut.KernelLauncher(tut._gmm_alltoallv_oneshot_kernel),
                    )

                    def invoke(
                        recv=case["recv"],
                        weights=case["weights"],
                        metadata=case["metadata"],
                        block_m=case["gmm_block_m"],
                        block_n=case["gmm_block_n"],
                        block_k=case["gmm_block_k"],
                        output=output,
                        workspace=workspace,
                        tail_guard=tail_guard,
                        launchers=launchers,
                    ):
                        tut.launch_gmm_alltoallv_udma(
                            recv,
                            weights,
                            workspace,
                            output,
                            rank,
                            world_size,
                            block_m,
                            block_n,
                            block_k,
                            metadata=metadata,
                            tail_guard=tail_guard,
                            launchers=launchers,
                        )
                        return output

                    @prof(label="gmm_alltoallv_udma", barrier=udma_barrier)
                    def run_kernel():
                        return invoke()

                    # The mixed part of this variant: the torch_npu reference is an
                    # HCCL operation (grouped matmul + result all-to-all) submitted
                    # while ACLSHMEM UDMA is live, with the default dist.barrier
                    # rounds.  Only the kernel rounds keep the TCPStore barrier.
                    reference_invoke = prepare_reference(
                        case["recv"], case["weights"], baseline_buffer, case["baseline_plan"])

                    @prof(label="torch_npu_reference")
                    def run_reference():
                        return reference_invoke()

                    if perf_enabled() or verify_enabled():
                        reference_actual = run_reference()
                    actual = run_kernel()
                    expected = case["expected"]
                    if perf_enabled():
                        prof_report(run_reference, run_kernel, context=f"m{m}_k{k}_n{n}")
                        speedup = run_reference.prof.mean / run_kernel.prof.mean
                    # The reference buffer is the largest tensor only one of the two
                    # measurements needs; release it before the next case allocates.
                    del case, run_reference, baseline_buffer
                except Exception as error:  # noqa: BLE001 - one shape must not stop the sweep
                    ok = False
                    record_case_error(m, k, n, error)

                # A rank that stopped early skipped its share of the collectives
                # inside build_case, so every rank reports its outcome here before
                # this case is validated: a rank without a result must not leave the
                # others waiting on the store keys they read next.
                payloads = case_fence(
                    store,
                    f"cases_status_{case_index}",
                    rank,
                    world_size,
                    ok,
                )
                if verify_enabled():
                    if all(payload["ok"] for payload in payloads):
                        try:
                            torch.npu.synchronize()
                            tut.validate_against_torch_npu(
                                store, f"reference_validation_{case_index}",
                                reference_actual, expected, rank, world_size, tut.precision_rtol(k))
                            tut.validate_against_torch_npu(
                                store,
                                f"cases_validation_{case_index}",
                                actual,
                                expected,
                                rank,
                                world_size,
                                tut.precision_rtol(k),
                            )
                            if rank == 0:
                                print(f"[PASS] m{m} k{k} n{n}: output matches the fp32 golden", flush=True)
                        except Exception as error:  # noqa: BLE001 - one shape must not stop the sweep
                            ok = False
                            record_case_error(m, k, n, error)
                    elif ok:
                        skipped.append(f"m{m}_k{k}_n{n}: validation skipped, another rank failed the case")

                if perf_enabled():
                    # Aggregate only cases completed successfully on every rank,
                    # including validation when --mode=both. Outside timing.
                    accepted = case_fence(
                        store, f"cases_summary_{case_index}", rank, world_size,
                        ok and all(payload["ok"] for payload in payloads) and speedup is not None,
                    )
                    if rank == 0 and all(payload["ok"] for payload in accepted):
                        speedups.append(speedup)

            finally:
                if output is not None:
                    ash.aclshmem_free_tensor(output)

        if rank == 0:
            print(
                f"GMM_ALLTOALLV_CASES_DONE mode=udma cases={len(cases)} "
                f"failed={len(failures)} skipped={len(skipped)}",
                flush=True,
            )
            if speedups:
                print(
                    f"[AVERAGE] ranks={world_size} cases={len(speedups)} "
                    f"mean_speedup={statistics.mean(speedups):.4f} "
                    f"geomean_speedup={statistics.geometric_mean(speedups):.4f} "
                    "(golden_mean/kernel_mean, rank 0 timings, equal weight per shape)",
                    flush=True,
                )
            else:
                print("[AVERAGE] mean_speedup=N/A geomean_speedup=N/A (no successful timed cases)", flush=True)
    finally:
        ash.aclshmem_finalize()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
