# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

r"""Experimental mixed Cube/Vector FC2, UDMA return and token combine.

GMM, send and reduce share one mixed kernel launch. Routing metadata and
the final cross-rank acknowledgment use separate kernels. Reduction waits
for all required rows; GMM and UDMA overlap within the main kernel.

Example for case002 (four NPUs):
    ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    HCCL_NPU_SOCKET_PORT_RANGE=auto \
    torchrun --standalone --nproc-per-node=4 \
        tutorials/ascend/gmm-combine-udma.py \
        --tokens-per-rank 2040 --hidden-size 1536 \
        --ffn-size 480 --topk 8 --num-experts 640
"""

from __future__ import annotations


import argparse
import math
import os
from functools import lru_cache
from typing import Callable, Dict, NamedTuple, Optional

import shmem as ash
import torch
import torch_npu  # noqa: F401
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton.runtime import driver
import triton.language.extra.cann.extension as ext
from triton_dist.language.extra import libshmem_device
from utils import ACLSHMEM_MAX_LOCAL_SIZE, init_aclshmem

SIG_STRIDE = 16  # int32 elements per 64-byte signal slot.


# Verified pipeline configuration. These are implementation details rather than
# user-facing inputs, so they deliberately have no command-line or environment
# variable override.
PIPELINE_BATCH_ROWS = 2048
GATHER_TOKEN_BLOCK = 4
PIPELINE_PUT_BYTES = 4 * 1024 * 1024


ACCURACY_RTOL = 3e-2


ACCURACY_ATOL = 3e-2


@lru_cache(maxsize=1)
def _aicore_num() -> int:
    """Query the local Cube core count once per process."""
    return driver.active.utils.get_aicore_num()


@lru_cache(maxsize=1)
def _aivector_core_num() -> int:
    """Query the local Vector core count once per process."""
    return driver.active.utils.get_aivector_core_num()


class GmmCombineShape(NamedTuple):
    name: str
    tokens_per_rank: int
    hidden_size: int
    ffn_size: int
    topk: int
    num_experts: int


def all_to_all(group, inputs, output_split_sizes=None, input_split_sizes=None):
    if dist.get_world_size(group=group) == 1:
        return inputs

    inputs = inputs.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(inputs)
    else:
        output = torch.empty(
            (sum(output_split_sizes), inputs.size(1)),
            dtype=inputs.dtype,
            device=inputs.device,
        )
    dist.all_to_all_single(
        output,
        inputs,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def ugly_concat_list(splits_list, inputs, nx, ny):
    inputs_list = torch.split(inputs, splits_list)
    outputs_list = []
    for idy in range(nx):
        for idx in range(ny):
            outputs_list.append(inputs_list[idx * nx + idy])
    return torch.cat(outputs_list, dim=0)


def torch_native_scatter_backward(
    expert_output: torch.Tensor,
    scatter_index: torch.Tensor,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
) -> torch.Tensor:
    """Restore original route order and sum the top-k expert outputs."""

    if start_idx is None and end_idx is None:
        in_range = torch.ones_like(scatter_index, dtype=torch.bool)
        effective_scatter_index = scatter_index
    else:
        if start_idx is None or end_idx is None:
            raise ValueError(
                "start_idx and end_idx must be both set or both omitted"
            )
        if start_idx == end_idx:
            return torch.zeros(
                (scatter_index.shape[0], expert_output.shape[-1]),
                device=expert_output.device,
                dtype=expert_output.dtype,
            )
        in_range = (scatter_index >= start_idx) & (
            scatter_index < end_idx
        )
        effective_scatter_index = torch.where(
            in_range,
            scatter_index,
            torch.zeros_like(scatter_index),
        )

    reverted_output = expert_output[
        effective_scatter_index.flatten().long()
    ].reshape(
        -1,
        effective_scatter_index.shape[1],
        expert_output.shape[-1],
    )
    reverted_output = reverted_output * in_range.unsqueeze(-1).to(
        reverted_output.dtype
    )
    return reverted_output.sum(dim=1)


def torch_native_grouped_linear_forward(
    expert_counts: torch.Tensor,
    inputs: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Apply native matmul to expert-major rows and [expert, out, in] weights."""

    counts = expert_counts.to(device="cpu", dtype=torch.int64).tolist()
    outputs = []
    start = 0
    for expert, count in enumerate(counts):
        end = start + count
        if count:
            outputs.append(
                inputs[start:end] @ weights[expert].transpose(0, 1)
            )
        start = end

    if start != inputs.shape[0]:
        raise AssertionError(
            f"expert counts cover {start} rows, but input has {inputs.shape[0]}"
        )
    if not outputs:
        return inputs.new_empty((0, weights.shape[1]))
    return torch.cat(outputs, dim=0)


def gmm_combine_golden(
    ep_group: dist.ProcessGroup,
    weighted_silu_output: torch.Tensor,
    fc2_weight: torch.Tensor,
    a2a_splits: torch.Tensor,
    splits: torch.Tensor,
    hidden_states: torch.Tensor,
    scatter_index: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> Dict[str, object]:
    """Run native FC2 GMM, reverse All-to-All, and scatter backward."""

    ep_size = dist.get_world_size(ep_group)
    if splits.ndim != 1 or splits.numel() % ep_size:
        raise ValueError(
            f"splits shape {tuple(splits.shape)} is incompatible with "
            f"EP size {ep_size}"
        )
    if a2a_splits.shape != splits.shape:
        raise ValueError("a2a_splits and splits must have the same shape")
    if scatter_index.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "scatter_index token dimension must match hidden_states"
        )
    if not 0 <= start_idx <= end_idx <= scatter_index.numel():
        raise ValueError("invalid scatter slice")
    if end_idx - start_idx != int(splits.sum().item()):
        raise ValueError("scatter slice rows must equal sum(splits)")

    num_local_experts = splits.numel() // ep_size
    if fc2_weight.shape[0] != num_local_experts:
        raise ValueError(
            f"weight has {fc2_weight.shape[0]} local experts; "
            f"expected {num_local_experts}"
        )

    a2a_splits_cpu = a2a_splits.cpu()
    a2a_split_matrix = a2a_splits_cpu.reshape(
        ep_size, num_local_experts
    )
    experts_a2a_splits = a2a_splits.reshape(
        ep_size, num_local_experts
    ).sum(dim=0, dtype=torch.int32)
    if weighted_silu_output.shape[0] != int(
        experts_a2a_splits.sum().item()
    ):
        raise ValueError(
            "weighted_silu_output rows must equal sum(experts_a2a_splits)"
        )

    a2a_splits_cpu_list = a2a_split_matrix.sum(dim=-1).tolist()
    splits_cpu_list = (
        splits.reshape(ep_size, num_local_experts)
        .sum(dim=-1)
        .cpu()
        .tolist()
    )
    permute_a2a_splits = (
        a2a_split_matrix.permute(1, 0).flatten().tolist()
    )

    permute_a2a_expert_output = torch_native_grouped_linear_forward(
        experts_a2a_splits,
        weighted_silu_output,
        fc2_weight,
    )
    a2a_expert_output = ugly_concat_list(
        permute_a2a_splits,
        permute_a2a_expert_output,
        ep_size,
        num_local_experts,
    )
    combined_expert_output = all_to_all(
        ep_group,
        a2a_expert_output,
        splits_cpu_list,
        a2a_splits_cpu_list,
    )
    expert_output = torch_native_scatter_backward(
        combined_expert_output,
        scatter_index,
        start_idx,
        end_idx,
    )
    output = expert_output.reshape(hidden_states.shape).bfloat16()

    return {
        "experts_a2a_splits": experts_a2a_splits,
        "a2a_splits_cpu_list": a2a_splits_cpu_list,
        "splits_cpu_list": splits_cpu_list,
        "permute_a2a_splits": permute_a2a_splits,
        "permute_a2a_expert_output": permute_a2a_expert_output,
        "a2a_expert_output": a2a_expert_output,
        "combined_expert_output": combined_expert_output,
        "expert_output": expert_output,
        "output": output,
    }


def _build_case(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Dict[str, object]:
    if args.num_experts % world_size:
        raise ValueError(
            f"num_experts={args.num_experts} must be divisible by "
            f"world_size={world_size}"
        )
    if args.topk > args.num_experts:
        raise ValueError("topk must not exceed num_experts")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + rank)
    base_expert = torch.randint(
        args.num_experts,
        (args.tokens_per_rank, 1),
        generator=generator,
        dtype=torch.int64,
    )
    expert_index = (
        base_expert + torch.arange(args.topk, dtype=torch.int64)
    ) % args.num_experts
    flat_expert_index = expert_index.flatten()
    permutation = flat_expert_index.argsort(stable=True)
    splits = torch.bincount(
        flat_expert_index, minlength=args.num_experts
    ).to(torch.int64).to(device)

    scatter_index = torch.empty_like(permutation, dtype=torch.int32)
    scatter_index[permutation] = torch.arange(
        permutation.numel(), dtype=torch.int32
    )
    scatter_index = scatter_index.view_as(expert_index).to(device)

    hidden_generator = torch.Generator(device="cpu")
    hidden_generator.manual_seed(args.seed + 5_000 + rank)
    hidden_states = torch.randn(
        (args.tokens_per_rank, args.hidden_size),
        generator=hidden_generator,
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16, device=device)

    a2a_splits = torch.empty_like(splits)
    dist.all_to_all_single(a2a_splits, splits, group=dist.group.WORLD)
    num_local_experts = args.num_experts // world_size
    experts_a2a_splits = a2a_splits.reshape(
        world_size, num_local_experts
    ).sum(dim=0, dtype=torch.int32)
    received_routes = int(experts_a2a_splits.sum().item())

    activation_generator = torch.Generator(device="cpu")
    activation_generator.manual_seed(args.seed + 10_000 + rank)
    weighted_silu_output = torch.randn(
        (received_routes, args.ffn_size),
        generator=activation_generator,
        dtype=torch.float32,
    ).to(dtype=torch.bfloat16, device=device)

    weight_generator = torch.Generator(device="cpu")
    weight_generator.manual_seed(args.seed + 20_000 + rank)
    fc2_weight = torch.randn(
        (
            num_local_experts,
            args.hidden_size,
            args.ffn_size,
        ),
        generator=weight_generator,
        dtype=torch.float32,
    )
    fc2_weight.mul_(1.0 / math.sqrt(args.ffn_size))
    fc2_weight = fc2_weight.to(dtype=torch.bfloat16, device=device)

    return {
        "expert_index": expert_index.to(device),
        "hidden_states": hidden_states,
        "scatter_index": scatter_index,
        "splits": splits,
        "a2a_splits": a2a_splits,
        "weighted_silu_output": weighted_silu_output,
        "fc2_weight": fc2_weight,
        "start_idx": 0,
        "end_idx": int(splits.sum().item()),
    }


def _assert_accuracy(
    golden: torch.Tensor,
    output: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    """Compare every output element directly with the Torch golden."""

    torch.testing.assert_close(
        golden.float().cpu(),
        output.float().cpu(),
        rtol=rtol,
        atol=atol,
    )


def required_ash_size_mb(
    shape: GmmCombineShape,
    rank_size: int,
) -> int:
    """Return the symmetric heap required by one shape."""

    peer_bytes = (
        2
        * rank_size
        * shape.tokens_per_rank
        * shape.topk
        * shape.hidden_size
        * 2
    )
    slot = shape.tokens_per_rank * shape.topk
    batches = (slot + PIPELINE_BATCH_ROWS - 1) // PIPELINE_BATCH_ROWS
    signal_bytes = (2 * rank_size * batches + rank_size) * SIG_STRIDE * 4
    MB = 1024 * 1024
    required_mib = math.ceil((peer_bytes + signal_bytes) * 1.1 / MB)
    return max(64, (required_mib + 1) // 2 * 2)


def _select_gmm_blocks(
    hidden_size: int, received_rows: int, num_local_experts: int, ffn_size: int
) -> tuple[int, int, int]:
    """Select offline-profiled tiles using tensor shapes, without host routing copies."""
    block_n = 16
    while block_n < hidden_size and block_n < 256:
        block_n *= 2

    if ffn_size == 480:
        # Larger row tiles help narrow outputs, but regress once H exceeds 128.
        if 64 <= hidden_size <= 128 and received_rows >= 512 * num_local_experts:
            return 128, block_n, 64
        # Keep small calls on the latency-oriented baseline. Unaligned widths
        # above 256 benefit from one wider column tile; H=512/768 do not.
        wide_columns = hidden_size >= 1024 or (
            hidden_size > 256 and hidden_size % 256 != 0
        )
        if (wide_columns and received_rows >= 8192
                and received_rows >= 128 * num_local_experts):
            return 32, 512, 128
    elif 128 <= ffn_size <= 1024 and ffn_size % 128 == 0:
        # Aligned reduction widths benefit at smaller expert group sizes.
        if (hidden_size >= 1024 and received_rows >= 8192
                and received_rows >= 64 * num_local_experts):
            return 32, 512, 128

    # Retain the previous policy outside the profiled regions.
    wide_columns = hidden_size >= 1024 or (
        hidden_size > 512 and hidden_size % 256 != 0
    )
    if wide_columns and received_rows >= 512 * num_local_experts:
        return 32, 512, 64
    return 64, block_n, 64


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--tokens-per-rank", type=int, default=512)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=1536)
    parser.add_argument("--ffn-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--rtol", type=float, default=ACCURACY_RTOL)
    parser.add_argument("--atol", type=float, default=ACCURACY_ATOL)
    args = parser.parse_args()
    for name in (
        "num_experts",
        "tokens_per_rank",
        "topk",
        "hidden_size",
        "ffn_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    return args


def initialize_shmem(shape: GmmCombineShape) -> int:
    """Initialize a symmetric heap sized for one shape."""

    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    ash_size_mb = required_ash_size_mb(shape, rank_size)
    if ash_size_mb * 1024 * 1024 > ACLSHMEM_MAX_LOCAL_SIZE:
        raise ValueError(
            f"symmetric heap of {ash_size_mb / 1024:.2f} GB exceeds the "
            "40 GB ACLSHMEM_MAX_LOCAL_SIZE cap; shrink the swept shapes"
        )
    dist.barrier()
    init_aclshmem(rank, rank_size, ash_size_mb * 1024 * 1024, ash.OpEngineType.UDMA)
    return ash_size_mb


def finalize_shmem() -> None:
    ash.aclshmem_finalize()


def run_case(
    shape: GmmCombineShape,
    case_executor: Optional[Callable] = None,
    seed: int = 20260903,
    rtol: float = ACCURACY_RTOL,
    atol: float = ACCURACY_ATOL,
    verify: bool = True,
) -> None:
    """Run one shape after SHMEM has been initialized."""

    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    if shape.num_experts % rank_size:
        raise ValueError(
            f"num_experts={shape.num_experts} must be divisible by "
            f"rank_size={rank_size}"
        )
    if shape.topk > shape.num_experts:
        raise ValueError("topk must not exceed num_experts")
    args = argparse.Namespace(
        tokens_per_rank=shape.tokens_per_rank,
        hidden_size=shape.hidden_size,
        ffn_size=shape.ffn_size,
        topk=shape.topk,
        num_experts=shape.num_experts,
        seed=seed,
        rtol=rtol,
        atol=atol,
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"npu:{local_rank}")
    _run_single_case(
        args,
        rank,
        rank_size,
        device,
        shape.name,
        case_executor,
        verify,
    )


def run(
    shape: GmmCombineShape,
    case_executor: Optional[Callable] = None,
    seed: int = 20260903,
    rtol: float = ACCURACY_RTOL,
    atol: float = ACCURACY_ATOL,
    verify: bool = True,
) -> None:
    """Initialize resources and run exactly one shape."""

    ash_size_mb = initialize_shmem(shape)
    rank = dist.get_rank()

    try:
        if rank == 0:
            print(
                f"GMM_COMBINE_CONFIG case={shape.name} " f"ash_heap={ash_size_mb}MB",
                flush=True,
            )
        run_case(shape, case_executor, seed, rtol, atol, verify)
    finally:
        finalize_shmem()


@triton.jit(do_not_specialize=["epoch", "BATCHES"])
def finish_round(signals, epoch, R: tl.constexpr, RANK: tl.constexpr,
                 BATCHES):
    # Each rank publishes local completion. A faster peer may have advanced
    # again before this rank polls, so acknowledgments use >=, not equality.
    ack = signals + 2 * R * BATCHES * 16
    for peer in range(R):
        if peer != RANK:
            dl.notify(ack + RANK * 16, peer, signal=epoch)
    for peer in range(R):
        if peer != RANK:
            seen = tl.atomic_add(ack + peer * 16, 0, sem="acquire", scope="sys")
            while seen < epoch:
                seen = tl.atomic_add(ack + peer * 16, 0, sem="acquire", scope="sys")


@triton.jit
def prepare_scan(a2a, splits, counts_out, input_offsets, slot_offsets,
                 tile_offsets, tile_total, recv_rows,
                 R: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
                 EP: tl.constexpr):
    ix = tl.arange(0, EP * R)
    counts = tl.load(a2a + ix, ix < R * E, 0).to(tl.int32)
    prefix = tl.cumsum(counts) - counts
    base = tl.zeros((EP * R,), tl.int32)
    split_counts = tl.load(splits + ix, ix < R * E, 0)
    for p in range(R):
        before = tl.sum(tl.where(ix < p * E, counts, 0))
        base = tl.where(ix // E == p, before, base)
        tl.store(recv_rows + p, tl.sum(tl.where(ix // E == p, split_counts, 0)))
    tl.store(counts_out + ix, counts, ix < R * E)
    tl.store(slot_offsets + ix, prefix - base, ix < R * E)
    expert, source = ix // R, ix % R
    bucket = source * E + expert
    ec = tl.load(a2a + bucket, expert < E, 0).to(tl.int32)
    eprefix = tl.cumsum(ec)
    tl.store(input_offsets + bucket, eprefix - ec, expert < E)
    # One tile contains rows from one destination peer and one expert.
    tiles = tl.cdiv(counts, BM)
    tl.store(tile_offsets + ix, tl.cumsum(tiles) - tiles, ix < R * E)
    tl.store(tile_total, tl.sum(tiles))


@triton.jit(do_not_specialize=["SLOT", "BATCHES"])
def prepare_tiles(counts, input_offsets, slot_offsets, tile_offsets,
                  experts, starts, row_counts, tile_destinations, tile_batches,
                  tile_batch_counts, R: tl.constexpr, E: tl.constexpr,
                  SLOT, ROWS: tl.constexpr, BATCHES, BM: tl.constexpr):
    bucket = tl.program_id(0)
    peer = bucket // E
    expert = bucket % E
    rows = tl.load(counts + bucket)
    begin = tl.load(input_offsets + bucket)
    dest = tl.load(slot_offsets + bucket)
    first_tile = tl.load(tile_offsets + bucket)
    for m in range(tl.cdiv(rows, BM)):
        tile = first_tile + m
        count = tl.minimum(BM, rows - m * BM)
        offset = dest + m * BM
        first_batch = offset // ROWS
        last_batch = (offset + count - 1) // ROWS
        tl.store(experts + tile, expert)
        tl.store(starts + tile, begin + m * BM)
        tl.store(row_counts + tile, count)
        tl.store(tile_destinations + tile, peer * SLOT + offset)
        tl.store(tile_batches + tile * 2, peer * BATCHES + first_batch)
        if last_batch != first_batch:
            tl.store(tile_batches + tile * 2 + 1, peer * BATCHES + last_batch)
        tl.store(tile_batch_counts + tile, tl.where(last_batch != first_batch, 2, 1))


@triton.jit(do_not_specialize=["BATCHES"])
def batch_metadata(counts, slot_offsets, expected,
                   completed, send_rows, E: tl.constexpr,
                   BATCHES, ROWS: tl.constexpr, BM: tl.constexpr,
                   EPAD: tl.constexpr):
    peer = tl.program_id(0)
    b = tl.program_id(1)
    contributors = tl.full((), 0, tl.int32)
    total = tl.full((), 0, tl.int32)
    # Large expert vectors trigger a BiSheng backend failure on 950DT.
    for chunk in range(tl.cdiv(E, EPAD)):
        e = chunk * EPAD + tl.arange(0, EPAD)
        count = tl.load(counts + peer * E + e, e < E, 0)
        slot = tl.load(slot_offsets + peer * E + e, e < E, 0)
        lo = tl.maximum(b * ROWS, slot)
        hi = tl.minimum((b + 1) * ROWS, slot + count)
        first = (lo - slot) // BM
        end = tl.cdiv(hi - slot, BM)
        contributors += (
            tl.sum(tl.where((e < E) & (hi > lo), end - first, 0))
        ).to(tl.int32)
        total += tl.sum(count)
    tl.store(expected + peer * BATCHES + b, contributors)
    tl.store(completed + (peer * BATCHES + b) * 16, 0)
    if b == 0:
        tl.store(send_rows + peer, total)


@triton.jit
def prepare_gather(scatter, recv_rows, gather_rows,
                   num_routes, SLOT, R: tl.constexpr, RANK: tl.constexpr,
                   BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logical = tl.load(scatter + idx, idx < num_routes, 0)
    offset = tl.full((), 0, tl.int64)
    physical = tl.zeros((BLOCK,), tl.int32)
    for source in range(R):
        count = tl.load(recv_rows + source)
        within = logical - offset
        valid = (within >= 0) & (within < count)
        base = tl.where(source == RANK, RANK * SLOT, (R + source) * SLOT)
        physical = tl.where(valid, (base + within).to(tl.int32), physical)
        offset += count
    tl.store(gather_rows + idx, physical, idx < num_routes)


@triton.jit(do_not_specialize=["epoch", "SLOT", "BATCHES"])
def batch_send(peer_mem, completed, expected, send_rows, signals,
               epoch, rank: tl.constexpr, R: tl.constexpr, SLOT,
               H: tl.constexpr, BATCHES, ROWS: tl.constexpr,
               PUT_BYTES: tl.constexpr):
    peer = tl.program_id(0)
    if peer != rank:
        rows = tl.load(send_rows + peer).to(tl.int64)
        count = tl.cdiv(rows, ROWS)
        for batch in range(count):
            idx = peer * BATCHES + batch
            wanted = tl.load(expected + idx)
            token = dl.wait(completed + idx * 16, 1, "sys", "acquire", waitValue=wanted)
            source = dl.consume_token(peer_mem, token)
            begin = batch * ROWS
            elements = tl.minimum(ROWS, rows - begin) * H
            src = (peer * SLOT + begin).to(tl.int64) * H
            dst = (R * SLOT + rank * SLOT + begin).to(tl.int64) * H
            for offset in range(0, elements, PUT_BYTES // 2):
                size = tl.minimum(elements - offset, PUT_BYTES // 2)
                libshmem_device.putmem(peer_mem + dst + offset,
                                      source + src + offset, size * 2,
                                      peer.to(tl.int32))
        for batch in range(count):
            dl.notify(signals + (rank * BATCHES + batch) * 16,
                      peer.to(tl.int32), signal=epoch)


@triton.jit(do_not_specialize=["epoch", "T", "BATCHES"])
def single_reduce_tokens(peer_mem, gather_rows, expected,
                                signals, output, epoch, T, H: tl.constexpr, TOPK: tl.constexpr,
                                R: tl.constexpr, BATCHES,
                                BT: tl.constexpr, BN: tl.constexpr,
                                END: tl.constexpr, worker, workers, recv_rows,
                                RANK: tl.constexpr, ROWS: tl.constexpr):
    # Wait once for each valid received batch, then reduce entirely in Triton.
    # GMM and UDMA remain concurrent. This variant starts reduction after all
    # required local and remote rows arrive.
    chain = tl.full((), 0, tl.int32)
    for peer in range(R):
        count = tl.cdiv(tl.load(recv_rows + peer), ROWS)
        for batch in range(count):
            idx = peer * BATCHES + batch
            dep = idx + tl.where(peer == RANK, R * BATCHES, 0)
            wanted = tl.where(peer == RANK, tl.load(expected + idx), epoch)
            token = dl.wait(signals + dep * 16, 1, "sys", "acquire", waitValue=wanted)
            chain = dl.consume_token(chain, token)
    source = dl.consume_token(peer_mem, chain)
    if H % BN == 0:
        nt = tl.cdiv(H, BN)
        for task in range(worker, END, workers):
            tokens = task * BT + tl.arange(0, BT)
            for column_task in range(nt):
                cols = column_task * BN + tl.arange(0, BN)
                acc = tl.zeros((BT, BN), tl.float32)
                for k in range(TOPK):
                    row = tl.load(gather_rows + tokens * TOPK + k, tokens < T, 0).to(tl.int64)
                    value = tl.load(source + row[:, None] * H + cols[None, :],
                                    tokens[:, None] < T, 0)
                    acc += value.to(tl.float32)
                tl.store(output + tokens.to(tl.int64)[:, None] * H + cols[None, :],
                         acc.to(tl.bfloat16), (tokens[:, None] < T) & (cols[None, :] < H))
    else:
        nt = tl.cdiv(H, BN)
        for task in range(worker, END, workers):
            for i in range(BT):
                token_id = task * BT + i
                if token_id < T:
                    for column_task in range(nt):
                        cols = column_task * BN + tl.arange(0, BN)
                        acc = tl.zeros((BN,), tl.float32)
                        for k in range(TOPK):
                            row = tl.load(gather_rows + token_id * TOPK + k).to(tl.int64)
                            value = tl.load(source + row * H + cols, cols < H, 0)
                            acc += value.to(tl.float32)
                        tl.store(output + token_id.to(tl.int64) * H + cols,
                                 acc.to(tl.bfloat16), cols < H)

@triton.jit
def single_batch_gmm_cube(x, w, tile_destinations, experts, starts, row_counts,
                          tile_total, out, acknowledgments, epoch,
                          H: tl.constexpr, F: tl.constexpr,
                          WE: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nt = tl.cdiv(H, BN)
    tasks = tl.load(tile_total)
    rr = tl.arange(0, BM)
    columns = tl.arange(0, BN)
    ack = acknowledgments + pid * 16
    for task in range(pid, tasks, tl.num_programs(0)):
        expert = tl.load(experts + task).to(tl.int64)
        start = tl.load(starts + task).to(tl.int64)
        count = tl.load(row_counts + task)
        physical = tl.load(tile_destinations + task).to(tl.int64)
        sequence = epoch * (tasks + 1) + task + 1
        chain = tl.full((), 0, tl.int32)
        if task != pid:
            chain = dl.wait(ack, 1, "sys", "acquire",
                            waitValue=sequence - tl.num_programs(0))
        source = dl.consume_token(x, chain)
        for column_task in range(nt):
            cols = column_task * BN + columns
            acc = tl.zeros((BM, BN), tl.float32)
            for kk in range(tl.cdiv(F, BK)):
                k = kk * BK + tl.arange(0, BK)
                a = tl.load(source + (start + rr[:, None]) * F + k[None, :],
                            (rr[:, None] < count) & (k[None, :] < F), 0)
                b = tl.load(w + expert * WE + cols[None, :] * WN + k[:, None] * WK,
                            (k[:, None] < F) & (cols[None, :] < H), 0)
                acc = tl.dot(a, b, acc)
            ext.compile_hint(acc, "matmul_at_least_once")
            # All rows are contiguous in the destination peer's buffer.
            tl.store(out + (physical + rr[:, None]) * H + cols[None, :],
                     acc.to(tl.bfloat16),
                     (rr[:, None] < count) & (cols[None, :] < H))
        ext.sync_block_set("cube", "vector", 15, ext.PIPE.PIPE_FIX, ext.PIPE.PIPE_MTE2)


@triton.jit
def single_batch_gmm_publish(tile_total, tile_batches, tile_batch_counts,
                             completed, acknowledgments, epoch):
    pid = tl.program_id(0)
    tasks = tl.load(tile_total)
    ack = acknowledgments + pid * 16
    for task in range(pid, tasks, tl.num_programs(0)):
        sequence = epoch * (tasks + 1) + task + 1
        ext.sync_block_wait("cube", "vector", 15, ext.PIPE.PIPE_FIX, ext.PIPE.PIPE_MTE2)
        libshmem_device.quiet_lowlevel(
            transport=libshmem_device.ACLSHMEM_TRANSPORT_MTE
        )
        for i in range(tl.load(tile_batch_counts + task)):
            batch = tl.load(tile_batches + task * 2 + i)
            tl.atomic_add(completed + batch * 16, 1, sem="acq_rel", scope="sys")
        tl.store(ack, sequence)
        libshmem_device.quiet_lowlevel(
            transport=libshmem_device.ACLSHMEM_TRANSPORT_MTE
        )


@triton.jit(do_not_specialize=["epoch", "T", "SLOT", "BATCHES"])
def single_gmm_send_reduce(x, w, tile_destinations, experts, starts, row_counts, tile_total,
                          tile_batches, tile_batch_counts, peer_mem, completed,
                          expected, send_rows, signals, gather_rows,
                          output, acknowledgments, recv_rows, epoch, T, SLOT, BATCHES,
                          H: tl.constexpr, F: tl.constexpr, TOPK: tl.constexpr,
                          R: tl.constexpr, RANK: tl.constexpr,
                          WE: tl.constexpr, WN: tl.constexpr, WK: tl.constexpr,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          ROWS: tl.constexpr, PUT_BYTES: tl.constexpr,
                          BT: tl.constexpr, GN: tl.constexpr,
                          END: tl.constexpr):
    pid = tl.program_id(0)
    cores = tl.num_programs(0)
    # The AIC computes GMM tiles, publishing each tile to its paired AIV0.
    with ext.scope(core_mode="cube"):
        single_batch_gmm_cube(x, w, tile_destinations, experts, starts, row_counts,
                              tile_total, peer_mem, acknowledgments, epoch,
                              H, F, WE, WN, WK, BM, BN, BK)

    with ext.scope(core_mode="vector"):
        vector_lane = ext.sub_vec_id()
        if vector_lane == 0:
            # AIV0 publishes completed batches, then joins the reducers.
            single_batch_gmm_publish(tile_total, tile_batches, tile_batch_counts,
                                     completed, acknowledgments, epoch)
        if vector_lane == 1 and pid < R:
            # The first R AIV1 lanes send one peer's batches each.
            batch_send(peer_mem, completed, expected, send_rows,
                       signals, epoch, RANK, R, SLOT, H, BATCHES,
                       ROWS, PUT_BYTES)
        else:
            # AIV1 reducers use [0, cores-R); AIV0 reducers use the rest.
            worker = tl.where(vector_lane == 0, cores - R + pid, pid - R)
            single_reduce_tokens(
                peer_mem, gather_rows, expected, signals, output,
                epoch, T, H, TOPK, R, BATCHES, BT, GN, END,
                worker, 2 * cores - R, recv_rows, RANK, ROWS)


def _allocate_gmm_combine_workspace(case, ranks, slot):
    """Allocate reusable buffers; routing metadata is computed per call."""
    x, w = case["weighted_silu_output"], case["fc2_weight"]
    e, h, f = w.shape
    t, topk = case["scatter_index"].shape
    bm, bn, _ = _select_gmm_blocks(h, x.shape[0], e, f)
    bk = min(256, triton.next_power_of_2(f))
    if bn == 512:
        bm = 64
    batches = triton.cdiv(slot, PIPELINE_BATCH_ROWS)
    buckets = ranks * e
    capacity = triton.cdiv(x.shape[0], bm) + buckets

    def buf(n, dtype=torch.int32):
        return torch.empty(n, device=x.device, dtype=dtype)

    workspace = {
        "slot": slot,
        "blocks": (bm, bn, bk),
        "acknowledgments": torch.zeros(_aicore_num() * 16, device=x.device, dtype=torch.int32),
        "gmm_workers": _aicore_num(),
        "epoch": 0,
        "counts": buf(buckets),
        "in_offsets": buf(buckets),
        "slot_offsets": buf(buckets),
        "tile_offsets": buf(buckets),
        "tile_total": buf(1),
        "experts": buf(capacity),
        "starts": buf(capacity),
        "row_counts": buf(capacity),
        "tile_batches": buf(capacity * 2),
        "tile_batch_counts": buf(capacity),
        "tile_destinations": buf(capacity),
        "gather": buf(t * topk),
        "recv_rows": buf(ranks, torch.int64),
        "send_rows": buf(ranks),
        "expected": buf(ranks * batches),
    }
    return workspace


def gmm_combine_udma(case, output, peer, signal, workspace, rank, ranks):
    """Launch Triton metadata, GMM, UDMA send and token combine for one call."""

    x, w = case["weighted_silu_output"], case["fc2_weight"]
    e, h, f = w.shape
    t, topk = case["scatter_index"].shape
    bm, bn, bk = workspace["blocks"]
    slot = workspace["slot"]
    rows = PIPELINE_BATCH_ROWS
    gather_columns = min(2048, triton.next_power_of_2(h))
    if h % gather_columns != 0 and h % 512 == 0:
        gather_columns = 512
    batches = triton.cdiv(slot, rows)
    completed = signal[ranks * batches * 16:2 * ranks * batches * 16]
    workspace["epoch"] += 1
    epoch = workspace["epoch"]
    current = torch.npu.current_stream()
    # Complete framework producers/readers before reusing the work buffers.
    # Metadata generation and all synchronization below are included in timing.
    current.synchronize()
    prepare_scan[(1,)](
        case["a2a_splits"], case["splits"], workspace["counts"], workspace["in_offsets"],
        workspace["slot_offsets"], workspace["tile_offsets"],
        workspace["tile_total"], workspace["recv_rows"], R=ranks, E=e, BM=bm,
        EP=triton.next_power_of_2(e))
    prepare_tiles[(ranks * e,)](
        workspace["counts"], workspace["in_offsets"], workspace["slot_offsets"],
        workspace["tile_offsets"], workspace["experts"], workspace["starts"], workspace["row_counts"],
        workspace["tile_destinations"], workspace["tile_batches"], workspace["tile_batch_counts"],
        R=ranks, E=e, SLOT=slot, ROWS=rows, BATCHES=batches, BM=bm)
    prepare_gather[(triton.cdiv(t * topk, 512),)](
        case["scatter_index"], workspace["recv_rows"], workspace["gather"],
        t * topk, slot, R=ranks, RANK=rank, BLOCK=512)
    batch_metadata[(ranks, batches)](
        workspace["counts"], workspace["slot_offsets"],
        workspace["expected"], completed, workspace["send_rows"],
        E=e, BATCHES=batches, ROWS=rows, BM=bm,
        EPAD=min(32, triton.next_power_of_2(e)))

    cores = workspace["gmm_workers"]
    if _aivector_core_num() != 2 * cores or ranks > cores:
        raise RuntimeError("single kernel requires two Vector subcores per Cube and ranks <= Cube cores")
    single_gmm_send_reduce[(cores,)](
        x, w, workspace["tile_destinations"], workspace["experts"], workspace["starts"],
        workspace["row_counts"], workspace["tile_total"], workspace["tile_batches"],
        workspace["tile_batch_counts"], peer, completed, workspace["expected"],
        workspace["send_rows"], signal, workspace["gather"],
        output, workspace["acknowledgments"], workspace["recv_rows"], epoch, t, slot, batches,
        H=h, F=f, TOPK=topk, R=ranks, RANK=rank,
        WE=w.stride(0), WN=w.stride(1), WK=w.stride(2),
        BM=bm, BN=bn, BK=bk, ROWS=rows, PUT_BYTES=PIPELINE_PUT_BYTES,
        BT=GATHER_TOKEN_BLOCK, GN=gather_columns,
        END=triton.cdiv(t, GATHER_TOKEN_BLOCK),
        enable_dynamic_cv_pipeline=False, enable_mixed_cv=True, compile_mode="simd",
        enable_preload=False, multibuffer=False, disable_auto_inject_block_sync=True)
    finish_round[(1,)](signal, epoch, R=ranks, RANK=rank, BATCHES=batches)
    # Complete this call before reusing its buffers; this wait is timed too.
    current.synchronize()
    return output


def _run_single_case(
    args,
    rank,
    ranks,
    device,
    name,
    case_executor=None,
    verify=True,
):
    case = _build_case(args, rank, ranks, device)
    slot = args.tokens_per_rank * args.topk
    rows = PIPELINE_BATCH_ROWS
    local_rank = int(os.environ["LOCAL_RANK"])
    peer = ash.aclshmem_create_tensor(
        [2 * ranks * slot * args.hidden_size],
        dtype=torch.bfloat16,
        device_id=local_rank,
    )
    signal = ash.aclshmem_create_tensor(
        [(2 * ranks * triton.cdiv(slot, rows) + ranks) * 16],
        dtype=torch.int32,
        device_id=local_rank,
    )
    output = torch.empty_like(case["hidden_states"])
    workspace = _allocate_gmm_combine_workspace(case, ranks, slot)

    def triton_op():
        return gmm_combine_udma(case, output, peer, signal, workspace, rank, ranks)

    def golden():
        return gmm_combine_golden(
            dist.group.WORLD, case["weighted_silu_output"], case["fc2_weight"],
            case["a2a_splits"], case["splits"], case["hidden_states"],
            case["scatter_index"], case["start_idx"], case["end_idx"])

    try:
        signal.zero_()
        peer.fill_(float("nan"))
        output.fill_(float("nan"))
        torch.npu.synchronize()
        dist.barrier()
        case_info = {
            "name": name,
            "tokens_per_rank": args.tokens_per_rank,
            "hidden_size": args.hidden_size,
            "ffn_size": args.ffn_size,
            "topk": args.topk,
            "num_experts": args.num_experts,
            "rank_count": ranks,
        }
        if case_executor is None:
            truth = golden()
            output = triton_op()
        else:
            truth = case_executor(golden, triton_op, case_info)
        if truth is None:
            raise AssertionError("golden recipe must return its output state")

        torch.npu.synchronize()
        dist.barrier()
        if verify:
            passed = 1
            try:
                _assert_accuracy(truth["output"], output, args.rtol, args.atol)
            except AssertionError as exc:
                passed = 0
                print(
                    f"ACCURACY case={name} rank={rank} FAIL {exc}",
                    flush=True,
                )
            accuracy = torch.tensor([passed], dtype=torch.int32, device=device)
            dist.all_reduce(accuracy, op=dist.ReduceOp.MIN)
            if not accuracy.item():
                raise AssertionError(f"GMM Combine accuracy failed for case={name}")
            if rank == 0:
                print(f"[PASS] case={name} ranks={ranks}", flush=True)
    finally:
        torch.npu.synchronize()
        dist.barrier()
        ash.aclshmem_free_tensor(signal)
        ash.aclshmem_free_tensor(peer)


if __name__ == "__main__":
    cli_args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    try:
        run(
            GmmCombineShape(
                "single",
                cli_args.tokens_per_rank,
                cli_args.hidden_size,
                cli_args.ffn_size,
                cli_args.topk,
                cli_args.num_experts,
            ),
            seed=cli_args.seed,
            rtol=cli_args.rtol,
            atol=cli_args.atol,
        )
    finally:
        dist.destroy_process_group()
