# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for reverse-a2a-norm-matmul-2.py (dual-stream fused kernel).

Correctness path lives in tutorials/ascend/reverse-a2a-norm-matmul-2.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/reverse-a2a-norm-matmul-2.py
"""

import os

import torch
import torch_npu
import triton
import shmem as ash
import torch.distributed as dist

from prof import (
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
    verify_enabled,
)
from utils import init_aclshmem

tut = load_tutorial("reverse-a2a-norm-matmul-2")


def run_single_case(
    rank,
    rank_size,
    device,
    S,
    H,
    D,
    eps,
    ash_size_mb,
    comm_block_s,
    buffer_num,
):
    """Run one S/H/D case with N equal to D. Returns True on accuracy pass."""
    local_rank = int(os.environ["LOCAL_RANK"])
    N = D
    comm_block_d = max(D, triton.next_power_of_2(D))
    peer_mem = None
    signal_mem = None
    try:
        S_total = S * rank_size
        peer_mem_elems = 2 * max(rank_size - 1, 1) * S * H * D
        peer_mem = ash.aclshmem_create_tensor(
            [peer_mem_elems], dtype=torch.bfloat16, device_id=local_rank
        )
        num_blocks_s = triton.cdiv(S, comm_block_s)
        signal_elements = num_blocks_s * rank_size * tut.SIG_STRIDE
        signal_mem = ash.aclshmem_create_tensor(
            [signal_elements], dtype=torch.int32, device_id=local_rank
        )

        torch.manual_seed(2026)
        norm_weight = torch.randn([D], dtype=torch.bfloat16, device=device)
        linear_weight = torch.randn([N, 2 * D], dtype=torch.bfloat16, device=device)
        torch.manual_seed(42 + rank)
        tensor_1 = torch.randn([S_total, H, D], dtype=torch.bfloat16, device=device)
        tensor_2 = torch.randn([S_total, H, D], dtype=torch.bfloat16, device=device)
        recv_1 = torch.empty([S, H * rank_size, D], dtype=torch.bfloat16, device=device)
        recv_2 = torch.empty_like(recv_1)
        output = torch.empty([S, H * rank_size, N], dtype=torch.bfloat16, device=device)

        def golden_launch(return_received=True):
            send_1 = torch.cat(
                torch.chunk(tensor_1, rank_size, dim=0), dim=1
            ).contiguous()
            send_1 = send_1.transpose(1, 0).contiguous()
            send_2 = torch.cat(
                torch.chunk(tensor_2, rank_size, dim=0), dim=1
            ).contiguous()
            send_2 = send_2.transpose(1, 0).contiguous()

            golden_recv_1 = torch.empty_like(send_1)
            golden_recv_2 = torch.empty_like(send_2)
            dist.all_to_all_single(golden_recv_1, send_1)
            dist.all_to_all_single(golden_recv_2, send_2)

            golden_recv_1 = golden_recv_1.transpose(0, 1).contiguous()
            golden_recv_2 = golden_recv_2.transpose(0, 1).contiguous()
            norm_1 = torch_npu.npu_rms_norm(golden_recv_1, norm_weight, eps)[0]
            norm_2 = torch_npu.npu_rms_norm(golden_recv_2, norm_weight, eps)[0]
            result = torch.cat([norm_1, norm_2], dim=-1) @ linear_weight.T
            if return_received:
                return result, golden_recv_1, golden_recv_2
            return result

        def kernel_launch():
            tut.hccl_reverse_a2a_norm_matmul_2(
                tensor_1,
                tensor_2,
                norm_weight,
                linear_weight,
                output,
                recv_1,
                recv_2,
                peer_mem,
                signal_mem,
                rank,
                rank_size,
                eps=eps,
                buffer_num=buffer_num,
                comm_block_s=comm_block_s,
                comm_block_d=comm_block_d,
            )

        @prof(label="reverse_a2a_norm_matmul_2")
        def run_golden():
            return golden_launch()

        @prof(
            label="reverse_a2a_norm_matmul_2",
            prepare=lambda: signal_mem.fill_(0),
        )
        def run_kernel():
            return kernel_launch()

        golden, golden_recv_1, golden_recv_2 = run_golden()
        run_kernel()
        prof_report(run_golden, run_kernel, context=f"S={S} H={H} D={D} N={N}")

        if not verify_enabled():
            return True

        passed = torch.tensor([1], dtype=torch.int32, device=device)
        error_message = ""
        accuracy_method = ""
        accuracy_details = {}
        try:
            accuracy_method, accuracy_details = tut._assert_accuracy(
                golden,
                output,
                golden_recv_1,
                golden_recv_2,
                norm_weight,
                linear_weight,
                eps,
            )
        except AssertionError as error:
            passed[0] = 0
            error_message = str(error)
        all_passed = [torch.zeros_like(passed) for _ in range(rank_size)]
        dist.all_gather(all_passed, passed)
        if rank == 0:
            print(
                "CONFIG "
                f"engine=UDMA ranks={rank_size} S={S} H={H} D={D} N={N} "
                f"ash_heap={ash_size_mb}MB "
                f"norm_weight=[{D}] linear_weight=[{N}, {2 * D}]",
                flush=True,
            )
        for rank_id in range(rank_size):
            if rank == rank_id:
                if passed.item():
                    details = " ".join(
                        f"{name}_ratio={value:.6g}"
                        for name, value in accuracy_details.items()
                    )
                    suffix = f" {details}" if details else ""
                    print(
                        f"[PASS] Rank {rank}: method={accuracy_method}{suffix}",
                        flush=True,
                    )
                else:
                    print(f"[FAIL] Rank {rank}:\n{error_message}", flush=True)
            dist.barrier()
        return all(value.item() != 0 for value in all_passed)
    finally:
        if peer_mem is not None:
            ash.aclshmem_free_tensor(peer_mem)
        if signal_mem is not None:
            ash.aclshmem_free_tensor(signal_mem)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    rank_size = dist.get_world_size()
    device = torch.device(f"npu:{local_rank}")

    # Same knobs and defaults as the tutorial.
    bench_mode = "FUSED_RUN_BENCH" in os.environ
    eps = float(os.environ.get("FUSED_EPS", "1e-6"))
    ash_size_mb = int(os.environ.get("FUSED_ASH_SIZE_MB", str(tut.DEFAULT_ASH_SIZE_MB)))
    comm_block_s = int(os.environ.get("FUSED_COMM_BLOCK_S", "1024"))
    buffer_num = 2
    if bench_mode:
        cases = [
            (S, tut.HIDDEN_GROUPS, D)
            for S in tut.SEQUENCE_LENGTHS
            for D in tut.FEATURE_DIMS
        ]
    else:
        cases = [
            (
                int(os.environ.get("FUSED_S", "128")),
                int(os.environ.get("FUSED_H", "4")),
                int(os.environ.get("FUSED_D", "128")),
            )
        ]

    dist.barrier()
    init_aclshmem(rank, rank_size, ash_size_mb * 1024 * 1024, ash.OpEngineType.UDMA)

    try:
        cases = select_cases(cases)
        failed_cases = []
        for case_id, (S, H, D) in enumerate(cases):
            N = D
            if rank == 0 and len(cases) > 1:
                print(
                    f"CASE {case_id + 1}/{len(cases)}: S={S} H={H} D={D} N={N}",
                    flush=True,
                )
            case_passed = run_single_case(
                rank,
                rank_size,
                device,
                S,
                H,
                D,
                eps,
                ash_size_mb,
                comm_block_s,
                buffer_num,
            )
            if not case_passed:
                failed_cases.append((S, H, D, N))
        if failed_cases:
            summary = ", ".join(
                f"S={S} H={H} D={D} N={N}" for S, H, D, N in failed_cases
            )
            raise AssertionError(
                "hccl_reverse_a2a_norm_matmul_2 accuracy check failed for cases: "
                f"{summary}"
            )
    finally:
        ash.aclshmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    main()
