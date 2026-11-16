# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Perf script for transpose-all2all-udma.py (UDMA transpose all-to-all).

Correctness path lives in tutorials/ascend/transpose-all2all-udma.py;
this script re-uses its kernel and golden functions and only adds event timing:

    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    A2A_BENCH_S=2048,4096 A2A_BENCH_H=1,2 \
    torchrun --standalone --nproc_per_node=2 \
      python/triton_dist/test/ascend/transpose-all2all-udma.py
"""

import os

import torch
import shmem as ash
import torch.distributed as dist

from prof import (
    check_all_ranks_close,
    load_tutorial,
    prof,
    prof_report,
    run_options,
    select_cases,
)
from utils import init_aclshmem

tut = load_tutorial("transpose-all2all-udma")


def main():
    pe = dist.get_rank()
    world_size = dist.get_world_size()

    bench_s = tut._parse_int_list("A2A_BENCH_S", tut.SWEEP_S)
    bench_h = tut._parse_int_list("A2A_BENCH_H", tut.SWEEP_H)

    init_aclshmem(
        pe,
        world_size,
        tut.ash_pool_bytes(world_size, bench_s, bench_h),
        ash.OpEngineType.UDMA,
    )

    # Optional subset filter for quick validation / tuning sweeps.
    sweep_cases = [(s, h) for s in bench_s for h in bench_h]

    sweep_cases = select_cases(sweep_cases)
    total_cases = len(sweep_cases)
    if pe == 0:
        print(
            f"[START] put transpose All2All sweep "
            f"({total_cases} cases) on world_size={world_size}...",
            flush=True,
        )

    try:
        for case_idx, (S, H) in enumerate(sweep_cases):
            dist.barrier()

            dtype = torch.bfloat16
            n_head = H * world_size
            D = tut.HEAD_DIM

            # Symmetric output C (putmem dest) + local staging buffer.
            # Any rank failing to allocate -> all ranks skip this case.
            C_local = tut.make_symmetric_output(world_size, S, H, D, dtype)
            can_alloc = torch.tensor([1], dtype=torch.int32).npu()
            if C_local is None:
                can_alloc[0] = 0
            dist.all_reduce(can_alloc, op=dist.ReduceOp.MIN)
            if can_alloc.item() == 0:
                if C_local is not None:
                    ash.aclshmem_free_tensor(C_local)
                if pe == 0:
                    print(f"  case S={S} H={H} SKIP (OOM)", flush=True)
                continue
            staging = torch.zeros([world_size * S * H * D], dtype=dtype).npu()

            # Create input tensor with seeded data
            torch.manual_seed(42 + case_idx * world_size + pe)
            A_local = torch.randn([S, n_head, D], dtype=dtype).npu()

            @prof(label="transpose_all2all_udma")
            def run_golden():
                return tut.torch_transpose_a2a(A_local, world_size)

            # UDMA kernel.  quiet() inside the kernel means kernel completion
            # implies every put landed, so back-to-back launches are safe on
            # the same stream.
            @prof(label="transpose_all2all_udma")
            def run_kernel():
                return tut.hccl_transpose_a2a_kernel_launcher(
                    A_local, C_local, staging, pe, world_size
                )

            C_golden = run_golden()
            launch_ai_cores = run_kernel()
            prof_report(
                run_golden,
                run_kernel,
                context=f"S={S} H={H} cores={launch_ai_cores}",
            )
            check_all_ranks_close(
                C_local, C_golden, rtol=1e-3, atol=1e-3, context=f"S={S}, H={H}"
            )

            # Release symmetric memory
            ash.aclshmem_free_tensor(C_local)
    finally:
        # Skip explicit aclshmem_finalize — it SIGABRTs on this environment
        # after a clean run; the OS reclaims resources at process exit.
        dist.destroy_process_group()


if __name__ == "__main__":
    run_options()
    local_pe = int(os.environ["LOCAL_RANK"])
    pe = int(os.environ["RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=pe)

    dist.barrier()
    main()
    if local_pe == 0:
        print("[INFO] Perf run finished successfully")
