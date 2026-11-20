# Ascend Triton Operator Examples

This directory contains standalone operator examples for Triton Distributed Ascend. Each `.py` file can be used directly as an entry point for execution and correctness checking: it runs the Torch golden reference and the Triton kernel once, then verifies the results. Event timing and speedup measurement live in the [`python/triton_dist/test/ascend/`](../../python/triton_dist/test/ascend/README.md) directory; except for the autotune example described separately, no additional analysis or pytest driver is required.

## Prerequisites

- An Ascend NPU, CANN/torch_npu, Triton, and `triton_dist` must be installed and importable in the current Python environment.
- Multi-device examples use `torchrun` and HCCL; at least two devices are usually required.
- `ASCEND_RT_VISIBLE_DEVICES` must match the number of processes specified by `--nproc_per_node`; for example, `0,1` corresponds to two processes.
- The commands below assume that the current shell is in the repository root. Otherwise, replace the script paths with absolute paths.

Verify the environment first:

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401
import triton
import triton_dist
print("torch/triton/torch_npu/triton_dist import: OK")
PY
```

## General Launch Method

Most examples can be launched as follows:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/<example-file>.py
```

shmem initialization is controlled by ASH_MASTER_ADDR and ASH_MASTER_PORT, and the scripts combine them into tcp://{ASH_MASTER_ADDR}:{ASH_MASTER_PORT}. The shared bootstrap — the endpoint derivation plus the `init_aclshmem` call every example and test uses — lives in `utils.py`; examples import it by plain name under `torchrun`, and the test harness puts this directory on `sys.path`.

| Environment variable | Default | Description |
| --- | ---: | --- |
| ASH_MASTER_ADDR | 127.0.0.1 | shmem master address |
| ASH_MASTER_PORT | 8666 | shmem master port |

Shape sweep parameters (they select which cases run and are used by both the tutorials and the perf scripts):

| Environment variable | Default | Description |
| --- | ---: | --- |
| `QKV_BENCH_S` | Script-defined shape set | QKV sequence lengths, comma-separated |
| `QKV_BENCH_H` | `1,2,3,4,5,6,7` | Number of heads per rank for QKV, comma-separated |
| `A2A_BENCH_S` | Script-defined shape set | Transpose UDMA sequence lengths, comma-separated |
| `A2A_BENCH_H` | Script-defined shape set | Number of heads per rank for transpose UDMA, comma-separated |

For event timing and speedup numbers, run the matching script under `python/triton_dist/test/ascend/`; see [python/triton_dist/test/ascend/README.md](../../python/triton_dist/test/ascend/README.md) for the `@prof` decorator and its `ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS` controls.

## Example Overview and Launch Instructions

`<torchrun_args>` in the tables below represents the launch arguments for `torchrun`.
For a single-node run with two NPUs, replace it with `--standalone --nproc_per_node=2`.
Adjust `--nproc_per_node` for other device counts. Do not enter the placeholder or angle brackets literally.

### Distributed GEMM

These examples use fixed matrix sizes (the code defaults to `M=N=K=4096`, FP16) and HCCL and ACLSHMEM on two or more devices. Each script runs both a Torch golden reference and a Triton kernel, then checks the results.

| File | Operator | Launch command |
| --- | --- | --- |
| `allgather-gemm.py` | AllGather fused with GEMM using MTE | `torchrun <torchrun_args> tutorials/ascend/allgather-gemm.py` |
| `gemm-reduce-scatter.py` | GEMM fused with ReduceScatter using MTE | `torchrun <torchrun_args> tutorials/ascend/gemm-reduce-scatter.py` |
| `gemm-reduce-scatter-udma.py` | GEMM fused with ReduceScatter using UDMA | `torchrun <torchrun_args> tutorials/ascend/gemm-reduce-scatter-udma.py` |
| `gemm-allreduce-oneshot.py` | GEMM fused with one-shot AllReduce using MTE | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-oneshot.py` |
| `gemm-allreduce-oneshot-notify.py` | GEMM fused with AllReduce using MTE and notify/wait synchronization | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-oneshot-notify.py` |
| `gemm-allreduce-udma.py` | GEMM fused with AllReduce using UDMA | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-udma.py` |

The following is a complete launch command for the GEMM and ReduceScatter fused operator:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/gemm-reduce-scatter.py
```

Inputs for these GEMM examples are generated on the device by each rank, so no additional input files are required.

### Reverse All-to-All

| File | Version/Purpose | Launch command |
| --- | --- | --- |
| `reverse-all2all.py` | MTE version, wait/notify pipeline | `torchrun <torchrun_args> tutorials/ascend/reverse-all2all.py` |
| `reverse-all2all-udma.py` | UDMA version | `torchrun <torchrun_args> tutorials/ascend/reverse-all2all-udma.py` |
| `reverse-all2all-udma-autotune.py` | UDMA autotune experiment entry point | See [UDMA Autotune](#udma-autotune) |

Run the MTE/UDMA version:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=9570 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-all2all-udma.py
```

By default, `04` iterates over the shape set defined in the script and checks every case.

#### UDMA Autotune

This example implements autotuning for the reverse-all-to-all operator:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=9579 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-all2all-udma-autotune.py
```

### QKV All-to-All

| File | Version | Device requirements |
| --- | --- | --- |
| `qkv-all2all.py` | MTE | An Ascend environment supporting the Triton/ACLSHMEM MTE kernel |
| `qkv-all2all-udma.py` | UDMA | An Ascend environment supporting UDMA |

Both files iterate over multiple sequence lengths and numbers of heads per rank by default. To run a smaller set of cases for a quick verification, use:

```bash
QKV_BENCH_S=2048,4096 QKV_BENCH_H=1,2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8666 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/qkv-all2all.py
```

Replace the filename with `qkv-all2all-udma.py` to test the UDMA version. For each case, the script runs the Torch golden reference first, then the Triton version, and performs an accuracy check.

### Transpose All-to-All

| File | Version | Launch method |
| --- | --- | --- |
| `transpose-all2all.py` | MTE | General launch method |
| `transpose-all2all-udma.py` | UDMA | General launch method |

MTE example:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8899 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/transpose-all2all.py
```

UDMA example (run only a small number of cases):

```bash
A2A_BENCH_S=2048,4096 A2A_BENCH_H=1,2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8899 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/transpose-all2all-udma.py
```

The UDMA version may print `SKIP (OOM)` for an individual case and continue with subsequent cases;
this is expected protective behavior.

### Fused Reverse A2A, RMSNorm, and MatMul

| File | Operator | Shapes |
| --- | --- | --- |
| `reverse-a2a-norm-matmul.py` | Reverse A2A of one tensor, followed by RMSNorm and then MatMul | `tensor [S_total, H, D]`, `norm_weight [D]`, `linear_weight [N, D]` |
| `reverse-a2a-norm-matmul-2.py` | Reverse A2A of two tensors, followed by two RMSNorm operations and one MatMul | `tensor_1`/`tensor_2 [S_total, H, D]`, `norm_weight [D]`, `linear_weight [N, 2*D]` |

Run the single default case (S=128, H=4, D=128, N=128):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-a2a-norm-matmul.py
```

| Environment variable | Default | Description |
| --- | ---: | --- |
| `FUSED_RUN_BENCH` | unset | Set to `1` to sweep all built-in cases instead of the single case |
| `FUSED_S` | `128` | Sequence length per rank for the single case |
| `FUSED_H` | `4` | Hidden groups per rank for the single case |
| `FUSED_D` | `128` | Feature dimension for the single case |
| `FUSED_N` | `128` | Output dimension (single-input variant only; dual-input variant uses `N = D`) |
| `FUSED_EPS` | `1e-6` | RMSNorm epsilon |
| `FUSED_ASH_SIZE_MB` | `1024` | shmem heap size per rank, MB (positive multiple of 2) |
| `FUSED_COMM_BLOCK_S` | `1024` | Communication block size along S |

### AllToAllV-GMM

The UDMA version fuses dispatch, local copy, ready waits and grouped GEMM into one mixed kernel. The unused output tail is initialized to zero once per fixed route, outside timing. It uses fixed tiles and expert-local `gemm_swizzle2d` scheduling; batch testing lives in `python/triton_dist/test/ascend/alltoallv-gmm-udma.py`.

| File | Version | Device requirement |
| --- | --- | --- |
| `alltoallv-gmm.py` | MTE | Atlas A2 products and compatible environments |
| `alltoallv-gmm-udma.py` | UDMA | Ascend 950PR&950DT products |

MTE two-card fixed case:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=auto ASH_MASTER_PORT=28980 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/alltoallv-gmm.py
```

UDMA two-card fixed case (Ascend 950PR&950DT products):

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=28980 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/alltoallv-gmm-udma.py
```
