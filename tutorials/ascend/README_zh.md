# Ascend Triton 算子用例

本目录包含 Triton Distributed Ascend 的独立算子示例。每个 `.py` 文件都可以直接作为一个运行/校验入口：脚本会执行一次 Torch Golden 参考实现和一次 Triton kernel，然后校验结果。
Event 计时和 speedup 测试放在[`python/triton_dist/test/ascend/`](../../python/triton_dist/test/ascend/README.md) 目录；除特别注明的autotune 用例外，不需要额外的 analysis 或 pytest 驱动。

## 运行前提

- Ascend NPU、CANN/torch_npu、Triton 和 `triton_dist` 已安装并可在当前 Python 环境导入。
- 多卡用例使用 `torchrun` 和 HCCL；通常需要至少 2 张卡。
- `ASCEND_RT_VISIBLE_DEVICES` 应与 `--nproc_per_node` 的进程数一致，例如 `0,1` 对应 2 进程。
- 当前 shell 已在仓库根目录时，可直接使用下文命令；否则请把脚本路径改为绝对路径。

先确认环境：

```bash
python - <<'PY'
import torch
import torch_npu  # noqa: F401
import triton
import triton_dist
print("torch/triton/torch_npu/triton_dist import: OK")
PY
```

## 通用启动方式

多数示例按下面方式启动：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/<用例文件>.py
```

shmem 初始化地址由 ASH_MASTER_ADDR 和 ASH_MASTER_PORT 控制，脚本会拼成 tcp://{ASH_MASTER_ADDR}:{ASH_MASTER_PORT}。共享的初始化逻辑（地址推导和所有示例、测试共用的 `init_aclshmem` 调用）位于 `utils.py`；示例在 `torchrun` 下按同目录模块名直接导入，测试侧由 prof/conftest 把该目录加入 `sys.path`。

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| ASH_MASTER_ADDR | 127.0.0.1 | shmem master 地址 |
| ASH_MASTER_PORT | 8666 | shmem master 端口 |

shape 遍历参数（选择要跑的 case，教程和 perf 脚本通用）：

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `QKV_BENCH_S` | 脚本默认 shape 集合 | QKV 的序列长度，逗号分隔 |
| `QKV_BENCH_H` | `1,2,3,4,5,6,7` | QKV 每 rank 的 head 数，逗号分隔 |
| `A2A_BENCH_S` | 脚本默认 shape 集合 | transpose UDMA 的序列长度，逗号分隔 |
| `A2A_BENCH_H` | 脚本默认 shape 集合 | transpose UDMA 每 rank 的 head 数，逗号分隔 |

需要 Event 计时和 speedup 数据时，请运行 `python/triton_dist/test/ascend/` 下的对应脚本；`@prof` 装饰器和 `ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS` 的说明见 [python/triton_dist/test/ascend/README_zh.md](../../python/triton_dist/test/ascend/README_zh.md)。

## 用例总览与运行指引

下列表格中的 `<torchrun_args>` 表示 `torchrun` 启动参数。以单机双卡为例，请将其替换为 `--standalone --nproc_per_node=2`；其他设备数量请相应调整 `--nproc_per_node`。
占位符及尖括号不应原样输入。

### 分布式 GEMM

这些示例是使用 HCCL 和 ACLSHMEM 的两卡或多卡用例，矩阵尺寸固定（代码默认 `M=N=K=4096`、FP16）。每个脚本都会执行 Torch Golden 与 Triton kernel，并校验结果。

| 文件 | 算子 | 启动命令 |
| --- | --- | --- |
| `allgather-gemm.py` | 基于 MTE 的 AllGather 与 GEMM 融合算子 | `torchrun <torchrun_args> tutorials/ascend/allgather-gemm.py` |
| `gemm-reduce-scatter.py` | 基于 MTE 的 GEMM 与 ReduceScatter 融合算子 | `torchrun <torchrun_args> tutorials/ascend/gemm-reduce-scatter.py` |
| `gemm-reduce-scatter-udma.py` | 基于 UDMA 的 GEMM 与 ReduceScatter 融合算子 | `torchrun <torchrun_args> tutorials/ascend/gemm-reduce-scatter-udma.py` |
| `gemm-allreduce-oneshot.py` | 基于 MTE 的 GEMM 与 one-shot AllReduce 融合算子 | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-oneshot.py` |
| `gemm-allreduce-oneshot-notify.py` | 采用 MTE 和 notify/wait 同步机制的 GEMM 与 AllReduce 融合算子 | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-oneshot-notify.py` |
| `gemm-allreduce-udma.py` | 基于 UDMA 的 GEMM 与 AllReduce 融合算子 | `torchrun <torchrun_args> tutorials/ascend/gemm-allreduce-udma.py` |

下面以 GEMM 与 ReduceScatter 融合算子为例，给出完整的启动命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/gemm-reduce-scatter.py
```

这些 GEMM 示例的输入由各 rank 在设备上生成，不需要额外的输入文件。

### Reverse All-to-All

| 文件 | 版本/用途 | 启动命令 |
| --- | --- | --- |
| `reverse-all2all.py` | MTE版本，wait/notify 流水线 | `torchrun <torchrun_args> tutorials/ascend/reverse-all2all.py` |
| `reverse-all2all-udma.py` | UDMA 版本 | `torchrun <torchrun_args> tutorials/ascend/reverse-all2all-udma.py` |
| `reverse-all2all-udma-autotune.py` | UDMA autotune 实验入口 | 见 [UDMA 自动调优](#udma-autotune) |

运行 MTE/UDMA 版本：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=9570 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-all2all-udma.py
```

`04` 默认会遍历脚本内定义的 shape 集合并校验每个 case。

#### UDMA Autotune

该用例实现 reverse-all-to-all 算子的自动调优功能：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=9579 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-all2all-udma-autotune.py
```

### QKV All-to-All

| 文件 | 版本 | 设备要求 |
| --- | --- | --- |
| `qkv-all2all.py` | MTE | 支持该 Triton/ACLSHMEM MTE kernel 的 Ascend 环境 |
| `qkv-all2all-udma.py` | UDMA | 支持 UDMA 的 Ascend 环境 |

两个文件都默认遍历多个序列长度和每 rank head 数。为了快速验证，可限制 case：

```bash
QKV_BENCH_S=2048,4096 QKV_BENCH_H=1,2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8666 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/qkv-all2all.py
```

将文件名替换为 `qkv-all2all-udma.py` 即可测试 UDMA 版本。每个 case 会先运行 Torch Golden，再运行 Triton 版本，最后做精度校验。

### Transpose All-to-All

| 文件 | 版本 | 启动方式 |
| --- | --- | --- |
| `transpose-all2all.py` | MTE | 通用启动方式 |
| `transpose-all2all-udma.py` | UDMA | 通用启动方式 |

MTE 示例：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8899 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/transpose-all2all.py
```

UDMA 示例（只跑少量 case）：

```bash
A2A_BENCH_S=2048,4096 A2A_BENCH_H=1,2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=8899 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/transpose-all2all-udma.py
```

UDMA 版本可能在单个 case 分配失败时输出 `SKIP (OOM)` 并继续后续 case；这是预期的保护行为。

### Reverse A2A、RMSNorm 与 MatMul 的融合实现

| 文件 | 算子 | 形状 |
| --- | --- | --- |
| `reverse-a2a-norm-matmul.py` | 对单个输入执行 Reverse A2A，再依次执行 RMSNorm 和 MatMul | `tensor [S_total, H, D]`、`norm_weight [D]`、`linear_weight [N, D]` |
| `reverse-a2a-norm-matmul-2.py` | 对两个输入执行 Reverse A2A，随后执行两次 RMSNorm 和一次 MatMul | `tensor_1`/`tensor_2 [S_total, H, D]`、`norm_weight [D]`、`linear_weight [N, 2*D]` |

运行单个默认 case（S=128、H=4、D=128、N=128）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/reverse-a2a-norm-matmul.py
```

| 环境变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `FUSED_RUN_BENCH` | 未设置 | 设为 `1` 时遍历脚本内置的全部 case，而不是单个 case |
| `FUSED_S` | `128` | 单 case 的每 rank 序列长度 |
| `FUSED_H` | `4` | 单 case 的每 rank hidden groups 数 |
| `FUSED_D` | `128` | 单 case 的特征维度 |
| `FUSED_N` | `128` | 输出维度（仅单输入版本；双输入版本固定 `N = D`） |
| `FUSED_EPS` | `1e-6` | RMSNorm 的 epsilon |
| `FUSED_ASH_SIZE_MB` | `1024` | 每 rank 的 shmem 堆大小，单位 MB（须为 2 的倍数） |
| `FUSED_COMM_BLOCK_S` | `1024` | S 维度的通信分块大小 |

### AllToAllV-GMM

UDMA 版本将分发、本地拷贝、就绪等待和分组 GEMM 融合在一个混合 kernel 中；固定路由的无效输出尾部在计时前一次性清零。使用固定分块和 expert 内的 `gemm_swizzle2d` 调度；批量测试脚本为 `python/triton_dist/test/ascend/alltoallv-gmm-udma.py`。

| 文件 | 版本 | 设备要求 |
| --- | --- | --- |
| `alltoallv-gmm.py` | MTE | Atlas A2系列产品及兼容环境 |
| `alltoallv-gmm-udma.py` | UDMA | Ascend 950PR&950DT系列产品 |

MTE 双卡固定用例：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=auto ASH_MASTER_PORT=28980 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/alltoallv-gmm.py
```

UDMA 双卡固定用例（Ascend 950PR&950DT系列产品）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
HCCL_NPU_SOCKET_PORT_RANGE=auto \
ASH_MASTER_ADDR=127.0.0.1 ASH_MASTER_PORT=28980 \
torchrun --standalone --nproc_per_node=2 \
  tutorials/ascend/alltoallv-gmm-udma.py
```
