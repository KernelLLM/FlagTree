# Ascend 算子精度与性能测试

本目录复用 [`tutorials/ascend/`](../../../../tutorials/ascend/README_zh.md) 中的 kernel 和 golden，支持精度校验、NPU Event 性能计时以及两者同时执行。

需要已安装 CANN、torch_npu、Triton、triton_dist 和 ACLSHMEM 的 Ascend 多卡环境。

## 启动方式与运行模式

在本目录启动全量脚本；默认参数为 `--mode=both --case=full`：

```bash
bash run.sh --mode=verify --case=full
bash run.sh --mode=perf --case=random
bash run.sh --mode=both --case=full
```

| 模式 | 行为 |
| --- | --- |
| `--mode=verify` | 只校验精度；`@prof` 每次调用只执行一次，不做预热、不创建计时 Event |
| `--mode=perf` | 预热并计时 kernel 和 golden，报告性能，不比较输出精度 |
| `--mode=both` | 执行性能计时和精度校验 |

用例选择规则：

- `--case=full`：运行脚本当前配置的全部 shape 和参数组合。
- `--case=random`：从上述组合中不放回抽取最多 10 组，并按原顺序运行。
- 组合不足 10 组时全部执行；固定单 shape 的脚本仍只运行该 shape。
- 抽样针对每个脚本的参数组合，不针对脚本列表。
- 环境变量对 shape 范围的配置先于抽样生效。

抽样复现规则：

- 所有 rank 使用相同的抽样种子。
- `run.sh` 在 random 模式下自动生成并打印 `ASCEND_CASE_SEED`；显式设置该变量可复现抽样。
- 直接使用 `torchrun` 启动时，优先使用 `ASCEND_CASE_SEED`；未设置时使用各 rank 共享的 `MASTER_PORT`，再回退到 `0`。

单个脚本可以从仓库根目录启动：

```bash
ASCEND_CASE_SEED=42 ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  python/triton_dist/test/ascend/qkv-all2all.py --mode=both --case=random
```

启动器行为：

- 按文件名顺序运行本目录中含 `__main__` 入口的 `.py` 文件，跳过辅助模块，不递归运行 `api/` 或 `unit/`。
- UDMA 用例仅在检测到 Ascend 950PR&950DT系列产品时运行，检测依据为 `npu-smi info` 或 `ASCEND_SOC_VERSION`。
- 一个脚本失败后继续执行其余脚本，最后汇总 PASS/FAIL/SKIP。
- 存在脚本失败时退出码为 `1`，参数错误时为 `2`。

| 运行环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `NPROC_PER_NODE` | `2` | 每个脚本的进程数 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,...,NPROC_PER_NODE-1` | 可见设备列表 |
| `SKIP_UDMA` | `0` | 设为 `1` 时跳过全部 UDMA 用例，包括 MoE 和融合 Reverse A2A |
| `RUN_AUTOTUNE` | `0` | 设为 `1` 时包含文件名带 `autotune` 的入口；当前目录无此类入口 |
| `ASCEND_CASE_SEED` | 自动生成 | random 模式的可复现抽样种子 |

通信地址与端口配置：

- 启动器为每个脚本设置独立的 `ASH_MASTER_PORT`，端口从 31007 开始，每次增加 7。
- 直接启动脚本时，使用各脚本自身的通信地址和端口默认值；基于 shmem 的脚本还支持 `ASH_MASTER_ADDR`。
- 启动器不设置 `HCCL_NPU_SOCKET_PORT_RANGE`。

## @prof 装饰器（`prof.py`）

`prof.py` 提供共享的计时装饰器：

```python
from prof import check_all_ranks_close, load_tutorial, prof, prof_report

tut = load_tutorial("allgather-gemm")   # 按路径导入 tutorials/ascend/<name>.py

@prof(label="allgather_gemm")                     # 每次调用包含预热和计时轮次
def run_kernel():
    return tut.allgather_gemm(A, B, C, peer_mem, ...)

@prof(label="torch_allgather_gemm")
def run_golden():
    return tut.torch_allgather_gemm(A, B, world_size)

C_golden = run_golden()
run_kernel()
prof_report(run_golden, run_kernel)               # rank 0 打印 EVENT_TIMING 行
check_all_ranks_close(C, C_golden, rtol=1e-3, atol=1e-3, context="C")
```

每一轮的流程是 `prepare() -> barrier -> sync -> invoke -> sync -> barrier`，计时轮额外在 invoke 两侧记录 NPU event，因此状态重置的开销不会进入计时窗口：

- `prepare=` — 可选钩子，在每轮开头、barrier 之前、event 窗口之外执行。用于必须在每次 launch 前完成的重置，例如 wait/notify kernel 的 `signal_mem.fill_(0)`，或累加型 kernel 之前的 `C.zero_()`。
- `barrier=` — 可选的自定义 rank 同步。默认 `dist.barrier()`。
- `label=` — `EVENT_TIMING` 输出中使用的名称。
- `warmup=` / `iters=` — 轮数。解析优先级：`ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS` 环境变量（若已设置）> 装饰器参数 > 默认值（`5` / `50`）。

计时结果保存在包装函数的 `.prof` 属性上（`run_kernel.prof.times_ms`）；每次调用都会覆写，因此 shape 循环脚本可以逐 shape 输出。

不同模式下的行为：

- `verify` 模式下，`.prof` 为 `None`，`prof_report()` 不输出计时报告。
- `perf` 模式下，`check_all_ranks_close()` 不执行精度比较。
- 自定义校验和统计流程也按相同模式切换。

`load_tutorial("allgather-gemm")` 从源码仓库加载对应教程；运行这些脚本需要保留教程目录。

## 计时与 shape 参数

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `ASCEND_PROFILE_WARMUP` | `5`（装饰器默认） | 每个被计时函数的预热轮数，必须非负 |
| `ASCEND_PROFILE_ITERS` | `50`（装饰器默认） | 每个被计时函数的计时轮数，必须大于零 |
| `MOE_PROFILE_WARMUP` / `MOE_PROFILE_ITERS` | `5` / `50` | MoE 脚本传给装饰器的轮数；显式设置 `ASCEND_PROFILE_*` 时仍优先使用后者 |

各脚本的计时与校验配置：

- AllToAllV UDMA 默认使用 3 次预热、10 次计时，覆盖装饰器默认值。
- AllToAllV MTE 的 `@prof` 使用公共默认值；预热配置（默认 3）非零时，另行记录首次预热耗时。
- 两种 AllToAllV 脚本均支持 `TRITON_VALIDATION_ITERS`（默认 1）；verify/both 模式至少校验一次，perf 模式跳过这些校验。
- verify 模式不执行首次预热计时。

shape 扫描范围配置：

- QKV：使用 `QKV_BENCH_S` 和 `QKV_BENCH_H` 配置扫描范围。
- Transpose UDMA：使用 `A2A_BENCH_S` 和 `A2A_BENCH_H` 配置扫描范围。
- 融合 Reverse A2A：默认运行一个 shape；设置 `FUSED_RUN_BENCH=1` 后扫描内置组合。
- MoE：使用 `MOE_N_LIST`、`MOE_TOPK_LIST`、`MOE_NLE_LIST` 和 `MOE_ROUTE_MODES` 配置组合。

random 模式从上述配置形成的组合中抽样。

从仓库根目录快速运行性能测试：

```bash
ASCEND_PROFILE_WARMUP=1 ASCEND_PROFILE_ITERS=2 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  python/triton_dist/test/ascend/allgather-gemm.py --mode=perf --case=full
```

Dispatch GMM UDMA 的 `dispatch-gmm-udma.py` 复用只有一个示例的交付算子，
测试36组 random 路由组合，覆盖空桶、尾块和超过 INT32 的地址偏移。
仅 GMM 输出使用 `rtol=atol=1e-3`，公开输入布局和计数列表要求完全相同。
`verify` 对双方各执行一次；`perf` / `both` 对双方均默认预热10次、计时30次，
可用 `ASCEND_PROFILE_WARMUP` / `ASCEND_PROFILE_ITERS` 覆盖。
计时包含路由求逆、计数和偏移交换、pack、UDMA、公开布局恢复、GMM及计数列表回传，
每轮取各 rank 最大 Event 耗时。输入构造和容量分配不计时。
交付示例只运行一轮正确性检查；更换输入和复用工作区的多轮验证
由专门测试脚本负责。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 ASH_MASTER_PORT=9598 \
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29798 \
  python/triton_dist/test/ascend/dispatch-gmm-udma.py --mode=both --case=full
```

## 脚本总览

下表列出当前算子入口。`prof.py` 是公共辅助模块。

| 脚本 | 算子 | 备注 |
| --- | --- | --- |
| `all-to-all-single-2d.py` | UDMA All-to-All-Single 2D | 288 矩阵 case 覆盖 (M, N, dtype, split_pattern) 组合 |
| `allgather.py` | 基于 UDMA 和 notify/wait 同步机制的 AllGather 纯通信算子 | 108 个矩阵 case，以及 9 个边界 case |
| `allgather-gemm.py` | 基于 MTE 的 AllGather 与 GEMM 融合算子 | 单个固定 shape |
| `allgather-gemm-udma.py` | 基于 UDMA 的 AllGather 与 GEMM 融合算子 | 从同目录 `allgather-gemm-udma.csv`读取并遍历M/K/N组合 |
| `alltoallv-gmm-udma.py` | 基于 UDMA 的 AllToAllV 与 GEMM 融合算子 | 读取 `gmm-alltoallv-cases.csv` 固定 300 组；FP16，topk=1/2/8，每卡专家数=8/40/160；统一索引重排 reference |
| `alltoallv-gmm.py` | 基于MTE的AllToAll与GEMM 融合算子 | 通过 `TRITON_M/K/N` 配置单个 shape；单独报告首次预热耗时 |
| `gemm-allreduce-oneshot-notify.py` | 采用 MTE 和 notify/wait 同步机制的 GEMM 与 AllReduce 融合算子 | 单个固定 shape；重置 `C` 和 `signal_mem` |
| `gemm-allreduce-oneshot.py` | 基于 MTE 的 GEMM 与 one-shot AllReduce 融合算子 | 单个固定 shape；`prepare=C.zero_` |
| `gemm-allreduce-udma.py` | 基于 UDMA 的 GEMM 与 AllReduce 融合算子 | BF16；从 `allgather-gemm-case.csv` 读取 M/K/N，从 `gemm-allreduce-udma-config.csv` 读取调优配置；根据 shape 选择 one-shot 或 Owner-RSAG |
| `gemm-reduce-scatter-udma.py` | 基于 UDMA 的 GEMM 与 ReduceScatter 融合算子 | 单个固定 shape；归一化误差校验 |
| `gemm-reduce-scatter.py` | 基于 MTE 的 GEMM 与 ReduceScatter 融合算子 | 单个固定 shape；`prepare=C.zero_` |
| `moe-combine-udma.py` | UDMA MoE Combine | 完整计算流程计时；每轮取各 rank 最大耗时；MoE 参数扫描 |
| `moe-dispatch-udma.py` | UDMA MoE Dispatch | 完整计算流程计时；每轮取各 rank 最大耗时；MoE 参数扫描 |
| `qkv-all2all-udma.py` | UDMA QKV All-to-All | `QKV_BENCH_S` / `QKV_BENCH_H` |
| `qkv-all2all.py` | MTE QKV All-to-All | `QKV_BENCH_S` / `QKV_BENCH_H` |
| `reverse-a2a-norm-matmul-2.py` | 基于 UDMA 的 Reverse A2A、RMSNorm 与 MatMul 融合算子 | 双输入版本；`FUSED_RUN_BENCH=1` 开启 shape 扫描 |
| `reverse-a2a-norm-matmul.py` | 基于 UDMA 的 Reverse A2A、RMSNorm 与 MatMul 融合算子 | 单输入版本；`FUSED_RUN_BENCH=1` 开启 shape 扫描 |
| `reverse-all2all-udma.py` | UDMA Reverse All-to-All | 抽样前有 45 × 11 组 shape 组合 |
| `reverse-all2all.py` | MTE Reverse All-to-All | 内置序列长度和头数扫描 |
| `sp_ulysses_qkv_gemm_all2all.py` | Ulysses QKV GEMM 与 All2All 融合算子 | BF16/FP16；独立 `--hidden-dim K`、`--out-features N`、全局 `--seq-len S_total`、`--head-dim D`、`--gqa G`（batch size 为 1） |
| `transpose-all2all-udma.py` | UDMA Transpose All-to-All | `A2A_BENCH_S` / `A2A_BENCH_H`；保留 `SKIP (OOM)` |
| `transpose-all2all.py` | MTE Transpose All-to-All | 内置序列长度和头数扫描 |
| `dispatch-gmm-udma.py` | UDMA Dispatch GMM | 36组 random case；完整流程 max-rank Event；BF16输出容差1e-3 |
