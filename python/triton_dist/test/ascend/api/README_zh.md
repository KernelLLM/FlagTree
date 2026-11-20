# Ascend API 测试

本目录包含 Triton-distributed-ascend 的 API 测试，覆盖对称内存寻址、数据传输、信号同步、屏障和自动调优等功能。测试由 pytest 运行，用于验证接口行为及结果正确性。

## 运行前提

- 按照[安装](../../../../../docs_ascend/zh/getting-started/installation.md)和[从源码构建](../../../../../docs_ascend/zh/getting-started/build.md)完成环境准备，确保当前 Python 环境中已安装 pytest、PyTorch、torch_npu、Triton、`triton_dist` 和 shmem。
- 当前 shell 已加载 CANN 环境变量，并按构建文档配置 AscendNPU-IR 工具路径。
- 多数多卡测试使用 2 张 NPU，子通信组屏障测试使用 4 张 NPU。运行下方整组测试命令时，至少需要 4 张可见的 NPU。
- UDMA 运行测试还需要支持 UDMA 的 Ascend 950PR&950DT系列产品及对应的软件环境。

## 运行测试

在仓库根目录执行：

```bash
pytest python/triton_dist/test/ascend/api -m dist
```

`-m dist` 只选择标记为 `dist` 的用例，不代表运行本目录中的所有测试。多卡用例由测试框架自动创建工作进程并配置 rank，无需额外使用 `torchrun`。

如果只有 2 张 NPU，可以先运行等待与通知接口测试：

```bash
pytest python/triton_dist/test/ascend/api/test_wait_notify.py -m dist
```

## 查看结果

pytest 会在结束时汇总结果：`passed` 表示用例通过，`failed` 表示断言或执行失败，`error` 表示收集、准备或清理阶段出错，`skipped` 表示跳过，`deselected` 表示未被筛选条件选中。跳过或未选中的用例不代表验证通过。

出现失败时，根据 pytest 输出中的用例名称、异常堆栈以及 rank 信息定位问题。

## 传输后端运行测试

`dist` 包含 AUTO、MTE 和 UDMA 的运行测试。RoCE 运行测试等待官方启用，暂不添加 `dist` 标记；仅在已配置 RoCE 的环境中显式选择相应用例。

| API 类别 | 运行测试文件 | 覆盖范围 |
|---|---|---|
| 分组原子操作和 compare-swap | `test_atomic_transport_runtime.py` | AUTO/MTE/UDMA 支持的操作选择器；返回值及远程更新 |
| 底层 get/put 和 quiet | `test_rma_transport_runtime.py` | MTE、普通 UDMA 和 UDMA QP 0/1；带类型的指针偏移、字节计数及目标缓冲区边界保护 |
| `putmem_signal_nbi_lowlevel` | `test_rma_transport_runtime.py` | UDMA 为默认且唯一支持的传输方式；普通及 QP 变体、载荷与 uint64 通知值 |
| 初始化 UDMA 后的 `sync` / `sync_all` | `test_sync_ops.py` | 显式 UDMA put + quiet 后执行通用 team/world 同步，验证多轮传输结果 |
| Relay get/put 和 quiet | `test_rma_transport_runtime.py` | 源、目标和中继使用不同 PE；需要四张可见 NPU；不添加 `dist`/`ci` 标记 |
| RoCE 原子操作、RMA、屏障和 sync | `test_atomic_transport_runtime.py`、`test_rma_transport_runtime.py`、`test_roce_collective_runtime.py` | 已提供运行用例，等待官方启用，暂不纳入 `dist` |

UDMA QP 用例在 SHMEM 初始化前通过 `aclshmemx_set_qp_num` 配置两个 QP，需要直连模式的 SHMEM 和模板库。普通 UDMA 和 QP 用例默认在 Ascend950 上运行。

初始化 UDMA 后，可以使用默认 `transport=AUTO` 的 `sync` 和 `sync_all`，但其通用实现通过共享内存信号同步，不提交 UDMA 同步请求，也不等待未完成的 UDMA 队列操作。进入集合通信前，应调用 `quiet_lowlevel(peer, transport=ACLSHMEM_TRANSPORT_UDMA)` 等待这些操作完成。SHMEM 没有专用的 UDMA sync 入口，因此仍拒绝显式指定 `transport=UDMA`。

原子操作分发会检查 `is_compile_on_910_95()`。在 Ascend950 之前的设备上，仅支持 AUTO/MTE `atomic_add` 的 int8、int16、int32、float16、bfloat16 和 float32 类型；其他原子操作、类型和传输方式组合要求 Ascend950 或更新设备。ABI 测试显式模拟 Ascend950 目标，并另设用例验证旧设备上的拒绝行为。

所有前端 dispatch 和 lowering 测试（包括 RoCE 和 relay）均添加 `ci` 和 `dist` 标记，不需要硬件支持。

默认 NPU-IR 包未启用 relay，因此 relay 运行测试不添加 `dist` 或 `ci` 标记。Relay 运行用例需要四张可见 NPU、使用 `-enable_relay` 构建的 SHMEM，以及启用 `BISHENGIR_SHMEM_RELAY_SUPPORT=ON` 的编译器模板。匹配该构建配置时执行：

```bash
TRITON_DIST_TEST_UDMA_RELAY=1 pytest python/triton_dist/test/ascend/api/test_rma_relay.py python/triton_dist/test/ascend/api/test_rma_transport_runtime.py -k relay
```

SHMEM 没有 Python relay 能力查询接口，因此通过此环境变量声明构建配置；设置后允许显式选择的 relay 运行用例执行，并跳过不兼容的 UDMA QP 用例，不会添加测试标记。未设置时跳过 relay 运行用例。硬件或 SoC 不满足条件时，运行测试也会跳过；跳过不代表验证通过。
