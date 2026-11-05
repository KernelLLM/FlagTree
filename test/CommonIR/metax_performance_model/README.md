# MetaX 指令测量与 Layout 成本评估

面向 MetaX C550（`xcore1000`、wave64）的实验工具，提供 lowering 规则查询、受控指令测量和 GEMM Layout 排序评估。成本单位为 ticks；预测包含显式的访存假设，用于受限配置内的对照实验。

## 目录

| 路径 | 功能 |
|---|---|
| `catalog/` | 源码/IR 到 LLVM 的 lowering 规则与查询 |
| `measurement/` | 微基准构建、Native 核验、正确性检查、采样和结果发布 |
| `cost_model/` | LLVM 事件提取、依赖分析、Timeline 评分和 GEMM 排序评价 |
| `run.py`、`reproduce.py` | 命令行入口与端到端流程 |
| `tests/` | 单元测试与 Native 编译核验测试 |

## 环境要求

- Python 3.10+、PyYAML；测试使用 pytest 9+。
- 编译与 Native 核验：MACA SDK、`mxcc`、`readelf`、`g++`。
- GPU 测量：C550 和 MACA runtime。
- GEMM 编译：已构建的 FlagTree MetaX 后端，设置 `FLAGTREE_BACKEND=metax`，保持 `FLAGTREE_COMMON_IR` 未设置。

已验证 SDK 为 MACA 3.7.2。SDK 默认读取 `MACA_PATH`，未设置时使用 `/opt/maca`；构建命令支持 `--sdk`、`--mxcc`。`--device` 使用 `MACA_VISIBLE_DEVICES` 映射后的设备编号。

以下命令均在本目录执行。输出路径须位于 FlagTree 源码树之外，且尚不存在。

```bash
python -m pip install PyYAML 'pytest>=9'
```

## 使用方法

### 查询 lowering 规则

```bash
python -B run.py catalog --source tl.load
python -B run.py catalog --llvm llvm.mxc.mma
python -B run.py catalog --source tensor.to --text 'f32 -> bf16'
```

Catalog 记录 lowering 条件和源码位置，不参与测量或评分。

### 测量指令成本

```bash
# 列出测量定义
python -B run.py measurements

# 仅构建和核验，不执行 GPU 采样
python -B run.py qualify --measurement fadd_f32.independent_issue \
  --output /path/to/new-qualification

# 构建、核验、采样并发布通过检查的结果
python -B run.py measure --measurement fadd_f32.independent_issue \
  --output /path/to/new-measurement --device 0

# 重新分析已采集的样本
python -B run.py analyze --run /path/to/new-measurement \
  --output /path/to/new-analysis.json
```

`independent_issue` 测量独立指令的发射间隔；`dependent_latency` 测量指定操作数角色的同类指令 RAW 间隔。后者不代表任意消费者的完成延迟。

每个微基准使用固定直线二进制，窗口内无循环、分支和 wait。N 为 0/1/2/3/4/5/6/8，bpermute issue 为 0/1/2/3/4。协议要求：

- 最终 ELF 与独立汇编 object 的函数字节一致；目标指令、寄存器绑定、地址及非目标代码和资源符合定义。
- 计时窗口位于同一 4 KiB 页，覆盖实际 PC mod32 为 0/8/16/24 的相位。
- 两个独立进程，每个二进制预热 8 次、采样 64 次；正确性检查覆盖全部 64 lanes 和结果分量。
- 样本及跨 N、进程、相位的指令增量满足严格一致性检查，不进行回归拟合。

仅通过全部检查时生成 `cost_result.json`。复用结果时应保留完整测量目录；loader 核对 qualification、分析间隔、measurement ID、计时协议和成本适用条件。

### 评估 GEMM Layout

复用已有测量结果：

```bash
python -B run.py layouts --costs /path/to/qualified-measurements \
  --output /path/to/new-layout-run --device 0
```

重新测量端到端发布集合，再执行同一 Layout 实验：

```bash
python -B run.py reproduce --output /path/to/new-run --device 0
```

当前固定域为 4 个 Shape：`16×32×32`、`32×32×64`、`32×64×64`、`64×32×128`；每个 Shape 使用 6 种显式 encoding，共 24 个候选。候选定义见 `cost_model/layouts.py`，端到端测量集合见 `FORMAL_MEASUREMENTS`。

GEMM 语义为 `A@B+C`，A/B 为 F16，C/输出为 F32，Global 存储均为 row-major。每个候选使用单 CTA、64 lanes、BLOCK_K=32、一个 warp/stage、basic pipeline，关闭 FP fusion。

流程为 TTGIR encoding 替换 → LLVM/Native 编译与核验 → 保存预测 → GPU 正确性与计时 → 排序评价。计费指令的 LLVM/Native 数量或宽度映射不一致时停止流程。K 已静态展开，不重复乘迭代次数。

计时使用全 1 输入；计时前后另以有符号非均匀输入检查全部输出。两个进程各预热 8 次、随机交错采样 64 次，进程中位数相对差异须不超过 5%。失败样本不从其他运行补齐。

### 重新评分已有结果

```bash
python -B -m cost_model.model \
  --kernels /path/to/layout-run/layouts \
  --costs /path/to/qualified-measurements \
  --output /path/to/new-predictions
```

输入须包含编译产物和两进程采样结果。该命令重新生成预测和评价，不重新编译或执行 GPU 采样。

## 模型与评价约定

预测输入为 LLVM SSA 和 TTGIR/MLIR 注解；最终 Native 仅用于离线核验。独立操作按源码分组，支付 `count × I`，保留 `max(D − I, 0)` 的 pending；独立工作消耗 pending，RAW 消费者暴露剩余等待。Wait/barrier 排空 pending，Shared 转换按 store/load 阶段计费。

实测成本须匹配操作数、输入限制及 RAW 角色。当前操作数证明限于全 1 计时输入。未知操作、缺失成本或无法证明的适用条件产生 model gap，该候选预测为 null。

访存使用以下代表性假设，不作为合格 Memory D 发布：

| 类别 | Issue I | Pending 参数 D |
|---|---|---:|
| LDS | 4，b32 代表值 | 64 |
| STS | 实测 STS b32 issue，跨宽度使用代表值 | 32 |
| Global load/store | 实测 FADD issue 代表值 | 256 |

地址/view/fence 的直接成本为 0，保留依赖传播。Bank conflict、Spill、cache、取指、资源竞争及 wait 服务成本未完整建模；输出是局部 Timeline 分数，不是完整 kernel 延迟。

所有评价均在同一 Shape 内进行。Pairwise/Spearman 使用可预测子集；Top-1 regret 和 Top-k coverage 使用全部合格候选作为 Oracle，并单独报告预测覆盖率。全部 abstain 时评价指标为 null。

## 输出

- `measurements/`：端到端重测结果，仅 `reproduce` 生成。
- `layouts/`：IR/ISA、二进制、核验信息、正确性和原始样本。
- `ranking/`：成本表、事件特征、Timeline trace、预测和排序指标。
- `summary.json`：测量与重复性检查状态、各 Shape 评价。

端到端流程的临时文件和 cache 位于 `output/work/`，退出时清理。生成产物不应提交到 Git。

## 测试

```bash
mkdir -p /path/to/test-work/tmp
TMPDIR=/path/to/test-work/tmp PYTHONDONTWRITEBYTECODE=1 \
  python -B -m pytest -q -p no:cacheprovider -m 'not compilation' \
  --basetemp=/path/to/test-work/pytest tests
```

上述测试使用合成输入，不编译 Native 或执行 GPU。将标记改为 `-m compilation` 可运行需要 MACA SDK 的 ELF、opcode、RAW 和 page 核验测试。
