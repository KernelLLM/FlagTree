# TLA API 参考手册

> 源码：`triton-ascend` GitHub main `third_party/ascend/language/tla/`
> 内核入口：`@tla.jit`（包装 Triton JIT）
> 编译目标：HIVM/AVE dialect → bishengir → NPU binary

---

## 目录

1. [核心类型](#1-核心类型)
2. [内存分配与视图](#2-内存分配与视图)
3. [DMA 数据搬运](#3-dma-数据搬运)
4. [Cube 矩阵计算](#4-cube-矩阵计算)
5. [同步原语](#5-同步原语)
6. [SIMD 向量寄存器操作](#6-simd-向量寄存器操作)
7. [Scope 核心区域标注](#7-scope-核心区域标注)
8. [标量工具](#8-标量工具)
9. [数学辅助](#9-数学辅助)
10. [编译器提示](#10-编译器提示)
11. [调试](#11-调试)
12. [选择操作](#12-选择操作)
13. [DMA 参数 dataclass 完整列表](#13-dma-参数-dataclass-完整列表)
14. [MXfp8 matmul 流水示例](#14-mxfp8-matmul-流水示例)

---

## 1. 核心类型

### `tla.AddressSpace`

Ascend NPU 物理地址空间枚举。

| 枚举成员 | 字符串别名 | 含义 |
|---------|-----------|------|
| `GM` | `"gm"` | 全局内存（HBM） |
| `UB` | `"ub"` | 统一缓冲区（片上，向量核直接操作） |
| `L1` | `"l1"` | L1 缓冲区（Cube 侧 ND2NZ 目标） |
| `L0A` | `"l0a"` | L0A 缓冲区（Cube 左矩阵） |
| `L0B` | `"l0b"` | L0B 缓冲区（Cube 右矩阵） |
| `L0C` | `"l0c"` | L0C 累加器（matmul 输出） |
| `BT` | `"bt"` | 偏置缓冲区（fixpipe 用） |
| `FB` | `"fb"` | fixpipe 缓冲区 |
| `SSBUF` | `"ssbuf"` | 软件同步缓冲区 |
| `L0A_SF` | — | 保留（MX scale，`NotImplementedError`） |
| `L0B_SF` | — | 保留（MX scale，`NotImplementedError`） |

### `tla.buffer_type(shape, dtype, space, strides=None, offset=0)`

构建 buffer 类型描述符（不分配内存）。`shape` 为整数元组，`dtype` 为 `tl.*` dtype，`space` 为 `AddressSpace` 或字符串。

### `tla.buffer`

物理地址空间 buffer 实例类型，由 `tla.alloc`、`tla.view`、`tla.subview` 产生。核心属性：
- `.shape` — 整数元组
- `.dtype` — 元素类型
- 支持 `buf[i0, i1, ...]` 下标访问（返回 `buffer_access`）

### `tla.vector_type(lanes, dtype)`

SIMD 寄存器类型描述符。`lanes` 为 2048 / elem_bits（固定宽度 2048-bit 寄存器）：

| dtype | lanes |
|-------|-------|
| `int8/uint8/fp8e4nv/fp8e5` | 256 |
| `int16/uint16/float16/bfloat16` | 128 |
| `int32/uint32/float32` | 64 |
| `int64/uint64` | 32 |
| `int1`（predicate） | 随上下文 |

### `tla.vector`

SIMD 寄存器实例类型，由 `tla.simd.*` 操作产生或返回。不可直接构造，必须通过 simd 操作得到。

### 辅助函数

```python
tla.space_of(value)     # 返回 buffer/vector 的 space 字符串，标量返回 None
tla.is_ub(value)        # True 若 value 在 UB 或为标量
tla.is_gm(value)        # True 若 value 在 GM 或为标量
```

---

## 2. 内存分配与视图

### `tla.alloc(shape, dtype, space=None, versions=1, *, scope=None)`
### `tla.alloc_buffer(...)`（别名）

分配片上 buffer（不可分配 GM），返回 `tla.buffer`。

- `space`/`scope` 二选一，默认 `"ub"`
- `versions`：多缓冲版本数（1=不多缓冲），配合 `annotate_buffer_versions`
- GM 必须用 `tla.view` 绑定
- L0A/L0B/L0C 须在 `tla.scope.cube()` 内分配

### `tla.view(src, shape, strides=None, offsets=None, space="gm")`

从 GM 指针（`tl.tensor` ptr）或已有 buffer 创建视图。

- **ptr 源**：lowers 到 `pointer_cast` + `memref.reinterpret_cast`；支持任意 strides/offsets
- **buffer 源**：等价 `subview`，`space` 参数被忽略
- `offsets` 可含运行时整数 tensor

### `tla.subview(src, offsets, sizes, strides=None)`

rank-preserving 切片（`memref.subview`）。`sizes`/`strides` 编译期正整数；`offsets` 可含运行时标量。

### `tla.reshape(src, shape)`

连续行主序 buffer 的 rank 变换（`collapse_shape` / `expand_shape`）。要求 buffer 连续。

---

## 3. DMA 数据搬运

### 控制枚举

| 枚举 | 成员 |
|------|------|
| `tla.PadMode` | `PadNull / PadFirstElem / PadValue` |
| `tla.EvictionPolicy` | `EvictNormal / EvictFirst / EvictLast` |
| `tla.AtomicKind` | `NONE / ADD / MAX / MIN / AND / OR / XOR / UMAX / UMIN` |
| `tla.UnitFlagMode` | `DISABLED(0) / RESERVED(1) / ENABLED_WITHOUT_UPDATE(2) / ENABLED_WITH_UPDATE(3)` |
| `tla.FixpipeDMAMode` | `NZ2ND / NZ2DN / NZ2NZ` |
| `tla.FixpipeDualDstMode` | `NO_DUAL / ROW_SPLIT / COLUMN_SPLIT` |
| `tla.FixpipeSubBlock` | `SUB_BLOCK_0 / SUB_BLOCK_1` |
| `tla.FixpipePreQuantMode` | `NO_QUANT / F322F16 / S322I8 / QF322F32_PRE / F322BF16` |
| `tla.FixpipePreReluMode` | `NO_RELU / NORMAL_RELU / LEAKY_RELU / P_RELU` |

### 路由函数

```python
tla.load_gm_to_ub(src, dst, params=None)              # GM→UB；params: LoadGM2UBParams
tla.store_ub_to_gm(src, dst, params=None)             # UB→GM；params: StoreUB2GMParams（含原子写）
tla.load_gm_to_l1(src, dst, params=None)              # GM→L1（ND2NZ）；params: LoadGM2L1Params
tla.load_gm_to_l1_mx_scale(src, dst)                  # GM→L1 MX E8M0 scale DMA（950 专用）
tla.copy_ub_to_l1(src, dst, params=None)              # UB→L1；params: CopyUB2L1Params
tla.copy_l1_to_ub(src, dst, params=None)               # L1→UB；params: CopyL12UBParams
tla.copy_l1_to_bt(src, dst, params=None)              # L1→BT；params: CopyL12BTParams
tla.copy_l1_to_fb(src, dst, params=None)              # L1→FB；保留，NotImplementedError
tla.copy_l1_to_l0(src, dst, params=None)              # L1→L0A/L0B；params: CopyL1ToL0Params
tla.store_l0c_to_gm(src, dst, params=None)            # L0C→GM（Fixpipe）；params: FixpipeL0C2GMParams
tla.copy_l0c_to_ub(src, dst, params=None)             # L0C→UB（Fixpipe）；params: FixpipeL0C2UBParams
tla.copy_l0c_to_l1(src, dst, params=None)             # L0C→L1（Fixpipe）；params: FixpipeL0C2L1Params
tla.copy(src, dst, params=None)                       # 统一路由分派（按 src/dst space 查表）
tla.set_ctrl(enable: bool, idx: int)                   # 设置/清除硬件 CTRL bit
tla.set_atomic(kind: AtomicKind, dtype)                # 持久设置后续 GM store 原子模式
```

Fixpipe 路由（L0C→*）须在 `tla.scope.cube()` 内调用。

---

## 4. Cube 矩阵计算

### `tla.MxFormat`（Enum）

| 成员 | 说明 |
|------|------|
| `E5M2` | FP8 E5M2 |
| `E4M3` | FP8 E4M3FN |
| `E2M1` | 压缩 FP4（每字节存 2 个逻辑 K 值） |

### `tla.cube.mmad(a, b, acc, transpose_a=False, transpose_b=False, clear_accum=False, unit_flag_ctrl=None, *, unit_flag_group_id=None, real_m=None, real_k=None, real_n=None)`

L1×L1→L0C 矩阵乘（`hivm.hir.mmadL1`）。

- `a`/`b`：L1 buffer（2D ND 或 4D NZ）；`acc`：L0C buffer
- `clear_accum=True`：初始化累加器
- `real_m/k/n`：编译期正整数（16 的倍数），None=全尺寸
- `unit_flag_ctrl`：配合 Fixpipe 实现 split-K atomic
- 须在 `tla.scope.cube()` 内调用

### `tla.cube.mmad_mx(a, b, scale_a, scale_b, acc, *, format_a=None, format_b=None, transpose_a=False, transpose_b=False, clear_accum=False, bias=None, real_m=None, real_k=None, real_n=None)`

MX microscaling matmul（`hivm.hir.mmadmxL1`），**950 专用**。

- `a`/`b`：L1 buffer，dtype 为 `float8e5/float8e4nv` 或 FP4 打包 int8
- `scale_a`/`scale_b`：L1 int8/uint8 E8M0 scale buffer，shape `(M, K//32)`
- `format_a/b`：`MxFormat`；FP8 自动推断，FP4 必须显式指定
- `bias`：可选 L1 per-channel 偏置 `(N,)` 或 `(1,N)`
- `real_k`：存储字节数（FP4 时逻辑 K = real_k × 2）
- scale buffer 须先用 `load_gm_to_l1_mx_scale` 加载
- 须在 `tla.scope.cube()` 内调用

---

## 5. 同步原语

### 枚举

```python
tla.sync.MODE.INTER      # 跨 block 逻辑同步
tla.sync.MODE.SUBBLOCK   # sub-block 间逻辑同步
tla.sync.MODE.BLOCK      # block 内逻辑同步
tla.sync.MODE.INTRA      # A5 单个物理计数器（不可与 BLOCK 互换）

tla.sync.PIPE.S / V / M / MTE1 / MTE2 / MTE3 / F / ALL
# F = PIPE_FIX；ALL 仅用于 barrier
```

### 函数

```python
tla.sync.barrier(pipe)                        # 等待 pipeline 全部前驱完成
tla.sync.set_flag(pipe1, pipe2, event_id)     # pipe1→pipe2 发送完成信号（event_id: 0-7）
tla.sync.wait_flag(pipe1, pipe2, event_id)   # 等待 pipe1→pipe2 信号
tla.sync.sync_all(mode: bool)                 # 全局 barrier；False=AIV, True=AIC+AIV
tla.sync.block_set_flag(mode, pipe, flagid)   # 跨核/intra 信号（flagid: 逻辑0-15, INTRA 0-31）
tla.sync.block_wait_flag(mode, pipe, flagid)   # 等待跨核/intra 信号
tla.sync.mutex_lock(bufid, pipe, mode=False)   # 获取 A5 buffer token（bufid 0-31）
tla.sync.mutex_unlock(bufid, pipe, mode=False)  # 释放 A5 buffer token
tla.sync.sub_block_id()                        # 当前 AIV sub-block 索引（= tla.sub_block_id()）
```

`MODE.INTRA` 操作单个 A5 物理计数器，不自动访问 `flagid+16` partner counter。

---

## 6. SIMD 向量寄存器操作

所有函数接受 `tla.vector` 参数，可选 `mask: tla.vector<int1>` 和 `mode: MODE`。

### 枚举

```python
tla.simd.MODE.ZEROING   # 非活跃 lane 结果为 0（默认）
tla.simd.MODE.MERGING   # 非活跃 lane 保留 acc/src0 值
tla.simd.CMP.GT/GE/LT/LE/EQ/NE   # 比较类型
tla.simd.POS.LOWEST / HIGHEST    # vdupv lane 选择
```

### load / store

```python
simd.vlds(src, dist, lanes, elem, mask=None, mode=ZEROING)
# 从 buffer_access 加载；dist=LoadDist 字符串（NORM/UNPK_B16/UNPK4_B8/BRC_B32/…）

simd.vsts(dst, val, dist, mask=None)
# 向量写入 buffer_access；dist=StoreDist（NORM_B32/ONEPT_B32/…）

simd.pld(src, pred_density, lanes, elem, mask=None)  # 带谓词密度 load
simd.pst(dst, val, pred_density, mask=None)          # 带谓词密度 store
simd.vgather(base, offsets, lanes, elem, mask=None)  # 32-bit 散读
simd.vld(base, offsets, lanes, elem, mask=None)      # 16-bit 散读
simd.vsstb(dst, val, mask=None)                      # 向量字节存
```

### 类型转换

```python
simd.vextf(src, out_dtype, part=None, mask=None, mode=ZEROING)
# 低精度→高精度展开；part='part_even'/'part_odd'；fp8→fp32、bf16→fp32 关键路径

simd.vunpack(src, elem, signed=True, part=None, mask=None, mode=ZEROING)
# 整数解包；FP4 压缩数据展开

simd.vbitcast(src, to_dtype)              # bit-pattern 重解释
simd.vtruncf(src, out_dtype, mask=None)   # 高→低精度截断（fp32→fp16/bf16）
simd.vpack(a, b, dist, mask=None)         # 向量打包
simd.vcvt(src, out_dtype, mask=None)      # 通用转换；不支持 MODE.MERGING
```

### 交织

```python
simd.vintlv(a, b, dist, mask=None, mode=ZEROING)   # 向量交织
simd.vdintlv(src, pos, mask=None, mode=ZEROING)    # 向量去交织
```

### 算术

```python
simd.vadd(lhs, rhs, mask=None, mode=ZEROING)
simd.vsub(lhs, rhs, mask=None, mode=ZEROING)
simd.vmul(lhs, rhs, mask=None, mode=ZEROING)
simd.vdiv(lhs, rhs, precision=None, mask=None, mode=ZEROING)  # precision: None/'exact'/'ftz_true'
simd.vmax(lhs, rhs, mask=None, mode=ZEROING)
simd.vmin(lhs, rhs, mask=None, mode=ZEROING)
simd.vxor(lhs, rhs, mask=None, mode=ZEROING)
simd.vor(lhs, rhs, mask=None, mode=ZEROING)
simd.vand(src0, src1, mask=None, mode=ZEROING)   # MERGING 保留 src0
simd.vmod(lhs, rhs, mask=None, mode=ZEROING)
simd.vfma(acc, lhs, rhs, mask=None, mode=ZEROING)  # acc + lhs*rhs
```

### 向量-标量

```python
simd.vmuls(src, scalar, mask=None, mode=ZEROING)   # 向量×标量；MERGING 保留 src
simd.vadds(src, scalar, mask=None, mode=ZEROING)   # 向量+标量
simd.vsubc(src, scalar, mask=None, mode=ZEROING)   # 向量-标量
simd.vaddc(src, scalar, mask=None, mode=ZEROING)   # 标量-向量
simd.vmaxs(src, scalar, mask=None, mode=ZEROING)   # max(向量, 标量)
simd.vmins(src, scalar, mask=None, mode=ZEROING)   # min(向量, 标量)
```

### 累加与 AXPY

```python
simd.vmula(acc, src0, src1, mask=None, mode=ZEROING)   # acc + src0*src1；MERGING 保留 acc
simd.vmadd(acc, src0, src1, mask=None, mode=ZEROING)   # vmula 别名
simd.vaxpy(acc, src, scalar, mask=None, mode=ZEROING)   # acc + scalar*src
```

### 移位

```python
simd.vshl(src, shift, mask=None, mode=ZEROING)    # 左移
simd.vshr(src, shift, mask=None, mode=ZEROING)    # 右移
simd.vshls(src, shift, mask=None, mode=ZEROING)   # 饱和左移
```

### 一元

```python
simd.vabs(src, mask=None, mode=ZEROING)
simd.vneg(src, mask=None, mode=ZEROING)
simd.vnot(src, mask=None, mode=ZEROING)
simd.vrelu(src, mask=None, mode=ZEROING)    # f16/f32/i32
simd.vsqrt(src, mask=None, mode=ZEROING)   # f16/f32
simd.vrsqrt(src, mask=None, mode=ZEROING)
simd.vexp(src, precision=None, mask=None, mode=ZEROING)  # f16/f32
simd.vlrelu(src, slope, mask=None, mode=ZEROING)
simd.vprelu(src, slope_vec, mask=None, mode=ZEROING)
simd.vabsdif(lhs, rhs, mask=None, mode=ZEROING)   # |lhs-rhs|
simd.vexpdif(lhs, rhs, mask=None, mode=ZEROING)   # exp(lhs-rhs) 融合（online softmax 关键）
```

### 广播

```python
simd.vdup(src, scalar, lanes, dtype, mask=None, mode=ZEROING)  # 标量→向量
simd.vdups(scalar, lanes, dtype, mask=None, mode=ZEROING)      # 带谓词标量广播
simd.vdupv(src, pos, mask=None, mode=ZEROING)                   # lane 内广播
simd.vbroadcast(src, out_lanes, mask=None, mode=ZEROING)        # 向量广播扩展
simd.vfull(scalar, lanes, dtype)                               # 全标量值向量
simd.vsplat(scalar, dtype, mask=None, mode=ZEROING)             # vdup 包装
```

### 比较与选择

```python
simd.vcmp(cmp_type, lhs, rhs, mask=None)   # 返回 vector<i1>
simd.vsel(mask, x, y)                       # predicate 选择
simd.vcmax(src, mask=None, mode=ZEROING)   # lane-wise cumulative max
simd.vcmin(src, mask=None, mode=ZEROING)   # lane-wise cumulative min
```

### 归约

```python
simd.vreduce(src, op, mask=None)        # op='sum'/'max'/'min'/'and'/'or'
simd.vreduce_sum(src, mask=None)
simd.vreduce_max(src, mask=None)
simd.vreduce_min(src, mask=None)
```

### 整数计数 / 谓词

```python
simd.vci(src, mask=None)      # 整数通道计数
simd.pge(pattern)             # 生成 i1 predicate；pattern='ALL'/'VL1'/'VL2'/'H'/'M3'/'M4'…
simd.membar()                  # SIMD 内存屏障
```

---

## 7. Scope 核心区域标注

```python
with tla.scope.vector():       # VECTOR core，非 VF（不 outline）
    tla.copy(gm_buf, ub_buf)

with tla.scope.simd():         # VECTOR core，SIMD VF（outline-scope 提取为 func.func）
    vec = tla.simd.vlds(ub_buf[0], dist="NORM_B16", lanes=128, elem="bf16")
    tla.simd.vsts(ub_buf[1], tla.simd.vmul(vec, scale_vec), dist="NORM_B32")

with tla.scope.cube():         # CUBE core（AIC）
    tla.cube.mmad(a_l1, b_l1, acc_l0c, clear_accum=True)
    tla.store_l0c_to_gm(acc_l0c, gm_out, params=...)
```

- `simd` 携带 `no_inline/outline/vector_mode="vec"/hivm.vf_mode=SIMD` 属性
- `cube` 携带 `hivm.tcore_type=CUBE`
- 嵌套：`simd` 可嵌于 `vector` 内；`cube` 独立

---

## 8. 标量工具

```python
tla.select(condition, true_value, false_value)    # 标量条件选择（不支持 bf16）
tla.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)  # 标量 max（不支持 fp16/bf16）
tla.minimum(a, b, propagate_nan=tl.PropagateNan.NONE)   # 标量 min
tla.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)  # 标量转换（不支持 bf16）
tla.max_value(dtype)      # dtype 最大有限值标量常量
tla.infinity(dtype)       # dtype 正无穷标量（fp16/bf16/fp32）
tla.clamp(x, min, max, propagate_nan=tl.PropagateNan.NONE)  # 标量 clamp
tla.cdiv(x, div)          # 标量向上整除
```

---

## 9. 数学辅助

```python
tla.compensated_rstd(square_sum, hidden, eps)
# 补偿倒数标准差（RMSNorm Newton 迭代，TwoSum 误差回收）
# square_sum: fp32 vector（全 lane 同值）；hidden: 编译期正整数；eps: float >= 2^-126

tla.truncmod(a, b)   # 截断整除余数（truncate-toward-zero）
tla.sqrt(x)          # fp32 scalar 或 fp16/fp32 vector 平方根
```

---

## 10. 编译器提示

```python
tla.annotate_buffer_versions(buf, versions)
# 声明 buffer 多缓冲版本数（annotation.mark hivm.multi_buffer=N）
# versions>=1；1=不多缓冲；通常在 alloc 后调用

tla.assume(cond)
# 编译期优化提示（llvm.intr.assume）；cond 为 bool 或 scalar int1
# 非运行时检查，条件为假时 UB
```

---

## 11. 调试

```python
tla.device_print(prefix, *args, hex=False)
# 设备端打印（hivm.hir.debug）
# args 可为 UB/GM buffer、buffer_access（buf[i,j]）或标量
# 仅在 tla.scope 外调用；L1/L0* 不可直接打印

tla.device_assert(cond, msg="")
# 设备端断言（scalar i1 条件）
# 仅在 options.debug=True 时生效，否则 no-op
```

---

## 12. 选择操作

```python
tla.where(condition, x, y)
# 两路 eager 选择：
#   condition=bool/int1 scalar + x,y 标量 → 标量路径（代理 tl.where）
#   condition=vector<i1> + x,y=vector → SIMD 路径（代理 simd.vsel）
#   两者不可混用（标量与 vector arm 不可混搭）
```

---

## 13. DMA 参数 dataclass 完整列表

所有 dataclass 为 `frozen=True`，字段默认值如下：

```python
LoadGM2UBParams(pad_mode=None, pad_value=None, left_padding_num=None,
               right_padding_num=None, eviction_policy=None)

StoreUB2GMParams(atomic_kind=None)

LoadGM2L1Params(l2_cache_ctrl=None, enable_nz2nd=False, enable_nz2dn=False)

CopyUB2L1Params(pad_mode=None, pad_value=None, collapse_reassociation=None)

CopyL12UBParams()                    # 无字段
CopyL12BTParams(n=None)              # native L12BT n 操作数
CopyL12FBParams()                    # 保留，NotImplementedError

CopyL1ToL0Params(is_transpose=False, actual_batch=None, k_part_idx=None,
                k_part=None, k_part_ceil=None, k_part_loop=None,
                k_part_actual=None, m=None)

FixpipeL0C2GMParams(dma_mode=NZ2NZ, pre_quant=NO_QUANT, pre_relu=NO_RELU,
                   quant_scale=None, unit_flag_ctrl=None, unit_flag_group_id=None)

FixpipeL0C2L1Params(dma_mode=NZ2NZ, pre_quant=NO_QUANT, pre_relu=NO_RELU,
                   channel_split=False, c0_pad_en=True,
                   unit_flag_ctrl=None, unit_flag_group_id=None, quant_scale=None)

FixpipeL0C2UBParams(dma_mode=NZ2NZ, dual_dst_mode=NO_DUAL,
                   sub_block_idx=SUB_BLOCK_0, pre_quant=NO_QUANT,
                   pre_relu=NO_RELU, unit_flag_ctrl=None,
                   unit_flag_group_id=None, quant_scale=None)
```

---

## 14. MXfp8 matmul 流水示例

```python
@tla.jit
def mxfp8_matmul(ptr_a, ptr_b, ptr_sa, ptr_sb, ptr_out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # 绑定 GM 指针
    gm_a  = tla.view(ptr_a,  shape=(M, K))
    gm_b  = tla.view(ptr_b,  shape=(N, K))
    gm_sa = tla.view(ptr_sa, shape=(M, K // 32))
    gm_sb = tla.view(ptr_sb, shape=(N, K // 32))
    gm_out = tla.view(ptr_out, shape=(M, N))

    with tla.scope.cube():
        # 分配 L1 / L0C buffer
        a_l1  = tla.alloc((M, K),       tl.float8e4nv, scope=tla.AddressSpace.L1)
        b_l1  = tla.alloc((N, K),       tl.float8e4nv, scope=tla.AddressSpace.L1)
        sa_l1 = tla.alloc((M, K // 32), tl.uint8,      scope=tla.AddressSpace.L1)
        sb_l1 = tla.alloc((N, K // 32), tl.uint8,      scope=tla.AddressSpace.L1)
        acc   = tla.alloc((M, N),       tl.float32,    scope=tla.AddressSpace.L0C)

        # 加载操作数和 scale
        tla.load_gm_to_l1(gm_a, a_l1)
        tla.load_gm_to_l1(gm_b, b_l1)
        tla.load_gm_to_l1_mx_scale(gm_sa, sa_l1)
        tla.load_gm_to_l1_mx_scale(gm_sb, sb_l1)

        # MXfp8 matmul
        tla.cube.mmad_mx(a_l1, b_l1, sa_l1, sb_l1, acc,
                         format_a=tla.MxFormat.E4M3,
                         format_b=tla.MxFormat.E4M3,
                         clear_accum=True)

        # Fixpipe 写回 GM（NZ2ND 布局转换）
        tla.store_l0c_to_gm(acc, gm_out,
                            params=tla.FixpipeL0C2GMParams(
                                dma_mode=tla.FixpipeDMAMode.NZ2ND))
```

### E4M3 FP8 decode 链（VF 单区域）

```python
with tla.scope.simd():
    # vlds 携带 dist="UNPK4_B8" 读 fp8，解包到 2048-bit 寄存器
    fp8_vec = tla.simd.vlds(ub_fp8[0], dist="UNPK4_B8", lanes=128, elem="fp8e4nv")
    # vextf 展开到 fp32（part_even = 低半段 64 lanes）
    f32_vec = tla.simd.vextf(fp8_vec, tla.float32, part="part_even")
    # 乘以 E8M0 scale（已 broadcast）
    out_vec = tla.simd.vmul(f32_vec, scale_vec)
    tla.simd.vsts(ub_out[0], out_vec, dist="NORM_B32")
```

### split-K atomic fixpipe（无额外 reduce barrier）

```python
with tla.scope.cube():
    tla.cube.mmad_mx(a_tile, b_tile, sa, sb, acc_l0c, clear_accum=True)
    # unit_flag_ctrl=3 → fixpipe atomic ADD 写 fp32 GM，所有 task 自动汇聚
    tla.store_l0c_to_gm(acc_l0c, gm_fp32,
                        params=tla.FixpipeL0C2GMParams(
                            dma_mode=tla.FixpipeDMAMode.NZ2ND,
                            unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE,
                            unit_flag_group_id=0,
                        ))
```

### online softmax（单遍 exp(qk - max)）

```python
with tla.scope.simd():
    # vexpdif(score, m) = exp(score - m) —— 融合指令，避免分开 vsub+vexp 的精度损失
    exp_vec = tla.simd.vexpdif(score_vec, m_broadcast)
    l_vec   = tla.simd.vadd(l_vec, tla.simd.vreduce_sum(exp_vec))
```
