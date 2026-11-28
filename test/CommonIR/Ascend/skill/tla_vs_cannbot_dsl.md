# TLA vs cannbot-dsl 差异对照

> 源码来源：
> - TLA: `triton-ascend` GitHub main 分支 `third_party/ascend/language/tla/`
> - cannbot-dsl: 已安装包 `/usr/local/.../site-packages/cannbotdsl/` (v0.6.0)

---

## 1. 架构本质差异

| 维度 | TLA | cannbot-dsl |
|------|-----|-------------|
| 定位 | triton-ascend 的 Ascend NPU 低层编程扩展，作为独立 `tla.*` 命名空间嵌入 Triton 内核 | 独立 MLIR-based Python DSL，有自己的 CANNIR/asccube/ascvec/ascscalar 方言体系 |
| IR 目标 | 通过 `tla_builder.cc` 生成 HIVM/AVE dialect IR，最终走 triton 编译路径 | 直接生成 CANNIR MLIR，通过 `cannbotdsl.core.compiler` 独立编译 |
| 编程粒度 | buffer（物理地址空间视图）+ vector（SIMD 寄存器），精确到 UB/L1/L0A/L0B/L0C | Tensor（逻辑张量）+ MemLoc 枚举，无需手动管理 L0A/L0B 分配 |
| 内核入口 | `@tla.jit`（包装 Triton JIT，可与 `@triton.jit` 混用） | `@cannbotdsl.compile` / `@cannbotdsl.aot` |
| 使用场景 | triton-ascend 内核内部，补充 Triton 无法表达的 Ascend 特有路径 | 独立算子开发，不依赖 Triton 框架 |

**关键认知**：cannbot-dsl 不是 TLA 的封装，也不是 `cann/extension` 的封装，它们是并列的两套独立体系。mega_mla_kernel 用的是 cannbot-dsl，FlagTree 算子用的是 `cann/extension`（TLA 的前身接口），而 TLA 是 triton-ascend GitHub main 上的新一代替换层。

---

## 2. 内存模型对照

### 地址空间

| 物理空间 | TLA | cannbot-dsl |
|---------|-----|-------------|
| 全局内存 | `tla.AddressSpace.GM` / `"gm"` | `MemLoc.GM` |
| 统一缓冲区 | `tla.AddressSpace.UB` / `"ub"` | `MemLoc.UB` |
| L1 缓冲区 | `tla.AddressSpace.L1` / `"l1"` | `MemLoc.L1` |
| L0A/L0B | `tla.AddressSpace.L0A/L0B` / `"l0a"/"l0b"` | `MemLoc.L0A/L0B` |
| L0C 累加器 | `tla.AddressSpace.L0C` / `"l0c"` | `MemLoc.L0C` |
| BT/FB（偏置/fixpipe） | `tla.AddressSpace.BT/FB` / `"bt"/"fb"` | 无对应 |
| SSBUF | `tla.AddressSpace.SSBUF` / `"ssbuf"` | 无对应 |
| L0A_SF/L0B_SF（scale） | 保留，当前 `NotImplementedError` | 无对应 |

### Buffer 创建

```python
# TLA：显式分配 + 视图绑定（两步走）
ub_buf = tla.alloc((128, 64), tl.float16, space="ub")          # 片上分配
gm_buf = tla.view(ptr, shape=(1024,), strides=(1,), space="gm")  # GM 指针绑定

# cannbot-dsl：Tensor 构造（逻辑张量，自动管理物理布局）
ub_tensor = Tensor((128, 64), dtypes.float16, MemLoc.UB)
gm_tensor = Tensor((1024,), dtypes.float16, MemLoc.GM)
```

---

## 3. 数据搬运对照

### DMA 路由

| 路由 | TLA 显式函数 | TLA 统一接口 | cannbot-dsl |
|------|------------|------------|-------------|
| GM→UB | `tla.load_gm_to_ub(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| UB→GM | `tla.store_ub_to_gm(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| GM→L1 (ND2NZ) | `tla.load_gm_to_l1(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| GM→L1 (MX scale) | `tla.load_gm_to_l1_mx_scale(src, dst)` | 无统一接口 | `mem_copy(dst, src, mx_scale=True)` (?) |
| UB→L1 | `tla.copy_ub_to_l1(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| L1→UB | `tla.copy_l1_to_ub(src, dst)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| L1→L0A/L0B | `tla.copy_l1_to_l0(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| L0C→GM (fixpipe) | `tla.store_l0c_to_gm(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src, unit_flag=3)` |
| L0C→UB (fixpipe) | `tla.copy_l0c_to_ub(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |
| L0C→L1 (fixpipe) | `tla.copy_l0c_to_l1(src, dst, params)` | `tla.copy(src, dst)` | `mem_copy(dst, src)` |

### DMA 参数配置差异

TLA 使用 **不可变 dataclass** 配置对象（类型安全，编译期检查）：

```python
# TLA：fixpipe 量化配置
params = tla.FixpipeL0C2GMParams(
    dma_mode=tla.FixpipeDMAMode.NZ2ND,
    pre_quant=tla.FixpipePreQuantMode.F322BF16,
    unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE,
    unit_flag_group_id=0,
)
tla.store_l0c_to_gm(l0c_buf, gm_buf, params=params)
```

cannbot-dsl 使用 **kwargs** 风格（更灵活但无类型提示）：

```python
# cannbot-dsl：等价 fixpipe（原子累加写 fp32 GM）
mem_copy(dst_gm, src_l0c, atomic_add=True, unit_flag=3,
         l2_cache_ctl=L2_CACHE_DISABLE)
```

---

## 4. Cube 计算对照

### 普通 matmul

```python
# TLA：操作 L1/L0C buffer，明确 transpose，支持 real_m/k/n 子块
with tla.scope.cube():
    tla.cube.mmad(a_l1, b_l1, acc_l0c,
                  transpose_a=False, transpose_b=False,
                  clear_accum=True,
                  unit_flag_ctrl=tla.UnitFlagMode.ENABLED_WITH_UPDATE,
                  unit_flag_group_id=0)

# cannbot-dsl：操作 Tensor 对象，隐式管理 L1/L0C
cube.matmul(C, A, B)
```

### MX microscaling matmul（950 专有）

```python
# TLA：显式传入 scale_a/scale_b L1 buffer，指定 MxFormat
with tla.scope.cube():
    tla.cube.mmad_mx(
        a_l1, b_l1, scale_a_l1, scale_b_l1, acc_l0c,
        format_a=tla.MxFormat.E4M3,  # 或 E5M2 / E2M1 (FP4)
        format_b=tla.MxFormat.E4M3,
        clear_accum=True,
    )

# cannbot-dsl：mx_mode=True 参数隐含
cube.matmul(C, A, B, mx_mode=True)
```

---

## 5. 同步原语对照

| 操作 | TLA | cannbot-dsl |
|------|-----|-------------|
| 管线内屏障 | `tla.sync.barrier(PIPE.MTE2)` | `vec_sync_all()` / `vec_sync_pipe(PIPE.MTE2)` |
| 核内事件 set | `tla.sync.set_flag(PIPE.MTE2, PIPE.V, event_id=0)` | `vec_sync_notify(src_pipe, dst_pipe, event_id)` |
| 核内事件 wait | `tla.sync.wait_flag(PIPE.MTE2, PIPE.V, event_id=0)` | `vec_sync_wait(src_pipe, dst_pipe, event_id)` |
| 跨核 set | `tla.sync.block_set_flag(MODE.BLOCK, PIPE.MTE2, flagid=0)` | `vec_sync_block_arrive(pipe, flag_id)` |
| 跨核 wait | `tla.sync.block_wait_flag(MODE.BLOCK, PIPE.MTE2, flagid=0)` | `vec_sync_block_wait(pipe, flag_id)` |
| AIC+AIV 全局同步 | `tla.sync.sync_all(True)` | `global_sync_all()` |
| AIV 内部全局同步 | `tla.sync.sync_all(False)` | 无直接对应（用 barrier） |
| buffer token 加锁 | `tla.sync.mutex_lock(bufid, PIPE.V)` | `vec_get_buf(pipe, buf_id)` |
| buffer token 解锁 | `tla.sync.mutex_unlock(bufid, PIPE.V)` | `vec_release_buf(pipe, buf_id)` |
| 获取 AIV sub-block ID | `tla.sync.sub_block_id()` | `get_subblock_id()` |

TLA 新增 `MODE` 枚举区分同步语义（`INTER/SUBBLOCK/BLOCK/INTRA`），比 cannbot-dsl 的 `mode=2` 整数更类型安全。`MODE.INTRA` 是 TLA 专有——直接操作 A5 单个物理计数器，cannbot-dsl 无对应。

---

## 6. SIMD 向量寄存器对照

TLA 有完整的 `tla.simd.*` 层，直接操作 AVE dialect 向量寄存器（`tla.vector` 类型），cannbot-dsl 在 `vf(mode="simd")` context 内通过 Tensor 操作隐式生成 SIMD 指令。

| 操作 | TLA simd | cannbot-dsl 等价（近似） |
|------|---------|----------------------|
| 从 buffer 加载向量 | `simd.vlds(buf[i], lanes=256, dist="NORM_B16")` | `Tensor` 访问自动映射 |
| 向量存到 buffer | `simd.vsts(buf[i], vec, dist="NORM_B32")` | `Tensor` 写 |
| 向量 gather（散读） | `simd.vgather(buf, index_vec)` | `reg.vgather(tensor, index_vec)` |
| 向量 scatter（散写） | `simd.vscatter`（通过 pld/pst） | `reg.vscatter(tensor, src, index)` |
| 向量乘法 | `simd.vmul(a, b, mask=m)` | 隐式 Tensor `*` |
| 向量加法 | `simd.vadd(a, b, mask=m)` | 隐式 Tensor `+` |
| 向量 max | `simd.vmax(a, b, mask=m)` | 隐式 |
| 向量 exp | `simd.vexp(a, mask=m)` | 隐式 |
| FP/int 类型转换 | `simd.vextf(src, dst_ty, part='part_even')` | `reg.cast(src, dtype)` |
| 格式解包（fp8/bf16） | `simd.vunpack(src, sign)` | 无直接对应 |
| 向量广播标量 | `simd.vdup(scalar, lanes, dtype)` | `Tensor` broadcast |
| 比较生成 predicate | `simd.vcmp(a, b, cmp=CMP.GT)` | `reg.compare.*` |
| predicate 条件选择 | `simd.vsel(mask, a, b)` | `tl.where` / `reg.where` |
| 向量 reduce（sum/max/min） | `simd.vreduce(vec, op='sum')` | `reg.reduce.*` |
| 合并模式 | `MODE.ZEROING` / `MODE.MERGING` | 无对应（隐式 zeroing） |

**TLA SIMD 相对 cannbot-dsl 的独特能力**：
- `vlds/vsts` 支持 `dist=` 参数控制 UNPK_B16/UNPK4_B8/NORM_B32 等打包格式，直接用于 fp8/bf16 decode 链
- `vunpack(src, sign)` 解包 4bit 压缩数据
- `vextf` fp8→fp32 展开（`ave.hir.vextf`），是 E4M3 decode 关键路径
- `vintlv/vdintlv` 交织/解交织
- `vexpdif` 融合 exp(a-b)（online softmax 关键指令）
- `predicate mode (MODE.MERGING)` 保留 accumulator 非活跃 lane

---

## 7. VF Scope（向量函数区域）对照

```python
# TLA：显式 simd scope（由 outline-scope pass 提取为 vector function）
with tla.scope.simd():
    vec = simd.vlds(ub_buf[0], ...)
    result = simd.vmul(vec, scale)
    simd.vsts(ub_buf[1], result, ...)

# TLA：普通 vector scope（不提取为 VF，只标记 core_type=VECTOR）
with tla.scope.vector():
    tla.copy(gm_buf, ub_buf)

# cannbot-dsl：vf context manager（等价 tla.scope.simd）
with vf(mode="simd"):
    # ... 内部操作自动走 VF 路径

# cannbot-dsl：AIC/AIV 分支
if is_aic():
    cube.matmul(...)
else:
    # AIV 路径
```

TLA 的 `scope.cube()` 和 `scope.vector()/simd()` 分别对应 `is_aic()` 分支内和 `else` 分支内的 cannbot-dsl 代码块。

---

## 8. TLA 独有能力（cannbot-dsl 无对应）

1. **`tla.view` / `tla.subview` / `tla.reshape`**：精确的 buffer 切片，memref.subview 语义，支持动态 offset，非对齐访问路径可在此表达

2. **`tla.load_gm_to_l1_mx_scale`**：MX scale DMA 专用路由，非标准 ND2NZ（FP8/FP4 matmul 专属）

3. **`tla.set_ctrl(enable, idx)`**：直接操作硬件 CTRL bit（如关闭 L2 预取 bit 48）

4. **`tla.set_atomic(kind, dtype)`**：持久设置 atomic 写模式，作用于后续 UB→GM store

5. **`tla.sync.MODE.INTRA`**：操作单个 A5 物理同步计数器，不经过逻辑 block 层

6. **`tla.cube.mmad` `real_m/k/n` 子块参数**：精确控制 MMAD 实际计算范围，不必填满 L1 tile

7. **`tla.FixpipePreQuantMode.QF322F32_PRE`**：量化 fp32 预处理模式（cannbot-dsl mem_copy 参数无法表达）

8. **`tla.simd.vunpack`**：4-bit 压缩数据解包（FP4 KV cache decode 关键路径）

9. **`tla.simd.vexpdif`**：融合 exp(lhs-rhs)，online softmax numerically stable 路径

10. **`tla.alloc` `versions=N`**：编译期多缓冲版本标注（配合后端自动 ring buffer 推导）

---

## 9. cannbot-dsl 独有能力（TLA 无对应）

1. **`RingBuffer`**：显式 depth=N 双/多缓冲抽象，自动管理 DMA/compute 流水通知
2. **`make_layout` / `RegLayout`**：张量布局描述，NZ/NZ4 等物理布局绑定
3. **`is_aic()` / `get_subblock_id()`**：运行时核类型查询（TLA 通过 scope 在编译期区分）
4. **`DataStruct` / `CArray` / `StructSchema`**：结构化数据类型，支持自定义 544B window cache record 等复合类型
5. **`onchip_rewind` / `channel_rewind`**：同步 flag id 复用机制（防止 id 耗尽）
6. **`PIPE.MTE1/MTE2/MTE3` 分离 DMA notify/wait**：`vec_sync_notify(MTE2, MTE3, 0)` 控制双 buffer 交替
7. **`global_sync_all(flag_ids=[10,11,12,13,14])`**：自定义同步 flag id 列表
8. **`ProfileSpec / get_platform_info`**：运行时平台信息查询（910B vs 950 条件编译支持）

---

## 10. FlagTree 迁移实践建议

FlagTree 中的算子当前使用 `cann/extension`（旧接口），新实现可选择：

**方案 A：保留 `cann/extension` 旧接口**（兼容 910B，改动最小）
- 适用：M3A/M3D 量化 matmul（无 SIMD 寄存器操作需求）
- 仍需修 M5（补 `_gather_load/_scatter_store/_index_put`）

**方案 B：迁移到 TLA**（950 最优化路径）
- `cann/extension.copy(src, dst, ...)` → `tla.copy(src, dst, params=...)`
- `cann/extension.fixpipe(...)` → `tla.store_l0c_to_gm(params=FixpipeL0C2GMParams(...))`
- `cann/extension.sync_block_arrive/wait` → `tla.sync.block_set_flag/block_wait_flag`
- `mem_ops.scatter_ub_to_out(旧)` → `tla.simd.vsts(dist="NORM_B32", ...)` + `tla.sync.set_flag`
- `mem_ops._scatter_store(新 M5)` → `tla.view + subview + store_ub_to_gm(params=...)`
- 950 MXfp8 matmul → `tla.cube.mmad_mx(format_a=MxFormat.E4M3)` + `load_gm_to_l1_mx_scale`
- E4M3 decode → `tla.simd.vlds(dist="UNPK4_B8") → vextf(f32) → vmul(scale)`

**方案 C：参考 cannbot-dsl 逻辑，在 TLA 中重现**
- mega_mla 的 VF 单区域量化链（P2 AIV1）→ `tla.scope.simd()` 中串联 norm→rope→quant SIMD ops
- online softmax → `simd.vexpdif` + `simd.vmul` + `simd.vreduce`
- split-K atomic fixpipe → `tla.cube.mmad_mx` + `FixpipeL0C2GMParams(atomic_kind=AtomicKind.ADD)`
