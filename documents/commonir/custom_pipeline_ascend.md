# Custom Pipeline Ascend 后端支持

## 1. 环境依赖
```bash
# 由于特性尚未完善，目前支持kernel launch时指定custom_pipeline参数来使能此功能，因此依赖临时包
download https://jwolpxeehx.feishu.cn/file/FVypb7hZJoFYqSxEeI3cCleNnmc
bash ascendnpu-ir-aarch64-pr_3023-master.run
```
> [!NOTE]
> custom_pipeline功能相对独立，在未使用custom_pipeline参数时，不用安装该依赖包

## 2. 使用方式

在 host 侧的 kernel launch 中添加：

```python
kernel[grid](
    input,
    output,
    custom_pipeline="cv-pipelining,hivm-enable-multi-buffer",
)
```

FlagTree backend 会生成：

```text
--custom-compilation-pipeline=cv-pipelining,hivm-enable-multi-buffer
```

## 3. 三种参数状态

| kernel 参数 | BishengIR 命令行行为 | 含义 |
| --- | --- | --- |
| 不传 `custom_pipeline` | 不追加参数 | 使用默认行为 |
| `custom_pipeline=""` | `--custom-compilation-pipeline=` | 启用自定义 pipeline，但 pass 列表为空 |
| `custom_pipeline="pass-a,pass-b"` | `--custom-compilation-pipeline=pass-a,pass-b` | 使用指定 pass 列表 |


## 4. 支持的 Pass

- `hfusion-reorder-ops`
- `auto-blockify-parallel-loop`
- `hivm-mark-multi-buffer`
- `cv-pipelining`
- `hivm-enable-multi-buffer`
- `hivm-bind-sub-block`
- `hivm-partition-and-bind-sub-block`
- `loop-invariant-code-motion`
- `loop-invariant-subset-hoisting`
- `hivm-mark-stride-align`
- `hivm-clone-tensor-empty`
- `hivm-sink-op-to-consumer-in-loop`
- `hivm-cross-core-gss`
- `hivm-inject-block-sync`
- `hivm-insert-load-store-for-mix-cv`
- `hivm-insert-load-store-for-scalar`
- `insert-workspace-for-mix-cv`
- `hivm-bind-workspace-arg`
- `hivm-auto-infer-buffer-size`
- `convert-arith-to-affine`
- `hivm-constantize-buffer-size`
- `hivm-set-buffer-size`
- `hivm-plan-memory`
- `hivm-insert-infer-workspace-size-func`

## 5. Pass 出现次数限制

可以在 pass 名称后使用 `#<number>` 限制出现次数。例如：

```text
cv-pipelining#4#5
```

表示选择排除第 4 次和第 5 次出现的 `cv-pipelining`；不添加编号表示开放该 pass 的所有出现：

```text
cv-pipelining
```

多个 pass 使用逗号分隔：

```python
custom_pipeline="hivm-enable-multi-buffer,cv-pipelining#4#5,hivm-plan-memory"
```