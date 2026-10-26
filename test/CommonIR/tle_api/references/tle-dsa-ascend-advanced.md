# TLE DSA Ascend Advanced

> **Layer**: Ascend hardware-specific — compile hints, sub-vector ID, and inter-core synchronization.
> **Source**: `tle.dsa.ascend.compile_hint.md`, `tle.dsa.ascend.sub_vec_id.md`, `tle.dsa.ascend.sync_block_set.md`, `tle.dsa.ascend.sync_block_wait.md`, `tle.dsa.ascend.sync_block_all.md`, `tle_inter_core_pipeline.md`

## 1. `compile_hint` — attach optimization hints to a tensor

```python
tle.dsa.ascend.compile_hint(
    ptr,           # tl.tensor — the tensor to annotate
    hint_name,     # str — hint type (see table below)
    hint_val=None, # None | bool | int | list[int] — optional value
)
```

Attaches compile-time metadata to a tensor. Does not change computation semantics. The same tensor can carry multiple hints. Unlike `tle.dsa.hint` (which is a `with` scope for `copy` calls), `compile_hint` targets a named tensor value.

### Available hints

| `hint_name` | `hint_val` | When to use |
|-------------|-----------|-------------|
| `"disable_bubble_up"` | none | Vector computation + `extract_slice` scatter-write loop: compiler hoists ops into the loop, causing redundant computation per row. This hint keeps the full-tensor Vector op outside the loop. |
| `"mayDiscretememaccess"` | none | K=1 or BLOCK_K=1: the `<BLOCK_M×1×f32>` load triggers 32B alignment expansion to `<BLOCK_M×8×f32>`, ballooning UB by 8×. This hint downgrades to scalar discrete access. |
| `"dot_pad_only_k"` | none | M/N already aligned but K is not: tells the compiler to pad only the K dimension in `tl.dot`, skipping unnecessary M/N padding. |
| `"hivm.tile_mix_cube_num"` | `int` (sub-tile count) | Multiple matmuls in the same kernel (e.g., FlashAttention QKᵀ + PV) with overlapping lifetimes cause L1 overflow. Sub-tiling factor controls how many tiles the compiler sees simultaneously. |
| `"bitwise_mask"` | none | `tl.where` condition is an i1 tensor stored as i8: compiler generates `vcast(i8→f16)→vcmp→vnot→vsel` chain. This hint skips the chain and uses a direct bitwise select. |

### Usage examples

```python
# disable_bubble_up — RMSNorm scatter-write pattern
y = (x * rrms * w).to(out_ptr.dtype.element_ty)
for i in tl.static_range(N):
    value_reload = tle.dsa.extract_slice(y, (i,), (1,), (1,))
    tle.dsa.ascend.compile_hint(value_reload, "disable_bubble_up")
    tl.store(out_ptr + pid * N + i + tl.arange(0, 1), value_reload)

# mayDiscretememaccess — K=1 matmul tile
a_block = tl.load(mat_a + a_offset, mask=a_mask, other=0.0)
b_block = tl.load(mat_b + b_offset, mask=b_mask, other=0.0)
tle.dsa.ascend.compile_hint(a_block, "mayDiscretememaccess")
tle.dsa.ascend.compile_hint(b_block, "mayDiscretememaccess")
acc = tl.dot(a_block, b_block, acc)

# hivm.tile_mix_cube_num — FlashAttention two-matmul kernel
s_acc = tl.dot(q_block, k_block, s_acc)
tle.dsa.ascend.compile_hint(s_acc, "hivm.tile_mix_cube_num", 4)
# ... softmax ...
o_acc = tl.dot(p_block, v_block, o_acc)
tle.dsa.ascend.compile_hint(o_acc, "hivm.tile_mix_cube_num", 4)

# bitwise_mask — tl.where on i1 condition
cond = tl.load(cond_ptr + xindex, xmask)
tle.dsa.ascend.compile_hint(cond, "bitwise_mask")
res = tl.where(cond, in1, in0)
```

> `compile_hint` is for named tensor values. For `tle.dsa.copy` calls that have no return value, use `tle.dsa.hint` (a `with` scope) instead.

## 2. `sub_vec_id` — current vector sub-core ID

```python
vec_id = tle.dsa.ascend.sub_vec_id() -> i16
```

Returns an integer in `[0, N)` identifying which Vector sub-core is executing. On Ascend 910B, each AI Core has 2 Vector sub-cores. The sub-vector ID lets you manually partition data across them.

Only valid inside an AIC/AIV mixed-mode kernel. Calling it in a pure Cube or pure Vector kernel is a compile error. When `sub_vec_id` is used, the generated module gets `{hivm.disable_auto_tile_and_bind_subblock}` to prevent the compiler from auto-partitioning (which would conflict with manual partitioning).

```python
@triton.jit
def kernel(out_ptr, N: tl.constexpr):
    with tle.dsa.ascend.scope(core_mode="vector"):
        sub_id = tle.dsa.ascend.sub_vec_id()
        offs = sub_id * N + tl.arange(0, N)
        tl.store(out_ptr + offs, sub_id.to(tl.int32))
```

## 3. Inter-core synchronization: `sync_block_set` / `sync_block_wait`

On Ascend 910B the Cube core and Vector core run concurrently. To pipeline them (Cube computes while Vector processes the previous result), use paired set/wait events.

```python
tle.dsa.ascend.sync_block_set(
    sender,        # str: "cube" or "vector"
    receiver,      # str: "cube" or "vector"  (must differ from sender)
    event_id,      # int: 0-15
    sender_pipe,   # tle.dsa.ascend.PIPE — which pipeline the sender uses
    receiver_pipe, # tle.dsa.ascend.PIPE — which pipeline the receiver uses
)

tle.dsa.ascend.sync_block_wait(
    sender,        # same as set
    receiver,      # same as set
    event_id,      # same as set
    sender_pipe,   # same as set
    receiver_pipe, # same as set
)
```

Every `sync_block_set` call must be paired with a `sync_block_wait` that has the same `sender`, `receiver`, `event_id`, `sender_pipe`, and `receiver_pipe`.

### PIPE enum

```python
pipe = tle.dsa.ascend.PIPE

pipe.PIPE_S     # Scalar pipeline (GetValue ops)
pipe.PIPE_V     # Vector compute + L0C→UB DMA
pipe.PIPE_M     # Matrix (Cube) compute
pipe.PIPE_MTE1  # L1→L0A, L1→L0B DMA
pipe.PIPE_MTE2  # GM→L1, GM→L0, GM→UB DMA
pipe.PIPE_MTE3  # UB→GM, UB→L1 DMA
pipe.PIPE_ALL   # All pipelines
pipe.PIPE_FIX   # L0C→GM, L0C→L1 DMA (Fixpipe)
```

**Typical PIPE combinations:**
- Cube → Vector notification: `sender_pipe=PIPE_FIX, receiver_pipe=PIPE_MTE2`
- Vector → Cube notification: `sender_pipe=PIPE_MTE2, receiver_pipe=PIPE_FIX`

### Simple example: Cube notifies Vector

```python
@triton.jit
def cv_sync_kernel(...):
    with tle.dsa.ascend.scope(core_mode="cube"):
        # ... cube compute ...
        tle.dsa.ascend.sync_block_set("cube", "vector", 0,
                                       pipe.PIPE_FIX, pipe.PIPE_MTE2)

    with tle.dsa.ascend.scope(core_mode="vector"):
        tle.dsa.ascend.sync_block_wait("cube", "vector", 0,
                                        pipe.PIPE_FIX, pipe.PIPE_MTE2)
        # ... vector compute on Cube's result ...
```

## 4. `sync_block_all` — global barrier

```python
tle.dsa.ascend.sync_block_all(
    mode,      # str: "all_cube" | "all_vector" | "all" | "all_sub_vector"
    event_id,  # int: 0-15
)
```

Inserts a barrier that waits until all cores of the specified type have reached this point. Use when multiple cores write shared GM and you need a happens-before guarantee before any core reads it.

| `mode` | Waits for |
|--------|-----------|
| `"all_cube"` | All Cube cores |
| `"all_vector"` | All Vector cores |
| `"all"` | All Cube and Vector cores |
| `"all_sub_vector"` | All Vector sub-cores within one AI Core |

## 5. CV double-buffer pipeline pattern

The canonical pattern for pipelining Cube and Vector concurrently uses two workspace slots and initializes the pipeline by pre-setting both flags before the loop.

```python
import triton.experimental.tle as tle
pipe = tle.dsa.ascend.PIPE

@triton.jit
def cv_pipeline_kernel(..., NUM_BLOCKS: tl.constexpr):
    # Init: pre-set both slots so Cube can compute 2 blocks before first wait
    tle.dsa.ascend.sync_block_set('vector', 'cube', 0, pipe.PIPE_MTE2, pipe.PIPE_FIX)
    tle.dsa.ascend.sync_block_set('vector', 'cube', 1, pipe.PIPE_MTE2, pipe.PIPE_FIX)

    for block_idx in range(NUM_BLOCKS):
        buf_id = block_idx % 2

        # ── Cube stage ─────────────────────────────────────────────────────
        result = tl.dot(...)                       # Cube computes this block

        # Wait for Vector to free buf_id (init pre-arms first two iterations)
        tle.dsa.ascend.sync_block_wait('vector', 'cube', buf_id,
                                        pipe.PIPE_MTE2, pipe.PIPE_FIX)

        tl.store(workspace_ptr + buf_id * BLOCK + ..., result)

        # Signal Vector: buf_id data is ready
        tle.dsa.ascend.sync_block_set('cube', 'vector', buf_id,
                                       pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # Wait for Vector to finish consuming buf_id before next write
        tle.dsa.ascend.sync_block_wait('cube', 'vector', buf_id,
                                        pipe.PIPE_FIX, pipe.PIPE_MTE2)

        # ── Vector stage ───────────────────────────────────────────────────
        data = tl.load(workspace_ptr + buf_id * BLOCK + ...)
        output = tl.sum(data * weight, 0)
        tl.store(out_ptr + ..., output)

        # Signal Cube: buf_id slot is free for reuse
        tle.dsa.ascend.sync_block_set('vector', 'cube', buf_id,
                                       pipe.PIPE_MTE2, pipe.PIPE_FIX)
```

**Why pre-set both flags**: the first two loop iterations (`buf_id=0` and `buf_id=1`) hit `sync_block_wait('vector', 'cube', ...)` immediately. Without the init, they would block waiting for Vector to release slots that Vector has never been given. Pre-setting lets Cube run two blocks ahead before it ever has to wait.

**Measured speedup**: on the Lightning Indexer (attention score accumulation) kernel with a complex Vector stage, manual double-buffer sync is ~38% faster than the auto-sync pass (121 ms vs 196 ms average on a 32×8192×8192×2048 shape).

**Flag pairing rule**: every set/wait pair must use identical `(sender, receiver, event_id, sender_pipe, receiver_pipe)`. The direction swaps between the two roles (Cube sets, Vector waits; then Vector sets, Cube waits), but the pipe parameters stay the same within each direction.
