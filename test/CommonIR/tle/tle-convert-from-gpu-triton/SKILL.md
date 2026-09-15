---
name: tle-convert-from-gpu-triton
description: >-
  Step-by-step checklist for migrating a GPU Triton kernel to Ascend DSA TLE. Read this
  when converting an existing CUDA/GPU Triton operator — it covers import changes, grid
  strategy, tiling, DSA write-back, kernel launch args, and Ascend hardware limits.
  Distinct from tle-convert-pytorch-operator which covers writing from a PyTorch reference.
---

# Convert GPU Triton Operator to Ascend DSA TLE

This skill covers migrating an existing GPU (CUDA) Triton kernel to run on Ascend hardware using the TLE DSA path. The source is a working GPU Triton kernel; the target is an Ascend `@triton.jit` kernel using `tle.dsa.*` ops.

Extracted from real operator pairs: swiglu, add_rms_norm, causal_conv1d_fn, causal_conv1d_update.

---

## Migration Checklist

### Step 1 — Adjust imports

```python
# Remove GPU-specific imports
# from flag_gems.utils import tl_extra_shim   ← remove
# from flag_gems.utils import libentry         ← remove
# torch.cuda.*                                 ← replace with torch.npu.*

# Add DSA imports
import triton.experimental.tle as tle
import triton.language.math as math   # for exp, log, etc. (not tl.sigmoid)

# Math functions: use triton.language.math instead of tl.sigmoid or tl_extra_shim.exp
# tl.sigmoid(x)        → 1.0 / (1.0 + math.exp(-x))
# tl_extra_shim.exp(x) → math.exp(x)
```

Remove `@libentry()` decorator — keep only `@triton.jit`.

### Step 2 — Add NPU core-count awareness

Every DSA operator should query the number of vector cores and use that as the grid size:

```python
_CACHED_CORE_NUM = None

def _get_core_num():
    global _CACHED_CORE_NUM
    if _CACHED_CORE_NUM is None:
        try:
            current_device = torch.npu.current_device()
            torch.npu.set_device(current_device)
            cores_dict = torch.npu.get_device_limit(current_device)
            _CACHED_CORE_NUM = cores_dict["vector_core_num"] or 24
        except (AttributeError, KeyError, TypeError):
            _CACHED_CORE_NUM = 24
    return _CACHED_CORE_NUM
```

### Step 3 — Restructure tiling: 2D grid → 1D grid + inner loop

**GPU style**: `grid = (cdiv(M, BM), cdiv(N, BN))` — one program per tile, fixed tile size.

**DSA style**: `grid = (num_cores,)` — one program per core, kernel loops over multiple tiles:

```python
# Python-side: compute how much data each core handles
def _get_rows_and_cores(M, num_cores):
    TILE_SIZE_M = min(triton.next_power_of_2(M), 32)
    num_tiles_m = triton.cdiv(M, TILE_SIZE_M)
    num_cores = min(num_cores, num_tiles_m)
    BLOCK_SIZE_M = triton.cdiv(num_tiles_m, num_cores) * TILE_SIZE_M
    return TILE_SIZE_M, BLOCK_SIZE_M, num_cores

# Small data: single core
if M * N < 256 * 64:
    num_cores = 1
```

Inside the kernel, iterate over the assigned tile range:

```python
@triton.jit
def kernel(ptr, M, N, BLOCK_SIZE_M: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, ...):
    core_id = tl.program_id(0)
    m_start = core_id * BLOCK_SIZE_M

    for tile_m in tl.range(0, BLOCK_SIZE_M, TILE_M):
        m_idx = m_start + tile_m
        if m_idx >= M:
            break
        for tile_n in tl.range(0, N, TILE_N):
            # process (TILE_M, TILE_N) tile at (m_idx, tile_n)
            ...
```

### Step 4 — Dynamic tile sizes

Tile sizes must be powers-of-2 and respect UB limits (192 KB). Compute them in Python before launch:

```python
TILE_SIZE_M = min(triton.next_power_of_2(M), 32)   # cap at 32 rows
TILE_SIZE_H = min(triton.next_power_of_2(H), 256)  # cap at 256 cols

# Check UB before launching
ub_bytes = TILE_SIZE_M * TILE_SIZE_H * element_bytes
assert ub_bytes <= 192 * 1024, f"UB overflow: {ub_bytes}"
```

### Step 5 — Use DSA write-back for full tiles

Replace `tl.store + mask` with `tle.dsa.to_buffer + tle.dsa.copy` on complete (non-boundary) tiles:

```python
# Determine if this is a full tile (no boundary padding needed)
is_full_tile = (m_idx + TILE_M <= M) and (tile_n + TILE_N <= N)

if is_full_tile:
    out_buf = tle.dsa.to_buffer(out_tensor, space=tle.dsa.ascend.UB)
    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(out_buf, out_ptr + out_offset, [TILE_M, TILE_N])
else:
    # boundary tile: fall back to masked store
    mask = (row_ids < M - m_idx) & (col_ids < N - tile_n)
    tl.store(out_ptr + out_offset, out_tensor, mask=mask)
```

> `tle.dsa.copy` does not support masks — use it only for full tiles. `tle.dsa.hint(inter_no_alias=True)` tells the compiler that the source and destination buffers do not alias, allowing more aggressive DMA scheduling.

### Step 6 — Pre-load reused weights with `tle.dsa.extract_slice`

When a weight vector is reused across a token loop (e.g., depthwise conv weights), pre-load it once and slice per tap:

```python
# Pre-load weight tile outside the loop (KERNEL_WIDTH × BLOCK_N)
w_tile_2d = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)
w_tile = tl.reshape(w_tile_2d, (KERNEL_WIDTH * BLOCK_N,))  # flatten to 1D

# Inside the tap loop: extract slice for each kernel position
for j in tl.static_range(KERNEL_WIDTH):
    w_j = tle.dsa.extract_slice(
        w_tile,
        offsets=(j * BLOCK_N,),
        sizes=(BLOCK_N,),
        strides=(1,),
    )
    # use w_j as the weight for tap j
    acc += x_col * w_j
```

This avoids repeated `tl.load` calls for constant data and keeps weights in registers/UB.

### Step 7 — Add kernel launch args for Ascend multibuffer

These three launch kwargs activate the Ascend compiler's DMA pipeline, letting load and compute overlap:

```python
kernel[(grid,)](
    ...,
    multibuffer=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    limit_auto_multi_buffer_only_for_local_buffer=False,
)
```

For kernels with explicit K-loops, also set `num_stages=2` at launch.

### Step 8 — Replace Python `range` with `tl.range` / `tl.static_range`

```python
# GPU style — Python range, Triton unrolls automatically
for k in range(K // BLOCK_K):
    ...

# DSA style — tl.range gives the compiler explicit pipeline control
for k in tl.range(0, K, BLOCK_K):
    ...

# Use tl.static_range when the loop variable is used as a constexpr index
# (e.g., as a tile_subview offset)
for j in tl.static_range(KERNEL_WIDTH):
    w_j = tle.dsa.extract_slice(w_tile, offsets=(j * BLOCK_N,), ...)
```

### Step 9 — Convert runtime conditions to constexpr flags

```python
# GPU style: runtime if branch
def forward(x, use_apc=False):
    kernel[grid](..., use_apc=use_apc)   # runtime arg

# DSA style: compile-time constexpr eliminates branch overhead
@triton.jit
def kernel(..., IS_APC_ENABLED: tl.constexpr):
    if IS_APC_ENABLED:
        ...  # zero cost when False

# At launch: pass as constexpr
kernel[grid](..., IS_APC_ENABLED=use_apc)
```

### Step 10 — Split input halves Python-side

For operators that read two halves of the same tensor (e.g., SwiGLU), split in Python rather than computing the offset inside the kernel:

```python
# GPU style: kernel computes both offsets internally
kernel[grid](input_ptr, H=H, ...)  # kernel reads at [0:H] and [H:2H]

# DSA style: split before kernel, pass two clean pointers
a, b = torch.split(input, input.shape[-1] // 2, dim=-1)
a, b = a.contiguous(), b.contiguous()
kernel[grid](a, b, H=H, ...)
```

### Step 11 — Protect against Ascend hardware limits

```python
# UB limit: avoid single-block tensors > 4096 elements wide
_BLOCK_N = 4096  # if N > 4096, use a loop kernel instead of a single-block kernel

# Program count limit: M is capped at 65535
M = min(math.prod(x.shape[:dim]), 65535)

# No stride args needed when inputs are contiguous (use dimension size directly)
# GPU: pass x_stride_r, x_stride_c; DSA: assume contiguous, use N as offset step
```

---

## Operator-Specific Examples

### Elementwise (SwiGLU)

Before (GPU):
```python
# 2D grid, fixed BLOCK_SIZE_M=64 BLOCK_SIZE_H=64
# reads both halves from same pointer: input_ptr + H * stride
# uses tl.sigmoid, tl.store with mask always
```

After (DSA):
```python
# Python-side: torch.split(input, H, dim=-1) → a, b
# 1D grid = (num_cores,), dynamic TILE_SIZE_M/H
# manual sigmoid: 1.0 / (1.0 + math.exp(-x_a_f))
# full tiles: tle.dsa.to_buffer + tle.dsa.copy with hint(inter_no_alias=True)
# boundary tiles: tl.store + mask
# launch: multibuffer=True, limit_auto_multi_buffer_of_local_buffer="no-limit"
```

### Normalization (RMSNorm / AddRMSNorm)

Before (GPU): one kernel, one program per row (grid = M), in-place writes.

After (DSA):
- Out-of-place: allocate new output tensor, never write back to input.
- Two-kernel split: `N <= 4096` → single-block kernel with weight preloaded outside row loop. `N > 4096` → column-loop kernel accumulates variance across `BLOCK=4096` chunks.
- Multi-row per core: `_get_rows_and_cores(M, num_cores)` assigns `N_ROWS` rows per program.
- `M = min(..., 65535)` program count cap.

### Depthwise Conv1d State Update

After (DSA):
- `B_TILE` sequences per program to reduce launch overhead. `lane_active` predicate gates out-of-bounds lanes (no early `return` inside tile loop).
- Three-path state roll: shift+append logic fully inside kernel using `tl.static_range` + T_CHUNK chunks.
- `IS_SPEC_DECODING` / `IS_VARLEN` constexpr feature flags — zero cost when disabled.
- `tle.dsa.extract_slice` for per-tap weight slices from a preloaded 1D buffer.
