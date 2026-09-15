# TLE Kernel Basics

> **Layer**: Shared by both GPU and DSA paths.
> **Source**: `FlagTree/python/triton/experimental/tle/__init__.py`, `FlagTree/python/triton/experimental/tle/language/__init__.py`

A TLE kernel is a standard Triton `@triton.jit` function. TLE ops appear inside that function body alongside standard `tl.*` calls; there is no separate entry point or compilation pipeline to invoke.

## 1. Canonical imports

```python
import triton
import triton.language as tl

# GPU (Hopper) path
from triton.experimental.tle.language import gpu as tle_gpu
# -- or, using the top-level alias after `from triton.experimental import tle`:
import triton.experimental.tle as tle   # exposes tle.alloc, tle.copy, tle.local_ptr, etc.

# DSA (Ascend) path
from triton.experimental.tle.language import dsa as tle_dsa
# Ascend address spaces live one level deeper:
import triton.experimental.tle.language.dsa.ascend as ascend  # UB, L1, L0A, L0B, L0C
```

The two paths are independent. A kernel targeting Ascend does not import `gpu`; a Hopper kernel does not import `dsa`.

## 2. GPU path kernel skeleton

```python
import triton
import triton.language as tl
import triton.experimental.tle as tle

@triton.jit
def kernel(a_ptr, b_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # Allocate on-chip SMEM buffer
    a_smem = tle.alloc([BLOCK], dtype=tl.float32, scope=tle.smem)

    # DMA from global memory into SMEM
    tle.copy(a_ptr + offsets, a_smem, [BLOCK])

    # Materialise SMEM pointers for standard tl.load
    idx = tl.arange(0, BLOCK)
    a_ptrs = tle.local_ptr(a_smem, (idx,))
    a_tile = tl.load(a_ptrs)

    # Standard compute then store to global memory
    result = a_tile + tl.load(b_ptr + offsets)
    tl.store(out_ptr + offsets, result)
```

## 3. DSA path kernel skeleton

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend

@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tail = tl.minimum(n - pid * BLOCK, BLOCK)

    # Allocate on-chip UB (Unified Buffer) for each operand
    x_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
    y_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)
    z_ub = tle_dsa.alloc([BLOCK], dtype=tl.float32, mem_addr_space=ascend.UB)

    # DMA from global memory into UB
    tle_dsa.copy(x_ptr + offsets, x_ub, [tail])
    tle_dsa.copy(y_ptr + offsets, y_ub, [tail])

    # Vector add on UB
    tle_dsa.add(x_ub, y_ub, z_ub)

    # DMA result back to global memory
    tle_dsa.copy(z_ub, out_ptr + offsets, [tail])
```

## 4. Key differences between paths

| Concern | GPU path | DSA path |
|---------|----------|----------|
| Buffer type | `buffered_tensor` (carries layout + scope) | `buffer` (carries `buffer_type` with address space) |
| On-chip scopes | `tle.smem`, `tle.tmem` | `ascend.UB`, `L1`, `L0A`, `L0B`, `L0C` |
| Copy direction | Auto-detected: global↔SMEM | Auto-detected: GM↔on-chip |
| Compute after copy | `tle.local_ptr` + `tl.load` → standard ops | Direct ops on buffers (`tle_dsa.add`, `tile_cube_launch`, etc.) |
| Lowering target | Hopper async-copy / TMA intrinsics | CommonIR `tile.*` MLIR dialect → HIVM/Linalg |
| Pipeline control | `tle.pipeline(start, stop, step, num_stages)` | `tle_dsa.pipeline` or `tle_dsa.parallel` |

> **Note**: DSA `buffer` objects are not `tl.tensor` instances. To feed a buffer into a standard `tl.dot` or Triton arithmetic op, first call `tle_dsa.to_tensor(buf, writable=False)`.

## 5. Verifying DSA kernel lowering

After writing a DSA kernel, verify it lowers to `tile.*` ops rather than legacy `memref` IR:

```python
import triton
from triton.experimental.tle.language import dsa as tle_dsa

mlir = triton.compile(kernel, ..., dump_ir="ttir")
assert "tile.alloc" in mlir
assert "tile.to_tensor" in mlir
assert "memref.alloc" not in mlir          # legacy path — must not appear
assert "bufferization.to_tensor" not in mlir  # legacy path — must not appear
assert "#hivm.address_space" not in mlir   # legacy path — must not appear
```

**Next**: [tle-memory-gpu.md](tle-memory-gpu.md) for GPU on-chip buffer details, or [tle-memory-dsa.md](tle-memory-dsa.md) for DSA.
