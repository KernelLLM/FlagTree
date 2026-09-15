---
name: tle-convert-pytorch-operator
description: >-
  Convert a PyTorch operator into a TLE DSA-backed Ascend kernel. Read this when writing
  a new TLE kernel from scratch given a PyTorch reference — it covers the kernel skeleton,
  tiling strategy, the alloc/copy/compute sequence, and validation.
---

# Convert PyTorch Operator to TLE DSA

This skill covers writing a TLE DSA Ascend kernel that matches the behavior of a given PyTorch reference operator. The output is a `@triton.jit` kernel that uses `tle.dsa.*` ops to compute the result on Ascend hardware.

## Inputs

- A PyTorch operator (a `torch.*` call, `nn.Module.forward`, or a Python function using torch ops)
- The target shapes and dtypes
- Optional: performance constraints (tile size hints, memory budget)

## Outputs

- A `@triton.jit` kernel using `tle.dsa.*` ops
- A factory function that compiles the kernel for specific shapes
- A correctness check against the PyTorch reference

## API Hierarchy

TLE DSA ops have a natural ordering from simplest to most hardware-explicit:

```
┌─────────────────────────────────────────────────────────────┐
│  Level 3: Multi-engine explicit  (DMA + Cube + Vector ring) │
│  · tile_alloc on L1/L0A/L0B/L0C                            │
│  · tile_set_flag / tile_wait_flag for engine sync           │
│  · tile_cube_launch / tile_cube_wait                        │
│  · tle.scope wrappers (cube/vector scope regions)           │
├─────────────────────────────────────────────────────────────┤
│  Level 2: Single-engine matmul  (Cube via L1)               │
│  · tile_alloc on L1                                         │
│  · tile_copy + tile_to_tensor + tl.dot                      │
│  · tl.debug_barrier between copy and dot                    │
├─────────────────────────────────────────────────────────────┤
│  Level 1: Vector-only  (UB ops, no Cube)                    │
│  · tle_dsa.alloc on ascend.UB                               │
│  · tle_dsa.copy + tle_dsa.add/sub/mul/div/max/min           │
│  · to_tensor for standard tl.* arithmetic                   │
└─────────────────────────────────────────────────────────────┘
```

Start at Level 1. Move to Level 2 only for matrix multiplication. Move to Level 3 only when Cube + Vector must run concurrently in the same kernel.

## Required Workflow

1. Identify the PyTorch op type: elementwise, reduction, matmul, or fused (matmul + elementwise).
2. Choose the API level (see table above).
3. Write the kernel using the pattern card for that level (see `tle-commonir-patterns`).
4. Choose tile sizes that fit in UB/L1 (see UB budget rule below).
5. Write a factory function with `BLOCK_SIZE` etc. as `tl.constexpr` parameters.
6. Validate correctness by comparing to the PyTorch reference.

## Tiling Rule

For a 1D tensor of size `N`: tile with `BLOCK = 2048` as a starting point. Adjust down if the element size is large.

For a 2D matrix of shape `[M, N]` (matmul `[M, K] × [K, N]`):
- `BLOCK_M = 128`, `BLOCK_N = 256`, `BLOCK_K = 128` for float16 — a reliable starting point on 910B.
- Verify UB/L1 budget before finalizing (see below).

For attention (`[B, H, S, D]`): tile over the sequence dimension with `BLOCK_M` (query rows) and `BLOCK_N` (KV columns).

## UB Budget Rule

Before writing a kernel, verify the UB allocations fit:

```
UB limit: 192 KB = 196,608 bytes

UB bytes used = sum over all tle_dsa.alloc(ascend.UB):
    product(shape) × element_size_bytes

float16 = 2 bytes, float32 = 4 bytes, bfloat16 = 2 bytes
```

If UB usage exceeds 192 KB, reduce tile sizes. For L1:

```
L1 limit: ~512 KB

L1 bytes used = sum over all tile_alloc(ascend.L1):
    product(shape) × element_size_bytes
```

## Kernel Skeleton — Level 1 (Elementwise)

For pointwise ops (add, mul, relu, gelu, cast, etc.):

```python
import triton
import triton.language as tl
from triton.experimental.tle.language import dsa as tle_dsa
from triton.experimental.tle.language.dsa import ascend

@triton.jit
def elementwise_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    tail = tl.minimum(n_elements - pid * BLOCK, BLOCK)

    x_ub = tle_dsa.alloc([BLOCK], dtype=tl.float16, mem_addr_space=ascend.UB)
    y_ub = tle_dsa.alloc([BLOCK], dtype=tl.float16, mem_addr_space=ascend.UB)
    z_ub = tle_dsa.alloc([BLOCK], dtype=tl.float16, mem_addr_space=ascend.UB)

    tle_dsa.copy(x_ptr + offsets, x_ub, [tail])
    tle_dsa.copy(y_ptr + offsets, y_ub, [tail])

    tle_dsa.add(x_ub, y_ub, z_ub)  # replace with sub/mul/div/max/min as needed

    tle_dsa.copy(z_ub, out_ptr + offsets, [tail])


def run_elementwise(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    BLOCK = 2048
    grid = (triton.cdiv(x.numel(), BLOCK),)
    elementwise_kernel[grid](x, y, out, x.numel(), BLOCK=BLOCK)
    return out
```

For ops not directly available (softmax, exp, sqrt, etc.), use `tle_dsa.to_tensor` to get a `tl.tensor` then use standard Triton arithmetic:

```python
# After copy into UB:
x_t = tle_dsa.to_tensor(x_ub, writable=False)
result_t = tl.exp(x_t)  # standard tl.* op
result_ub = tle_dsa.to_buffer(result_t, space=ascend.UB, bind_buffer=z_ub)
tle_dsa.copy(result_ub, out_ptr + offsets, [tail])
```

## Kernel Skeleton — Level 2 (Matmul)

For `torch.matmul(A, B)` or `torch.mm`:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Allocate doubled L1 for ping-pong
    a_l1 = tle_dsa.tile_alloc([2*BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
    b_l1 = tle_dsa.tile_alloc([BLOCK_K, 2*BLOCK_N], tl.float16, ascend.L1)

    a_l1_0 = tle_dsa.tile_subview(a_l1, [0,       0], [BLOCK_M, BLOCK_K], [1, 1])
    a_l1_1 = tle_dsa.tile_subview(a_l1, [BLOCK_M, 0], [BLOCK_M, BLOCK_K], [1, 1])
    b_l1_0 = tle_dsa.tile_subview(b_l1, [0, 0      ], [BLOCK_K, BLOCK_N], [1, 1])
    b_l1_1 = tle_dsa.tile_subview(b_l1, [0, BLOCK_N], [BLOCK_K, BLOCK_N], [1, 1])

    num_k = tl.cdiv(K, BLOCK_K)

    # Prologue: fill slot 0
    a0_ptr = tle_dsa.tile_gm_offset(a_ptr, [pid_m*BLOCK_M, 0], [K, 1])
    b0_ptr = tle_dsa.tile_gm_offset(b_ptr, [0, pid_n*BLOCK_N], [N, 1])
    tle_dsa.tile_copy(a0_ptr, a_l1_0, [BLOCK_M, BLOCK_K])
    tle_dsa.tile_copy(b0_ptr, b_l1_0, [BLOCK_K, BLOCK_N])
    tl.debug_barrier()

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in tl.static_range(num_k):
        a_cur = a_l1_0 if k % 2 == 0 else a_l1_1
        b_cur = b_l1_0 if k % 2 == 0 else b_l1_1
        a_nxt = a_l1_1 if k % 2 == 0 else a_l1_0
        b_nxt = b_l1_1 if k % 2 == 0 else b_l1_0

        # Dot on current slot
        a_t = tle_dsa.tile_to_tensor(a_cur, writable=False)
        b_t = tle_dsa.tile_to_tensor(b_cur, writable=False)
        acc = tl.dot(a_t, b_t, acc, input_precision="ieee")

        # Prefetch next slot
        if k + 1 < num_k:
            an_ptr = tle_dsa.tile_gm_offset(a_ptr, [pid_m*BLOCK_M, (k+1)*BLOCK_K], [K, 1])
            bn_ptr = tle_dsa.tile_gm_offset(b_ptr, [(k+1)*BLOCK_K, pid_n*BLOCK_N], [N, 1])
            tle_dsa.tile_copy(an_ptr, a_nxt, [BLOCK_M, BLOCK_K])
            tle_dsa.tile_copy(bn_ptr, b_nxt, [BLOCK_K, BLOCK_N])
            tl.debug_barrier()

    c_ptr_tile = tle_dsa.tile_gm_offset(c_ptr, [pid_m*BLOCK_M, pid_n*BLOCK_N], [N, 1])
    tl.store(c_ptr_tile + tl.arange(0, BLOCK_M)[:, None] * N
                        + tl.arange(0, BLOCK_N)[None, :],
             acc.to(tl.float16))


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape; _, N = b.shape
    c = torch.empty((M, N), dtype=torch.float16, device=a.device)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](a, b, c, M, N, K,
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return c
```

## Correctness Validation

After writing the kernel, always compare against the PyTorch reference:

```python
import torch

a = torch.randn(M, K, dtype=torch.float16, device="npu")
b = torch.randn(K, N, dtype=torch.float16, device="npu")

ref = torch.matmul(a.float(), b.float()).half()
out = matmul(a, b)

max_diff = (out - ref).abs().max().item()
assert max_diff < 0.1, f"Max diff {max_diff} exceeds tolerance"
print(f"PASS: max_diff={max_diff:.4f}")
```

For float16 matmul: `atol=0.1` is typical. For purely elementwise ops: `atol=1e-3`.

## When to Use Level 3

Use the three-task pipeline (Level 3) only when:
- The operator requires both Cube (matmul) and Vector (activation, softmax, bias) in the same kernel pass
- The K dimension is large enough that the pipeline overhead is justified (K ≥ 256)
- Both engines need to be busy simultaneously to meet throughput targets

For a first correct implementation, write Cube and Vector as separate kernels (or Level 2 for the matmul + standard `tl.*` for the Vector part), verify correctness, then fuse into Level 3 if profiling shows the boundary is a bottleneck.

## Constraints

- All tile shape arguments must be `tl.constexpr`.
- Do not use Python runtime values (non-constexpr) as `tile_subview` offsets — see repair guide entry 2.
- Do not allocate two separate `tile_alloc` with identical shape/dtype for double-buffering — see repair guide entry 1.
- Use `tl.static_range` when the loop variable is used as a constexpr index.
- Verify UB budget before writing the kernel (192 KB limit).
