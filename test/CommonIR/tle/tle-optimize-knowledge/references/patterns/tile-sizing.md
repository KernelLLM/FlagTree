---
priority: high
---

# Tile Size Selection — BLOCK_M / BLOCK_N / BLOCK_K

## Summary

Choosing correct tile sizes is the first and most impactful optimization decision for a TLE DSA Ascend matmul or attention kernel. Tile sizes control L1 residency, Cube pipeline utilization, and DMA transfer efficiency simultaneously. The goal is tiles large enough to keep the Cube engine fed (avoids per-tile setup overhead) while staying within L1 and UB capacity limits.

## Use When

- Writing a new matmul, batched matmul, or attention kernel and need starting tile sizes.
- The kernel runs correctly but is slow, and profiling shows either low Cube utilization or high DMA stall cycles.
- Changing BLOCK_K doesn't improve throughput — the tile may be too small to amortize Cube startup overhead.

## Avoid When

- The kernel is not yet correct — fix correctness first, then tune tile sizes.
- Tile sizes are already constrained by a semantic requirement (e.g., BLOCK_N must equal the head dimension D in FlashAttention).
- You haven't verified the UB/L1 budget for your chosen sizes (see budget formula below).

## Pattern

### Step 1: Choose starting tile sizes

For float16 matmul `[M, K] × [K, N]` on Ascend 910B:

| Dimension | Starting value | Notes |
|-----------|---------------|-------|
| BLOCK_M | 128 | Rows of A tile; must be divisible into M |
| BLOCK_N | 256 | Columns of B tile; must be divisible into N |
| BLOCK_K | 128 | Shared K tile; controls L1 residency of A and B |

For bfloat16 or float32, start with the same values but verify L1 budget more carefully (float32 doubles the byte count).

For attention kernels (`[B, H, S, D]`):

| Dimension | Starting value | Notes |
|-----------|---------------|-------|
| BLOCK_M (query rows) | 64 | Smaller than matmul because attention has more buffers live simultaneously |
| BLOCK_N (KV cols) | 64 | Must fit Q, K, V, and score tiles in L1 |

### Step 2: Verify L1 budget

```python
# float16 matmul double-buffer: two A tiles + two B tiles in L1
L1_bytes = (
    2 * BLOCK_M * BLOCK_K * 2 +   # a_l1_0 + a_l1_1 (float16)
    2 * BLOCK_K * BLOCK_N * 2      # b_l1_0 + b_l1_1 (float16)
)
assert L1_bytes <= 512 * 1024, f"L1 overflow: {L1_bytes} bytes"

# For attention with Q/K/V tiles + score buffer:
L1_bytes = (
    BLOCK_M * D * 2 +      # Q tile
    BLOCK_N * D * 2 +      # K tile
    BLOCK_N * D * 2 +      # V tile
    BLOCK_M * BLOCK_N * 2  # score tile (QKᵀ result)
)
assert L1_bytes <= 512 * 1024, f"L1 overflow: {L1_bytes} bytes"
```

### Step 3: Verify UB budget

```python
# UB limit: 192 KB
UB_bytes = (
    BLOCK_M * BLOCK_N * 4   # accumulator in float32
    # + any other UB buffers (bias, residual, activation intermediate)
)
assert UB_bytes <= 192 * 1024, f"UB overflow: {UB_bytes} bytes"
```

### Step 4: BLOCK_K trade-off

Larger BLOCK_K reduces K-loop iterations (less loop overhead, better DMA utilization) but increases L1 pressure. On 910B:

- `BLOCK_K = 128` is the practical upper bound for float16 when using double-buffered A and B tiles.
- `BLOCK_K = 64` if L1 is tight (large BLOCK_M × BLOCK_N) or if the total K is small.
- `BLOCK_K = 32` only as a last resort when K itself is small (K < 256).

Smaller BLOCK_K at fixed BLOCK_M × BLOCK_N increases the K-loop trip count, which can expose more pipeline stages but also adds more synchronization overhead per trip.

### Step 5: Divisibility

All tile dimensions must exactly divide the corresponding matrix dimensions, or you must add masking logic for the tail. For a first implementation, pad the input to the next multiple of the tile size and avoid masking complexity:

```python
M_padded = triton.cdiv(M, BLOCK_M) * BLOCK_M
N_padded = triton.cdiv(N, BLOCK_N) * BLOCK_N
K_padded = triton.cdiv(K, BLOCK_K) * BLOCK_K
a = torch.nn.functional.pad(a, (0, K_padded - K, 0, M_padded - M))
b = torch.nn.functional.pad(b, (0, N_padded - N, 0, K_padded - K))
```

## What To Verify After Applying

- L1 budget check passes (see Step 2 formula).
- UB budget check passes (see Step 3 formula).
- `triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)` gives the correct grid size.
- Output is numerically correct before profiling (compare to PyTorch reference with atol=0.1).

## Related Patterns

- **Autotune** — use after fixing tile sizes manually to search the space systematically.
- **compile_hint Usage** — when L1 still overflows after reducing tile sizes, `hivm.tile_mix_cube_num` can sub-tile internally.
