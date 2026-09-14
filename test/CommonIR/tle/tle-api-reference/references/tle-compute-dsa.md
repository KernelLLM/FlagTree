# TLE Compute — DSA Path (Vector Ops, Cube MMA, Synchronization)

> **Layer**: DSA path only. Targets Ascend 910B and compatible hardware.
> **Source**: `FlagTree/python/triton/experimental/tle/language/dsa/core.py`, `dsa/ascend/core.py`

This document covers element-wise vector ops on UB buffers, Cube MMA launch, and the cross-engine synchronization primitives used in 3-task pipelined kernels.

## 1. Element-wise vector ops on UB

These ops operate on `tle.dsa.buffer` objects in `ascend.UB`. All accept `(input, other, result)` and write into `result` in-place.

```python
tle_dsa.add(a_ub, b_ub, c_ub)   # c_ub = a_ub + b_ub
tle_dsa.sub(a_ub, b_ub, c_ub)   # c_ub = a_ub - b_ub
tle_dsa.mul(a_ub, b_ub, c_ub)   # c_ub = a_ub * b_ub
tle_dsa.div(a_ub, b_ub, c_ub)   # c_ub = a_ub / b_ub
tle_dsa.max(a_ub, b_ub, c_ub)   # c_ub = max(a_ub, b_ub)
tle_dsa.min(a_ub, b_ub, c_ub)   # c_ub = min(a_ub, b_ub)
```

The result buffer (`c_ub`) must be pre-allocated with `tle_dsa.alloc`. All three buffers must have the same shape and dtype, and all must reside in `ascend.UB`.

> **Note**: These ops lower to `arith.*` ops on `tl.tensor` values inside CommonIR. They are not generic `tl.*` tensor ops — they take `buffer` objects, not `tl.tensor` values directly. To combine with standard Triton arithmetic, first call `tle_dsa.to_tensor`.

## 2. Cube MMA: `tile_cube_launch` and `tile_cube_wait`

For matrix multiplication using the Cube engine, use the low-level CommonIR ops:

```python
tle_dsa.tile_cube_launch(
    a,           # tle.buffer in L0A
    b,           # tle.buffer in L0B
    acc,         # tle.buffer in L0C (accumulator, updated in-place)
    stage_a,     # pipeline stage index for A (int)
    stage_b,     # pipeline stage index for B (int)
    dst,         # tle.buffer for the MMA result (L0C)
    transpose_a, # bool
    transpose_b, # bool
    init,        # bool — True to zero-initialize acc before MMA
    mma,         # bool — True to perform the MMA; False for init-only
)

tle_dsa.tile_cube_wait()  # synchronization barrier after cube engine completes
```

Full MMA sequence:

```python
# Allocate operand buffers
a_l1   = tle_dsa.alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L1)
b_l1   = tle_dsa.alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L1)
a_l0a  = tle_dsa.alloc([BLOCK_M, BLOCK_K], tl.float16, ascend.L0A)
b_l0b  = tle_dsa.alloc([BLOCK_K, BLOCK_N], tl.float16, ascend.L0B)
c_l0c  = tle_dsa.alloc([BLOCK_M, BLOCK_N], tl.float32, ascend.L0C)

# Load from global memory to L1
tle_dsa.copy(a_ptr, a_l1, [BLOCK_M, BLOCK_K])
tle_dsa.copy(b_ptr, b_l1, [BLOCK_K, BLOCK_N])

# L1 → L0A / L0B
tle_dsa.copy(a_l1, a_l0a, [BLOCK_M, BLOCK_K])
tle_dsa.copy(b_l1, b_l0b, [BLOCK_K, BLOCK_N])

# Cube MMA (init=True zeros the accumulator first)
tle_dsa.tile_cube_launch(a_l0a, b_l0b, c_l0c, 0, 0, c_l0c,
                          False, False, init=True, mma=True)
tle_dsa.tile_cube_wait()

# Read result from L0C into UB for postprocessing
c_ub = tle_dsa.alloc([BLOCK_M, BLOCK_N], tl.float32, ascend.UB)
tle_dsa.copy(c_l0c, c_ub, [BLOCK_M, BLOCK_N])
```

## 3. `tle_dsa.parallel` — sub-vector-core fan-out

```python
for sub_id in tle_dsa.parallel(start, stop, step, num_stages):
    ...
```

On Ascend 910B, the Vector engine can be split into two sub-cores. `parallel` creates a loop that fans out over `ascend.sub_vec_num` sub-cores and uses `ascend.sub_vec_id` inside the loop body to identify which sub-core is executing.

```python
for i in tle_dsa.parallel(0, tl.cdiv(N, BLOCK), 1, 1):
    sub_off = i * BLOCK + ascend.sub_vec_id * (BLOCK // ascend.sub_vec_num)
    x_ub = tle_dsa.alloc([BLOCK // 2], tl.float32, ascend.UB)
    tle_dsa.copy(x_ptr + sub_off, x_ub, [BLOCK // 2])
    ...
```

## 4. Cross-engine synchronization: `tile_set_flag` / `tile_wait_flag`

The Ascend 910B has three concurrent engines: DMA (MTE1/MTE2), Cube, and Vector. To pipeline across engines without stalling unnecessarily, use flag-based synchronization:

```python
tle_dsa.tile_set_flag(
    producer_pipe,   # ascend.PIPE constant — which engine raised the event
    consumer_pipe,   # ascend.PIPE constant — which engine waits for it
    event_id,        # int 0-15 — unique identifier for this handshake
)

tle_dsa.tile_wait_flag(
    producer_pipe,   # same as in tile_set_flag
    consumer_pipe,   # same as in tile_set_flag
    event_id,        # same event_id as the matching set_flag
)
```

`tile_set_flag` must always be paired with exactly one `tile_wait_flag` that uses the same `(producer_pipe, consumer_pipe, event_id)` triple.

**Typical pipe constants** (under `ascend.PIPE`):

| Constant | Engine |
|----------|--------|
| `PIPE.MTE1` | DMA from L1 to L0 |
| `PIPE.MTE2` | DMA from GM to L1/UB |
| `PIPE.MTE3` | DMA from UB to GM |
| `PIPE.M` | Cube engine (MMA) |
| `PIPE.V` | Vector engine |
| `PIPE.FIX` | Fixed-point/scalar unit |

See the `tle-commonir-patterns` skill for the complete 3-task pipeline pattern that uses these flags.

## 5. `tile_pipe_barrier` — intra-engine barrier

```python
tle_dsa.tile_pipe_barrier(pipe)  # stalls until all prior ops on `pipe` complete
```

Use when you need to ensure all preceding ops on a single engine are visible before the next op on the same engine, without coordinating with other engines. This is lighter weight than `tl.debug_barrier`.

## 6. `tle_dsa.hint` — compile hint

```python
tle_dsa.hint(double_buffer=True, pipeline_stage=2)
```

`hint` is consumed by the AST parser and forwarded as `tile.compile_hint` annotations in MLIR. It is not a runtime op. Place it at the start of a loop body to annotate that loop's schedule.

**Next**: [tle-commonir-patterns](../tle-commonir-patterns/SKILL.md) for the 3-task DMA+Cube+Vector pipeline pattern on Ascend.
