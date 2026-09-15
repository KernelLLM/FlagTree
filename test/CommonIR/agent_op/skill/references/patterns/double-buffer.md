---
priority: high
---

# Double Buffering (TLE DSA)

## Summary

Overlap MTE data prefetch with Cube compute using ping-pong buffering, hiding GM→L1 latency behind MMA. TLE DSA offers three mechanisms; this file ranks them by how much evidence stands behind each on Ascend.

## Use When

- The K/V load loop is memory-bound — MTE2 copy time dominates Cube compute time.
- L1 capacity holds two buffer sets (or one 2×-sized buffer).
- The lockstep single-buffer version already passes its numeric gate on device.

## Avoid When

- The kernel is compute-bound.
- You cannot fit two buffer sets.
- The previous pipeline stage (cv-sync / workspace-pipeline) has not passed numerics yet — double buffering multiplies the debugging surface.

## The cardinal rule

**Never ping-pong between two same-shape `tile_alloc`s.** `commonir_to_hivm` merges identical shape/dtype allocs into one physical cbuf; both "buffers" alias and the prefetch overwrites the data dot is reading. `matmul_double_buffer.py` is kept in `success_case` with exactly this warning — it compiles and runs but is documented to produce WRONG results. Use one 2×-sized buffer plus `subview` slots instead.

## Mechanism A — `dsa.pipeline(num_stages=2)` (preferred first step)

Automatic software pipelining: the compiler double-buffers the `dsa.copy` DMA against compute. Idiomatic AscendC multi-buffer replacement, verified in `mhc_post.py` and tutorials:

```python
# Launch with multibuffer=True (default) so eligible buffers are doubled.
x_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=UB)   # allocated ONCE

for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
    tle.dsa.copy(x_ptr + offs, x_ub, [tail_d])        # DMA stage (MTE2)
    t = tle.dsa.to_tensor(x_ub).to(tl.float32)        # compute stage (Vector)
    ...

with tle.dsa.hint(inter_no_alias=True):               # copies across iterations don't alias
    tle.dsa.copy(y_ub, out_ptr + offs, [tail_d])
```

Per-tensor multi-buffer annotation also exists: `extension.multibuffer(tensor, 2)` (size must be 2). Related launch knobs: `limit_auto_multi_buffer_only_for_local_buffer=False`, `limit_auto_multi_buffer_of_local_buffer="no-limit"`.

Caution (from the tilelang `auto_pipeline` postmortem, mechanism-independent): pipelining multiplies the footprint of buffers in the loop body. Budget UB explicitly before raising `num_stages`.

## Mechanism B — one 2× L1 buffer + subview ping-pong (manual, FA-relevant)

Verified form family: `native_matmul_dsa_slice.py` (ping-pong inside one 2× buffer) + `native_fa.py` (whole-tile L1 → dot). For FA K/V tiles:

```python
# ONE allocation, doubled leading dim
kv_l1 = tle.dsa.alloc([2, BLOCK_N, DIM], dtype=tl.float16, mem_addr_space=L1)

for g in range(num_kv_tiles):
    cur = g % 2
    nxt = 1 - cur
    # prefetch NEXT tile into the other slot
    if g + 1 < num_kv_tiles:
        dst = tle.dsa.subview(kv_l1, offsets=[nxt, 0, 0], sizes=[1, BLOCK_N, DIM], strides=[1, 1, 1])
        tle.dsa.copy(k_block_ptr_next, dst, [1, BLOCK_N, DIM])   # or 2D view
    # compute on CURRENT slot: whole-tile to_tensor, then dot
    cur_view = tle.dsa.subview(kv_l1, offsets=[cur, 0, 0], sizes=[1, BLOCK_N, DIM], strides=[1, 1, 1])
    k_t = tle.dsa.to_tensor(cur_view, writable=False, target_shape=[BLOCK_N, DIM])
    s = tl.dot(q, tl.trans(k_t), out_dtype=tl.float32)
```

Forbidden chain (on-device failure): `subview → to_tensor → tl.trans → tl.dot`. If `target_shape` + trans on a subviewed buffer misbehaves, fall back to copying the slot into a plain register tensor (`tl.load` from GM, or a whole-tile L1 `to_tensor` per buffer half allocated with **distinct shapes** to dodge the merge) — and gate on numerics.

Ordering: when not using `dsa.pipeline`, order the prefetch DMA against compute with `tl.debug_barrier()` between copy and dot (the `matmul_double_buffer.py` ordering idiom — its *ordering* is fine; only its two-alloc structure is broken).

## Mechanism C — explicit prefetch + `tl.debug_barrier()`

From `matmul_double_buffer.py` / `matmul_double_buffer_serial.py`: prefetch next K-tile into the idle slot, `tl.debug_barrier()`, dot from the current slot, swap. Keep this only as a stepping stone toward Mechanism B; on its own it carries the alloc-merge hazard above.

## What about L0A/L0B/L0C double buffering?

The tilelang pattern double-buffers L0 registers per sub-block MMA. In TLE DSA the low-level surface exists (`dsa.alloc(..., L0A/L0B/L0C)`, `tile_cube_launch(a, b, acc, stage_a, stage_b, dst, transpose_b=..., init=..., mma=...)` + `tile_cube_wait()`), but per `fa_triton_arch.py` comments, production kernels still use synchronous `tl.dot` + `tl.store` for cube work — the full `tile_cube_launch` token semantics/lowering were not confirmed. **Recommendation: stay on `tl.dot` and let the compiler manage L0; revisit `tile_cube_launch` only with a dedicated bring-up.**

## Timing diagram (2-deep)

| Time | MTE Copy | Cube Compute |
|------|----------|-------------|
| t₀ | copy tile 0 → slot 0 | |
| t₁ | copy tile 1 → slot 1 | dot tile 0 |
| t₂ | copy tile 2 → slot 0 | dot tile 1 |
| t₃ | | dot tile 2 |

## What To Verify After Applying

- Ping-pong uses ONE 2×-sized alloc + `subview` (or allocs with deliberately different shapes), never two identical allocs.
- UB/L1 budget re-checked with the doubling in effect (and ×2 again if `multibuffer=True` applies to these buffers).
- dot operands avoid the subview→trans chain.
- Numeric gate first, then measure: prefetch without correctness is not progress.
- Run `tle_sync_lint.py` when combined with manual `sync_block` flags.

## Related Patterns

- `explicit-memory`: the alloc/subview machinery and the merge hazard.
- `cv-sync`: flag discipline if you hand-synchronize the prefetch.
- `workspace-pipeline`: cross-core ring that this per-core overlap complements.
