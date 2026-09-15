Read this file first. Then read only the most relevant detailed pattern cards.

## High Priority Patterns

### Tile Size Selection
**Summary**: Choose BLOCK_M, BLOCK_N, BLOCK_K values that fit in L1 and keep Cube utilization high. Starting points: BLOCK_M=128, BLOCK_N=256, BLOCK_K=128 for float16 matmul. Verify UB/L1 budget before running.
**Source**: [patterns/tile-sizing.md](patterns/tile-sizing.md)

### Autotune
**Summary**: Use autotune to search block sizes and pipeline stage counts when the kernel structure is already correct. Default to a direct config list; generate programmatically when the space is large.
**Source**: [patterns/autotune.md](patterns/autotune.md)

---

## Generated Pattern Summaries

### Tile Size Selection
**Priority**: high
**Source**: [patterns/tile-sizing.md](patterns/tile-sizing.md)

**Use When**:
- Writing a new matmul or attention kernel and need starting tile sizes.
- The kernel runs but is slow, and profiling shows low Cube utilization or high DMA stall time.
- Changing BLOCK_K doesn't improve throughput — the tile may be too small to fill the Cube pipeline.

**Avoid When**:
- The kernel is not yet correct — fix correctness first.
- Tile sizes are already constrained by a semantic requirement (e.g., BLOCK_N must equal the head dimension).

---

### Autotune
**Priority**: high
**Source**: [patterns/autotune.md](patterns/autotune.md)

**Use When**:
- The kernel structure is correct and the open question is parameter choice (block sizes, K tile, pipeline stages).
- Manual tuning has plateaued and you need to search more systematically.
- The factory function exposes free tuning parameters.

**Avoid When**:
- The kernel has a structural problem (wrong algorithm, wrong memory hierarchy).
- All relevant parameters are already fixed at module level.
- The kernel is not yet correct.

---

### compile_hint Usage
**Priority**: medium
**Source**: [patterns/compile-hints.md](patterns/compile-hints.md)

**Use When**:
- UB usage is unexpectedly large for small tensors (K=1 or BLOCK_K=1 scalar access pattern).
- The compiler is hoisting Vector ops out of a loop incorrectly (bubble-up symptom).
- A `tl.where` condition uses i1 tensors and UB usage is higher than expected.
- A multi-matmul kernel (e.g., FlashAttention) overflows L1 at moderate tile sizes.

**Avoid When**:
- The kernel has no performance problem — hints add coupling to compiler internals.
- The symptom is a correctness bug — hints don't fix logic errors.
