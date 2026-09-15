Read this file first. Then read only the most relevant detailed pattern files based on your kernel's engine structure.

## High Priority Patterns

### Double-Buffered Matmul
**Summary**: Ping-pong L1 buffers for A and B tiles to overlap DMA with Cube MMA. Requires `tl.debug_barrier()` between prologue copy and first MMA. Uses `tile_cube_launch` + `tile_cube_wait`.
**Source**: [patterns/double-buffer-matmul.md](patterns/double-buffer-matmul.md)

### Three-Task Pipeline
**Summary**: Full DMA → Cube → Vector ring pipeline using `tile_set_flag`/`tile_wait_flag` with distinct event IDs per handshake. Required when Cube output must feed Vector postprocessing at peak throughput.
**Source**: [patterns/three-task-pipeline.md](patterns/three-task-pipeline.md)

---

## Generated Pattern Summaries

### Vec Add (UB Only)
**Summary**: Baseline single-engine vector kernel. Allocates UB buffers, copies GM→UB, calls `tle_dsa.add`, copies UB→GM. No Cube, no L1, no flags needed.
**Priority**: medium
**Source**: [patterns/vec-add.md](patterns/vec-add.md)

**Use When**:
- The operation is purely element-wise (add, sub, mul, div, max, min, activation).
- No matrix multiply is needed.
- The working set fits in UB (Unified Buffer).

**Avoid When**:
- The kernel includes any matrix multiply — use Double-Buffered Matmul instead.
- The output of a Cube MMA must be post-processed — use Three-Task Pipeline instead.

---

### Double-Buffered Matmul
**Summary**: Two L1 buffer pairs for A and B enable DMA prefetch to overlap with Cube MMA. L0A/L0B/L0C used for MMA operands and accumulator. `tile_cube_wait` after each MMA before accessing L0C.
**Priority**: high
**Source**: [patterns/double-buffer-matmul.md](patterns/double-buffer-matmul.md)

**Use When**:
- The kernel performs matrix multiplication (GEMM, batched GEMM, attention QK or PV).
- The K dimension is large enough that DMA latency is a bottleneck (typically K ≥ 128).
- The output of MMA does not need Vector engine postprocessing (or postprocessing is done in a separate kernel).

**Avoid When**:
- K is small (< 64); a single-buffer L1 copy + MMA may be simpler.
- The MMA result must feed Vector ops in the same kernel — use Three-Task Pipeline instead.

---

### Three-Task Pipeline
**Summary**: Producer-consumer ring: MTE2 fills L1 buffers, Cube reads L0A/B and writes L0C, Vector reads L0C (via UB copy) and writes UB/GM. `tile_set_flag`/`tile_wait_flag` pairs coordinate each handshake. Each engine runs concurrently after prologue.
**Priority**: high
**Source**: [patterns/three-task-pipeline.md](patterns/three-task-pipeline.md)

**Use When**:
- The kernel requires Cube MMA followed by Vector postprocessing (e.g., matmul + bias + activation, flash-attention with softmax).
- Peak throughput requires all three engines (DMA, Cube, Vector) to run concurrently.
- The operation has enough K-loop iterations (≥ 4) to amortize pipeline fill/drain overhead.

**Avoid When**:
- The kernel is purely element-wise — use Vec Add pattern instead.
- The kernel does GEMM without Vector postprocessing — use Double-Buffered Matmul instead.
- The K dimension is very small (< 64); pipeline overhead dominates.
