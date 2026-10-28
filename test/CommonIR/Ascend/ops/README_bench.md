# bench_ops.py — standalone latency benchmark for the CommonIR Ascend ops

No reference implementation needed: it measures each of the six operators on its
own.  It exists so that a kernel change can be compared **before/after** on the
same box, with the same inputs, and with a check that the numbers did not move.

```bash
cd test/CommonIR/Ascend/ops

python bench_ops.py list        # case registry
python bench_ops.py selfcheck   # build every case on CPU (no device needed)
python bench_ops.py run  --label before --outdir out/before
# ...apply the change...
python bench_ops.py run  --label after  --outdir out/after
python bench_ops.py compare --before out/before --after out/after
```

`compare` writes `out/compare.md` (plus a sibling `compare.json`) with per-case
before/after milliseconds, the factor, the numerical digest, and a per-op
geomean.

## No git checkout needed for this port

The port is entirely module-level constants plus gates whose conditions read the
operator's *inputs*, so the pre-port configuration is reproducible on the
current tree by overriding five constants:

| constant | pre-port value | what it undoes |
|---|---|---|
| `_PROJ_BN`, `_PROJ_BK` | 128 | `compressor` projection tiling 256/256 |
| `_BLOCK_M_NARROW`, `_BLOCK_N_NARROW` | 128 | `gqa`'s 64x64 tile for causal / D>64 |
| `_HEAD_BLOCK_CMP` | 32 | `sparse_attn_sharedkv`'s 16-head compressed block |

`PRE_PORT_CONSTS` in this file holds them, so:

```bash
# one command, no git, no mutation of the tree
python bench_ops.py ab --before-consts --cases full --repeat 3 --root out/ab

# or one state at a time
python bench_ops.py run --before-consts --label before --outdir out/before
python bench_ops.py run                  --label after  --outdir out/after
python bench_ops.py compare --before out/before --after out/after
```

An operator that does not have a given constant ignores it and records it in
`results.json` under `consts_ignored` (fatal with `--strict-const`), so one const
list can be shared by all six operators without a typo passing silently.

Use `--before-ref <merge-commit>` instead only when the change is structural
rather than a constant, or to cross-check that `PRE_PORT_CONSTS` really does
reproduce the old code path.

## Reproducing a before/after comparison in one command

`ab` does the whole state-switch dance: it checks the operator files out of
`--before-ref`, runs the complete benchmark, checks them out of `--after-ref`,
runs it again, compares, and **always restores the working tree** (from an
in-memory snapshot, so a dirty tree survives).  `bench_ops.py` itself is never
swapped, so both sides are measured with the same tool, the same case set and
the same digests:

```bash
python bench_ops.py ab --before-ref 01af67551 --after-ref HEAD \
       --cases default --repeat 2 --root out/ab
# -> out/ab/compare.md        the ratio table + per-op geomean
# -> out/ab/ab_meta.json      both SHAs, their subjects, files swapped
# -> out/ab/{before,after}/   full results.json + summary.md + raw/ + logs/
```

Use `--cases full` for the record-worthy run.  Without `ab`, the same thing is
three commands (`run` with the tree in state A, `run` in state B, `compare`) —
`ab` exists so that a mis-ordered checkout cannot silently compare a state
against itself: it resolves both refs, refuses if they are the same commit, and
reports per state how many operator files actually differed.

## Timing modes

`--measure-mode` (default `npu-graph`):

| mode | what it measures |
|---|---|
| `loop` | wall clock of `run(**kwargs)` — what a caller sees, host work included |
| `npu-graph` | capture once, time `graph.replay()` — **device only**. Ops that synchronise the host inside `run()` cannot be captured; they report `graph_capture_failed` and fall back to `loop` |
| `host-split` | wall clock, plus the same call with every Triton launch stubbed out; `device_ms = total - host`, where host includes any device→host sync |

`npu-graph` and `host-split` answer different questions and their numbers are not
comparable to each other. In particular the ones that do `.cpu().tolist()` inside
`run()` (`causal_conv1d_fn`) cannot be captured at all — use `host-split` there.

Note that `graph.replay()` has a floor of roughly 60 µs on Ascend910B4, so any
sub-0.1 ms graph number is floor-dominated; use `loop`/`host-split` for those.

## Between-process noise (`--repeat`)

Within-run rounds are tight, but two runs minutes apart can disagree — a
before/after pair on `sparse_attn_sharedkv/swa_prefill_small` moved 12% with the
kernel configuration, the compilation cache entry and the output digest all
provably unchanged. That is drift between process invocations, not a regression,
and a single run cannot show it.

`--repeat N` re-runs every case N times in fresh processes; `results.json` then
holds the median across repeats plus `repeat_ms`, `repeat_min/max`,
`spread_pct` and `digest_identical_across_repeats`, and both `summary.md` and
`compare.md` show the spread next to each timing:

```bash
python bench_ops.py run --outdir out/after --label after --repeat 3
python bench_ops.py compare --before out/before --after out/after
```

`spread_pct` is necessary but **not sufficient**: it measures variation *within*
one run, while the drift that matters is *between* the before-run and the
after-run. Two reference points from three repeated before/after pairs on
Ascend910B4:

| case | same-kernel measurements across runs |
|---|---|
| `causal_conv1d_fn/*` (ms scale) | 2.108, 2.096, 2.118 ms → between-run floor **<1%** |
| `kv_rmsnorm_rope_cache/*`, `lightning_indexer/R*` (~60 µs) | geomean 0.95x, 1.02x, 0.97x → **±5%** |
| `sparse_attn_sharedkv/swa_prefill_small` | 0.517, 0.521, 0.554, 0.566, 0.589, 0.633 ms → **±20%**, unusable |

So: check the delta against the *spread for that specific case*, and treat a
change as real only if it is several times larger **and** reproduces across
runs. `compressor/prefill0_b2_rank3` at 1.44x with 1.8% spread, reproduced as
1.47x / 1.42x / 1.44x, is about 25x its spread; `sparse_attn_sharedkv/swa_decode`
at 1.26x with a 19.3% spread on one side is not a result at all.

## Numerical guard (no reference required)

Every run records an fp64 digest per output tensor — `sum`, `sumsq`, `absmax`,
`numel`, non-finite count — where "output" means the return value *and* whatever
the op mutated in place (`conv_states`, `state_cache`, the paged caches at the
touched rows).  `compare` reports `identical` or the relative delta next to the
timing.  A tile-size change legitimately perturbs the fp32 reduction order, so
this is reported as information, **not** as pass/fail; a change of order 1e-7 is
reduction order, a change of order 1e-2 is a bug.

## Isolation and knobs

* one subprocess per `(op, case)`: a kernel that faults the device
  (`aivec error`), hangs (`--timeout`), or fails to compile only costs that one
  row, and the traceback lands in `out/logs/`.
* `--const NAME=VALUE[,NAME=VALUE]` overrides module-level constants before the
  call, for ad-hoc A/B on one knob without editing the kernel, e.g.

  ```bash
  python bench_ops.py run --label t64 --outdir out/t64 --ops causal_conv1d_fn \
         --const _BLOCK_T=64
  python bench_ops.py compare --before out/before --after out/t64
  ```

  The overrides are echoed into `meta.json` and into every raw record, so a row
  can never be mistaken for the committed defaults.

## Output layout

```
<outdir>/
  meta.json                 label, ops, cases, mode, constants, warmup/iters/rounds
  results.json              one record per (op, case) incl. digest + timings
  summary.md                the table for this run
  raw/<op>__<case>@<label>.json   full record (samples, peak alloc, notes)
  logs/<op>__<case>@<label>.log   worker stdout/stderr (tracebacks)
```

## Case sets

`--cases quick|default|full` or an explicit comma-separated list of case ids.
The shapes are the ones that showed measurable differences while these operators
were compared against their non-CommonIR counterparts, so numbers here stay
comparable with that work. `full` adds the heavier production shapes
(`compressor/prefill0`, `causal_conv1d_fn/b10_d4096_s4096_pad_fp16`,
`sparse_attn_sharedkv/*_prefill`, …) which are the ones worth quoting when
comparing kernels, because the small shapes are dominated by fixed costs.
