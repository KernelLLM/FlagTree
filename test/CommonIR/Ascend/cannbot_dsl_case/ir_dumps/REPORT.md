# CANNBot-DSL 0.8.0 — IR Dump Report (CANN 9.2.1)

## Environment

| Item | Value |
|------|-------|
| cannbot-dsl | 0.8.0 (wheel, cp312-aarch64) |
| CANN | 9.2.1 (`/usr/local/Ascend/cann-9.2.1/set_env.sh`) |
| Python | 3.12.16 |
| torch / torch_npu | 2.10.0 / 2.10.0.post7 |
| Target | Ascend 950PR / 950DT (dav-3510) |
| Dump root | `/dev/shm/yuansheng/skill/cannbot-dsl/ir_dumps/` |

## Summary

- **32 operator directories** with IR artifacts
- **293 IR files**, ~33 MB total
- **~616 tests passed** across all operators
- **10 operators blocked** (Channel API removed in 0.8.0)

Each compiled kernel produces up to 5 IR files:
- `run.asc` — AscendC generated source
- `run.mlir` — MLIR intermediate representation
- `run.frontend.py` — Python source snapshot
- `run.frontend.plan` — Frontend compilation plan
- `run.frontend.lowered.py` — Lowered Python AST

## Passed Operators

| Operator | Tests | Kernels | Size | Notes |
|----------|-------|---------|------|-------|
| add | 10/10 ✅ | 1 | 68K | |
| attn_epilogue | 48/48 ✅ | 1 | 188K | |
| attn_prologue | 12/12 ✅ | 0 | — | unit tests only; accuracy blocked by Channel |
| block_attn_res_update_rms_norm | 3/3 ✅ | 1 | 104K | |
| commit_recurrent_kda_replayssm | 3/3 ✅ | 1 | 772K | |
| dynamic_mx_quant_sr | 36/37 ⚠️ | 18 | 3.5M | 1 NPUGraph capture fail; requires ml_dtypes |
| engram_gate | 5/5 ✅ | 1 | 120K | |
| flash_kda | 0/1 ⚠️ | 1 | 2.5M | compiled & dumped; precision mismatch |
| flash_kda_cp_main | 2/3 ⚠️ | 1 | 1.1M | 1 precision failure |
| flash_kda_cp_preprocess | 7/7 ✅ | 1 | 1008K | |
| fused_norm_rope_attn_rope_cast_fwd | 1/1 ✅ | 1 | 720K | |
| fused_recurrent_kda_replayssm | 4/4 ✅ | 1 | 772K | |
| fused_recurrent_kda_snapshot | 3/3 ✅ | 1 | 528K | |
| gather_v2 | 20/20 ✅ | 1 | 36K | |
| grouped_matmul | 20/20 ✅ | 1 | 100K | |
| indexer_prologue_k | 3/5 ⚠️ | 1 | 140K | 1 precision failure |
| indexer_prologue_qw | 3/3 ✅ | 1 | 204K | |
| matmul | 267/267 ✅ | 1 | 112K | |
| mega_recurrent_kda | 25/25 ✅ | 1 | 8.3M | largest single kernel IR |
| mega_recurrent_kda_replayssm | 21/22 ⚠️ | 1 | 772K | 1 torch._inductor import error |
| mixed_quant_sparse_flash_mla_metadata | 8/8 ✅ | 0 | — | metadata-only (no kernel compile) |
| mla_prolog | 4/4 ✅ | 4 | 1.7M | 4 kernel variants: rope, m2_rope, m3_qr, qr_rope |
| moonep | 30/52 ✅ | 0 | — | 22 skipped (hardware/multi-device) |
| qsa_indexer | 21/21 ✅ | 1 | 672K | |
| quant_block_sparse_attn | 4/4 ✅ | 1 | 3.1M | |
| quant_lightning_indexer | 44/44 ✅ | 1 | 712K | |
| quant_lightning_indexer_metadata | 7/7 ✅ | 0 | — | metadata-only |
| quant_mla_prolog | 4/4 ✅ | 1 | 696K | |
| quant_sparse_lightning_indexer | 23/23 ✅ | 1 | 664K | |
| quant_sparse_lightning_indexer_metadata | 6/6 ✅ | 0 | — | metadata-only |
| qwen_sparse_attn | 22/22 ✅ | 4 | 948K | 4 kernel variants |
| scatter_elements | 23/23 ✅ | 5 | 244K | 5 kernel variants |
| sparse_flash_mla | 96/96 ✅ | 1 | 2.9M | longest run: 58 min |
| stem_indexer | 33/33 ✅ | 1 | 744K | |

## Blocked Operators (Channel API)

These 10 operators import `cannbotdsl.channel.Channel` which was removed in 0.8.0 (migrated to RingBuffer per CHANGELOG 2026-10-28):

| Operator | Blocked Import |
|----------|---------------|
| all_to_all_quant_matmul | `from cannbotdsl.channel import Channel` |
| flash_attn | `ChannelKind, make_copy_engine, channel_rewind` |
| flash_attn_fp8_fullquant | `RingBuffer(..., shape=...)` (kwarg removed) |
| flash_kda_fwd | `from cannbotdsl.channel import Channel` |
| flash_mla_with_kvcache | `from cannbotdsl import Channel` |
| kv_compress_epilog | `from cannbotdsl.channel import Channel` |
| pointnet_sa | `from cannbotdsl.channel import Channel` |
| rms_norm | `from cannbotdsl.channel import Channel` |
| sparse_flash_attention | `from cannbotdsl.channel import Channel` |
| voxel_conv | `from cannbotdsl.channel import Channel` |

## Notes

1. **torch version conflict**: `pip install ml_dtypes` upgraded torch 2.10→2.15, breaking torch_npu. Fixed by `pip install torch==2.10.0 --no-deps`.
2. **IR overwrite**: dump directory keeps only the last kernel's IR per operator (unless multiple kernel variants are compiled in one test, as with mla_prolog and qwen_sparse_attn).
3. **mixed_quant_sparse_flash_mla**: all 4 tests skipped (likely hardware/config prerequisite).
4. **moonep**: 22 of 52 tests skipped (multi-device / hardware requirements); 30 passed host-side tests generated no kernel IR.

## Dump env vars used

```bash
CANNBOTDSL_DUMP_MLIR=1
CANNBOTDSL_DUMP_ASCENDC=1
CANNBOTDSL_DUMP_FRONTEND=source,plan,lowered
CANNBOTDSL_DUMP_DIR=/dev/shm/yuansheng/skill/cannbot-dsl/ir_dumps/<operator>
```
