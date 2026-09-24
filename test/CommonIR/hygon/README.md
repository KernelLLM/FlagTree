# 算子性能对比

下面是海光bw1000实测。每行取该算子已测 case 的 **GPU 延迟中位数**，再对各 case 等权取几何平均；单位 μs

| 算子 | case 数 | TLE | Gluon 基线 | 优化 Gluon | 优化 / TLE |
|---|---:|---:|---:|---:|---:|
| `add_rms_norm` | 28 | 3.93 | 3.94 | 3.69 | 0.938× |
| `dequantize_and_gather_k_cache` | 5 | 132.84 | 114.58 | 113.58 | 0.855× |
| `fp8_fp4_paged_mqa_logits` | 7 | 173.08 | 173.14 | 76.92 | 0.444× |
| `fused_deepseek_v4_qnorm_rope_kv_rope_insert` | 7 | 52.38 | 52.38 | 40.01 | 0.764× |
| `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | 22 | 538.33 | 538.53 | 502.80 | 0.934× |
| `fused_inv_rope_fp8_quant` | 28 | 10.34 | 10.30 | 9.82 | 0.950× |
| `group_mm` | 9 | 6009.21 | 5017.40 | 3821.16 | 0.636× |
| `hc_head_fused_kernel` | 16 | 34.00 | 33.94 | 27.25 | 0.801× |
| `hc_split_sinkhorn_forward` | 8 | 56.20 | 56.20 | 51.12 | 0.910× |
| `lightning_indexer` | 4 | 1713.93 | 885.98 | 666.25 | 0.389× |
| `megamoe` | 6 | 20229.19 | 20229.24 | 12536.23 | 0.620× |
| `mhc_bwd` | 4 | 20.00 | 20.00 | 13.33 | 0.666× |
| `mhc_post` | 6 | 179.90 | 179.97 | 169.34 | 0.941× |
| `mhc_pre` | 12 | 145.02 | 144.70 | 91.34 | 0.630× |
| `moe_sum` | 18 | 2.73 | 2.74 | 2.15 | 0.791× |
| `mv` | 9 | 7.69 | 7.69 | 7.58 | 0.986× |
| `rms_norm` | 40 | 4.17 | 4.19 | 3.77 | 0.902× |
| `rms_norm_w8a16_fp8` | 6 | 9.36 | 11.27 | 8.78 | 0.938× |
| `router_gemm` | 9 | 61.93 | 66.57 | 51.10 | 0.825× |
| `rwkv_ka_fusion` | 6 | 160.84 | 141.55 | 135.44 | 0.842× |

注：`router_gemm` 仅统计 MM kernel，不含原始 TLE 的 weight transpose；`mhc_pre` 不含公共的 `torch.mm`；`megamoe` 统计 L1+L2 双 kernel 链。`fused_inv_rope_fp8_quant` 使用归档的 28-case fresh 测量值（其中一个短 kernel 的波动另有专项复核）。不同算子的输入及计时范围不同，不宜横向比较绝对延迟。
