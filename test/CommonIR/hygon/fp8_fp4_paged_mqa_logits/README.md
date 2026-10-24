# fp8_fp4_paged_mqa_logits

## 1. 测试方法

- 在海光bw1000上进行测试。共三轮独立进程，每轮每版 20 次预热、50 个有效 GPU-event 样本。

## 2. 改变了哪些 op 的 layout

**算子顺序：Q/K 解量化 → QK 矩阵乘 → KV scale 乘法 → ReLU → head weight 乘法 → Head 求和 → logits 写回。**

改变了两个部分：

1. **QK 矩阵乘、ReLU 加权和 Head 求和的二维数据布局**：这些 op 共用 MFMA accumulator layout。
   - Gluon 基线：`#ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>`。
   - Gluon 优化版：`#ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>`。
   - head weight 向量使用该 MFMA layout 的 dim=1 slice；KV scale 向量及 Head 求和结果使用 dim=0 slice，随 MFMA parent 一起改变。
2. **logits 写回的布局**：
   - Gluon 基线：`#ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>`。
   - Gluon 优化版：归约结果直接使用 `#ttg.slice`，dim=0，parent 为上面的优化版 MFMA layout，不再转为一维 blocked 后写回。

Q/K 原始字节读取和 Q shared staging 布局不变。两版均 `num_warps=4 / num_stages=2`。下表配置为 rows×context，Q 为 FP8 E4M3。

## 3. P50

| rows×context | TLE P50 | Gluon 基线 P50 | Gluon 优化版（optimized）P50 | 相对 TLE 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 1×128 | 135.120 | 146.960 | 138.000 | 0.9791× |
| 4×512 | 176.800 | 189.760 | 151.680 | 1.1656× |
| 16×2048 | 206.560 | 219.760 | 159.360 | 1.2962× |
| 32×4096 | 446.800 | 459.200 | 247.360 | 1.8063× |
| 8×8192 | 344.640 | 364.400 | 214.640 | 1.6057× |
| 4×16384 | 545.440 | 563.920 | 270.080 | 2.0196× |
| 2×32768 | 543.601 | 558.402 | 271.840 | 1.9997× |

## 4. P20–P80

| rows×context | TLE P20–P80 | Gluon 基线 P20–P80 | Gluon 优化版（optimized）P20–P80 |
| --- | ---: | ---: | ---: |
| 1×128 | 121.760–157.888 | 135.488–170.944 | 123.328–148.512 |
| 4×512 | 174.400–179.680 | 187.967–191.200 | 149.920–153.600 |
| 16×2048 | 204.320–208.160 | 217.600–222.400 | 157.280–162.080 |
| 32×4096 | 440.448–453.471 | 454.240–467.360 | 242.528–253.952 |
| 8×8192 | 336.128–348.479 | 359.808–399.424 | 205.023–218.272 |
| 4×16384 | 541.856–552.640 | 556.160–672.000 | 267.200–277.664 |
| 2×32768 | 540.483–547.361 | 554.691–562.305 | 268.640–276.833 |
