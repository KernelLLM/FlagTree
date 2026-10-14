# lightning_indexer

## 1. 测试方法

- 在海光bw1000上进行测试。共三轮独立进程，每轮每版 20 次预热、50 个有效 GPU-event 样本。
- 使用 HIP graph，每窗口一次 kernel；编译、分配、参考计算、graph capture 及重置不计时。
- 下表合并每版全部 150 个样本；时间单位为 μs，加速比为 TLE P50 / Gluon 优化版 P50。

## 2. 改变了哪些 op 的 layout

**算子顺序：QK 矩阵乘 → FP16 舍入 → ReLU → head weight 乘法 → Head 求和 → FP16 舍入 → FP32 logits 写回。**

**QK 矩阵乘的 Q 读取布局**

- Gluon 基线：`#ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>`。
- Gluon 优化版：`#ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [1, 0]}>`。

**QK 矩阵乘及后续 ReLU/加权计算的 MFMA 布局**

- Gluon 基线：`#ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>`。
- Gluon 优化版：`#ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = false, mmacLayout = 1}>`。

**Head 求和的输入布局**（输入按 `[K tile,Q tile,head]=[64,1,16]` 组织，沿 head 求和）

- Gluon 基线：`#ttg.linear<{register = [[4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]], warp = [[0, 0, 0], [0, 0, 0]], block = []}>`。
- Gluon 优化版：`#ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>`。

归约结果分别是对应输入 layout 的 dim=2 slice；优化在 Head 求和前后显式转换布局，最终 logits 写回布局不变。两版均 `num_warps=4 / num_stages=3 / waves_per_eu=1`，H=16、D=32。下表为 Q tokens×K tokens，输入 BF16。

## 3. P50

| Q tokens×K tokens | TLE P50 | Gluon 基线 P50 | Gluon 优化版（optimized）P50 | 相对 TLE 加速比 |
| --- | ---: | ---: | ---: | ---: |
| 1024×2048 | 517.520 | 520.160 | 209.280 | 2.4729× |
| 1024×4096 | 990.880 | 994.000 | 381.600 | 2.5966× |
| 2048×4096 | 1896.960 | 1896.160 | 671.280 | 2.8259× |
| 4096×8192 | 10905.840 | 10928.400 | 3575.040 | 3.0506× |

## 4. P20–P80

| Q tokens×K tokens | TLE P20–P80 | Gluon 基线 P20–P80 | Gluon 优化版（optimized）P20–P80 |
| --- | ---: | ---: | ---: |
| 1024×2048 | 512.641–526.304 | 515.168–527.232 | 208.288–210.272 |
| 1024×4096 | 986.208–995.680 | 990.400–998.880 | 377.344–387.552 |
| 2048×4096 | 1892.640–1902.080 | 1891.200–1900.961 | 664.127–676.480 |
| 4096×8192 | 10900.321–10912.192 | 10922.817–10934.752 | 3571.648–3578.720 |
