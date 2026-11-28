# Scaled MM：環境、算子優化與性能對照

本目錄包含改寫前後的 Python、獨立精度／性能腳本、2026-11-12 從本目錄實際執行的三方測試結果，以及四份代表性 MLIR。六組規格的三個實作共 **18 項精度檢查全部通過**；本次 TLA 相對未改寫 Triton 加速 **1.19–1.64×**，相對官方 AscendC 組合加速 **1.25–1.47×**。

**目前實測前端是 Triton Ascend TLA，不是 FlagTree。**FlagTree 的接口移植與同版驗證尚待進行；不可將本包數據標示為 FlagTree 的測量結果。套件包含算子與重現腳本，需要既有 Ascend 950／CANN／TLA 工具鏈；沒有內嵌 wheel 或完整編譯器。

## 1. 環境配置

### 1.1 FlagTree 與本次實測前端

FlagTree 工作倉位於 `/home/ricky/git/FlagTree`，整理時 HEAD 為 `7ba0ab72077287c7eabc94b2909cb0718e2ea7c0`。這只是當前工作區版本，本包未在該前端執行改寫版；一般 Triton 的基線可透過 `--methods original` 單獨測試，但不會因此得到 TLA 改寫版的 FlagTree 結論。

本次採用的前端：

| 項目 | 實測值 |
|---|---|
| Triton Ascend source | `/home/ricky/git/Triton-Ascend-tla` |
| 倉庫 | `https://gitcode.com/Ascend/triton-ascend.git` |
| 固定 commit | `008dff39bdcc908d6706946189b0c53c5257e4d4` |
| 介面來源 | TLA PR3103 checkout，包含 PR2982 的 L0 API；以固定 commit 為準 |
| Python／venv | Python3.12.16，`/home/ricky/venv/triton-ascend-tla` |
| triton | 3.6.0，從上述 source 的 `python/triton` 載入 |
| 依賴 | torch2.10.0+cpu、torch_npu2.10.0.post7 |

TA 安裝流程會套用倉內 `third_party/ascend/patch/triton-ascend-3.6.0.patch`。因此只看 HEAD 不足以描述整個 build；[manifest.json](manifest.json) 保留固定版本、已知工作區差異的 hash 及安裝 patch hash。沒有在本次打包期間改動前端／編譯器倉庫。

FlagTree 後續需要對接／驗證本例使用的 `tla.jit`、cube/vector/simd scope、L1/L0/UB buffer、`copy_l1_to_l0`、`mmad_l0`、ROW_SPLIT fixpipe 與 block set/wait。不能只替換 import 名稱便假定具備相同 lowering。

### 1.2 AscendNPU IR

| 項目 | 實測值 |
|---|---|
| source | `/home/ricky/git/AscendNPU-IR-tla` |
| 倉庫／commit | `https://gitcode.com/Ascend/AscendNPU-IR.git`，`cfe3feea3da5cbc603cb71221958cbf6e7bb9f95` |
| LLVM source | `/home/ricky/git/AscendNPU-IR/third-party/llvm-project`；精確 commit 見 manifest |
| build／工具 | `/home/ricky/git/AscendNPU-IR-tla-build/bin/bishengir-compile`、同目錄 `bishengir-opt` |
| 編譯執行方式 | `BISHENG_BACKEND=process` |
| CANN／bisheng | CANN9.2.1，`/usr/local/Ascend/cann-9.2.1/tools/bisheng_compiler/bin` |
| build 配置 | Release、Ninja、LLVM projects=mlir、external project=bishengir；完整選定參數見 manifest 的 `npuir_cmake_config` |

[npuir_build_compat.patch](npuir_build_compat.patch) 是已有的 CANN ABI／靜態庫連結相容修改，不是性能 pass 修改。此 patch 在已配置環境中已套用，勿重複套用。此前驗證的是 `bishengir-compile/opt` 的 process 路徑；不宣稱獨立 hivmc 全量 target 已建置成功。

如在原已配置 build 目錄增量構建，可用：

```bash
cmake --build /home/ricky/git/AscendNPU-IR-tla-build \
  --target bishengir-compile bishengir-opt -j 8
```

使用本包不需重新構建。移植到其他機器時，應按固定 source/LLVM/CANN 版本準備工具鏈，依其官方構建流程及 manifest 配置重建；本包沒有提供未驗證的一鍵安裝命令。

### 1.3 啟用環境與快速重現

原測試位置：`baai-950 / yuansheng_flagtree`，`/home/ricky/etc/example/20261112/ScaledMM`；裝置 `Ascend950DT_9582`，可見 NPU2。

在容器內執行：

```bash
cd /home/ricky/etc/example/20261112/ScaledMM
source ./run_env.sh
python benchmark.py
```

`run_env.sh` 僅啟用已有 venv/CANN，設定 source 路徑和編譯器，不執行安裝。它進入本腳本所在目錄，因此 ZIP 可解壓到另一個目錄；如工具鏈位置不同，先設定 `SCALEDMM_VENV`、`SCALEDMM_TA_ROOT`、`SCALEDMM_NPUIR_BIN`、`SCALEDMM_CANN_ROOT`。改裝置可用 `export SCALEDMM_DEVICE=3` 再 source。

腳本快取預設放在 `/home/ricky/.cache/scaledmm-20261112`，可用 `SCALEDMM_CACHE_DIR` 修改。效能測試會關閉 `TRITON_DEBUG`、`TRITON_ALWAYS_COMPILE` 等調試設定；`results.json` 同時記錄實際 Python、triton 載入位置與工具鏈環境。

常用命令：

```bash
# 六組規格，原版／TLA／官方組合，先驗精度再測完整性能。
# 使用另一個檔名，保留包內的參考結果。
python benchmark.py --output rerun_results.json

# 單一規格的精度檢查。
python benchmark.py --m 1024 --dtype bf16 --check-only --output check_results.json

# 只比較原版與改寫版。
python benchmark.py --methods original tla --output triton_comparison.json
```

## 2. Scaled MM

### 2.1 計算語義、規格與基線

A/B 為 FP16 或 BF16，matmul 以 FP32 累加，乘 FP32 逐行 scale A 及逐列 scale B，最後轉回輸入 dtype：

```text
Y[m,n] = cast_out(sum_k(A[m,k] * B[k,n]) * scale_a[m] * scale_b[n])
```

本包驗證 `M=32/128/1024, N=K=4096`、連續 A/B、FP32 row/column scales，無 bias。測試輸入為 `randn * 0.1`，scales 為 `rand + 0.5`；seed=`123+M`。這不是 FP8 scaled matmul。

| 名稱 | 包內實作／定義 |
|---|---|
| original | [scaled_mm_original.py](scaled_mm_original.py)：FlagGems-vllm `377620c132aea9569275f2bcea27f1aea17ff426` 的 `src/flaggems_vllm/runtime/backend/_ascend/ops/triton_scaled_mm.py`，逐 byte 保留 |
| tla | [scaled_mm_tla.py](scaled_mm_tla.py)：此前已驗證的三個 TLA kernel，逐 byte 複製 `scaled_mm_optimized.py`；公開入口 `scaled_mm(a,b,sa,sb,out_dtype)` |
| ascendc | `benchmark.py` 的 official 函數：`npu_fused_matmul(16cast32)`＋兩次 FP32 mul＋cast，共四個官方 kernel |

官方組合不是單一融合 Scaled MM。`torch._scaled_mm` 在此環境要求 FP8；不能轉成 FP8 當作同一個問題。原 Triton 的 scale 寫成 `acc*(sa*sb)`，TLA／官方組合為 `(acc*sa)*sb`；FP32 舍入順序及 K 累加順序可不同，所以比對容差，不聲稱 bitwise 一致。

### 2.2 按規模選擇 tile 與獨立入口

| M | 原版 BM×BN×BK | TLA BM×BN | TLA L1 K | TLA L0 K | L0A/B slot |
|---:|---|---|---:|---:|---:|
| 32 | 64×64×256 | 32×128 | 256 | 256 | 各1 |
| 128 | 64×128×128 | 128×128 | 256 | 256 | 各1 |
| 1024 | 128×128×128 | 256×256 | 128 | 64 | 各2 |

M32 避免 M64 的額外行，擴大 N tile；M128 增加 tile 內重用；M1024 使用較大輸出 tile，並以更細的 L0 K 切片保留雙 buffer 空間。三個規模有獨立 JIT 入口，persistent loop 每次跨32個輸出 tile 前進。收益來自配置與排程，不能歸因於函數名字或單純拆三份 Python。

### 2.3 L1／L0 分層切塊與雙 buffer

M1024 的 BM=BN=256。若 L0 K128，A/B 各需要64 KiB，單份已占滿對應 L0；下一塊搬運必須等原 buffer 釋放。L0 K64 時每份 A/B 各32 KiB，能各配置兩份，讓 MTE1 與前一份 MMAD 有重疊機會。L1 仍以 K128 搬入，再切成兩份 K64 消費。

[ir/tla_m1024.mlir](ir/tla_m1024.mlir) 可看到兩組 L0 地址、`l12l0` 及 `mmadL0` 的交替。這是顯式 slot 與具體搬運安排，不是只增大 `num_stages` 或 `versions`。小 M 的已測最佳配置保留單 slot，不能把雙 buffer 無條件套到全部規模。

### 2.4 融合 scale／cast 與 Cube–Vector 分工

結果在 L0C 保持 FP32；fixpipe 以 ROW_SPLIT 搬入兩個 AIV 的 UB，各處理一半行。SIMD scope 在同一 kernel 內完成逐行／逐列 scale、cast 與輸出，column scale load 在 row loop 外重用，scale DMA 提早發出。

相對官方四 kernel 組合，融合減少中間結果 GM 往返及額外 kernel。未改寫 Triton 本身也已融合 scale/cast，故相對 Triton 的提升主要應結合 tile、L0 排程與向量實作來看，不能套用相同解釋。

### 2.5 同步與 buffer 重用

GSS 處理核內 DMA/MMAD 依賴；算子以 flag0 通知 fixpipe 結果已可讀，以 flag2 通知 AIV 已讀完結果 UB。persistent loop 下一個 job 覆寫同一 UB 前，需要等釋放；算子最後也排空對應事件。

本次沒有刪除必要同步來取得加速。核內自動同步與明確跨核協議共同保證 buffer 生存期。正文只列整體改寫的測量結果；沒有獨立對照的微調，不拆出個別收益百分比。

### 2.6 本包精度與完整性能

下表為 **2026-11-12 使用包內 benchmark.py，在同一 process 完成的三方配對實測**。加速比=基線時間/TLA時間，大於1代表 TLA 較快。取代舊文檔中不同批次的時間拼接；原始兩輪數值、精度和環境均在 [results.json](results.json)。

| M | dtype | 未改寫 Triton µs | TLA µs | 官方 AscendC 組合 µs | TLA 對 Triton 加速 | TLA 對官方加速 | 精度 |
|---:|---|---:|---:|---:|---:|---:|---|
| 32 | FP16 | 25.11 | 15.34 | 20.71 | 1.64× | 1.35× | 全部通過 |
| 128 | FP16 | 25.69 | 21.58 | 30.30 | 1.19× | 1.40× | 全部通過 |
| 1024 | FP16 | 111.15 | 86.71 | 108.65 | 1.28× | 1.25× | 全部通過 |
| 32 | BF16 | 24.92 | 15.27 | 20.94 | 1.63× | 1.37× | 全部通過 |
| 128 | BF16 | 25.83 | 21.57 | 31.65 | 1.20× | 1.47× | 全部通過 |
| 1024 | BF16 | 111.16 | 86.77 | 108.64 | 1.28× | 1.25× | 全部通過 |

測量先完成 JIT/warmup，再以 NPU Graph 捕獲10次完整算子、每組重播10次，5組取中位數；正反順序各測一次，再取兩次中位數。計入相應全部裝置 kernel，不含 Python 提交、host 配置或編譯時間。TLA 精度檢查會預填 NaN，計時關閉此操作。

三個實作六組規格全部通過 `torch.testing.assert_close(atol=0.01, rtol=0.01)`；reference 是 FP32 matmul/scale 後轉成輸出 dtype。最大絕對誤差 FP16 為0.001953125，BF16 可達0.015625，仍符合絕對＋相對容差。不能把此結果描述成所有絕對誤差小於0.01，也不能外推為任意數值分佈的保證。

## 3. 補充

### 3.1 編譯器可行的優化點

| 方向 | 目前證據 | 可做的工作 |
|---|---|---|
| L1/L0 分層成本模型 | 大 M 的 K64 雙 slot 有收益，小 M 不一定 | 依 tile、L0 容量、任務數及 DMA/MMAD 粒度選擇，而非只增大 stage 數 |
| 流水物化與排程 | TLA 顯式兩份 buffer 才形成需要的搬運安排 | 讓 pipeline/stage 標記對應可驗證的 preload、slot 與依賴，而不只留下標記 |
| 尊重顯式 layout 的向量優化 | TLA Affinity pipeline 略過普通完整 AVE 優化 | 研究安全的 load hoist、combine、VF 調度；不能直接把普通 pipeline 全部打開 |
| 跨 job buffer ownership | 需保證 AIV 讀完 UB 才由下一次 fixpipe 覆寫 | 改善可分析的 buffer／event 合約，降低手寫同步的錯誤率 |

以上是候選方向，沒有在本包實作成新的編譯器優化。`npuir_build_compat.patch` 的相容修改不計入這些性能方向。

### 3.2 TLA／FlagTree 接口

本例實際取得的收益，現有測試版 TLA 已可表達。FlagTree 對接應優先確保同樣 L0 buffer、copy/MMAD、ROW_SPLIT、SIMD 與同步語義正確，再使用同一腳本配對驗證。當前 L0 route 對靜態16倍數、compact tile 等限制，要與後端 lowering 一起處理；不能只放寬前端檢查。

本例沒有需要動態稀疏 gather；Attention 的 SIMT scope／動態 DMA 需求不屬於本包的 Scaled MM 性能結論。

### 3.3 支援範圍與檔案

改寫版公開 wrapper 僅支援本文測試的 M32/128/1024、N=K4096、連續匹配 FP16/BF16 A/B/輸出及 FP32 row/column scales。任意尾塊、bias、scalar scale、非連續輸入、FP8、910不是已完成的交付驗證範圍。原版本身有較廣入口，三方對比限制在共同已測規格。

| 檔案 | 用途 |
|---|---|
| [scaled_mm_original.py](scaled_mm_original.py) | 改寫前的上游算子，保留原內容 |
| [scaled_mm_tla.py](scaled_mm_tla.py) | 改寫後的三個已測 kernel 與 wrapper，保留原內容 |
| [benchmark.py](benchmark.py) | 獨立精度／性能腳本；不依賴舊實驗目錄或 Attention harness |
| [run_env.sh](run_env.sh) | 啟用已有工具鏈及隔離快取，可覆寫路徑 |
| [results.json](results.json)、[benchmark.log](benchmark.log) | 本次六組三方實測數據與執行記錄 |
| [manifest.json](manifest.json) | 來源、commit、patch hash、build 配置、算子／IR hash |
| [npuir_build_compat.patch](npuir_build_compat.patch) | 已使用的構建相容 patch |
| [ir/original_m1024.mlir](ir/original_m1024.mlir) | 未改寫 Triton M1024，HIVM→Standard 前 |
| [ir/tla_m32.mlir](ir/tla_m32.mlir)、[ir/tla_m128.mlir](ir/tla_m128.mlir)、[ir/tla_m1024.mlir](ir/tla_m1024.mlir) | 三個最終 TLA kernel，同一 pass 邊界的 MLIR |
| [SHA256SUMS](SHA256SUMS) | 打包前檔案校驗；執行 `sha256sum -c SHA256SUMS` |

IR 是既有固定版本的 FP16 代表 dump，保留來源行號與絕對路徑，僅供分析，不是需要執行的路徑。`mmadL1` 是 macro，不能直接以 IR 行數與 `mmadL0` 比較指令數。完整 before-all 大型 log、`.o`、cache 不放進交付包。

如要在自己的環境 dump 單一 kernel，可在完成性能測試後使用：

```bash
TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 \
  python benchmark.py --methods tla --m 1024 --dtype fp16 --check-only \
  --output "$TRITON_CACHE_DIR/scaledmm_dump_check.json" \
  > "$TRITON_CACHE_DIR/scaledmm_driver.log" 2>&1
```

從 driver log 的 `bishengir-compile` 命令加上 `--mlir-print-ir-before-all --mlir-disable-threading` 可重放完整流水；把輸出 `.o` 留在 cache。調試編譯與正式性能測量分開進行。
