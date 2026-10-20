**可獨立平行提交的訓練方法實驗**

2026-10-19，Asia/Taipei。這份方案替代前一版「先校準、再逐級晉級」的執行依賴：僅用 seed42；8 個新方法作業，加上1個沿用既有結果的歷史控制。每個 job 使用固定設定，自己完成數值前置檢查、隨機初始化、訓練與終點評估；不依賴另一個新實驗的產物。K1 使用的是已存在且固定校驗碼的教師。

本次只準備程式與設定，**沒有 sbatch 提交、GPU 預留或 GPU 執行結果**。完整模型的 H200 相容性、數值精度、記憶體與速度仍待作業內驗證；靜態測試通過不等同訓練成功。

**方法與對照**

| ID | 實驗方法 | 預先指定比較 | 驗證的問題 |
|---|---|---|---|
| N0 | 沿用 `ppl-next-20261017-B-seed42` 的 NorMuon + AdamW×2 結果 | seed42 歷史品質控制 | 不新增控制組訓練作業 |
| D1 | 官方 Dion3，fraction=1，NS5 | D1−N0 | 官方更新實作、動量與尺度移植差異 |
| D025 | 同一 Dion3，fraction=.25，LR 為 D1 的2倍 | D025−D1；對 N0 作次要比較 | 部分更新＋誤差回饋與指定尺度補償的效果 |
| A1 | AdaMuon v3，sign→NS5→逐元素二階矩 | A1−N0 | 列內自適應是否值得額外狀態成本 |
| S1 | 完整 SOAP，basis refresh 每10步 | S1−N0 | 在學得的旋轉座標中更新是否有價值 |
| K1 | .75 hard CE + .25 KL(teacher‖student)，T=1 | K1−N0 | 已有教師能否改善學生的 token 效率 |
| F1 | SF-NorMuon + SF-AdamW；平均權重回饋梯度位置 | F1−F2 | averaging feedback 本身是否改善學習 |
| F2 | 與 F1 相同內部更新，只在評估使用平均權重 | F2−N0 | 固定 LR＋平均與原 scheduled recipe 的差異 |
| C1 | `cce_exact`；與 N0 相同優化器、排程 | C1−N0 | 精確 loss kernel 是否降低時間與記憶體成本 |

所有新方法只用 seed42；N0為相同seed、相同訓練設定的歷史對照。D1 是移植對照，不預設它與 N0 數值等價。D025 同時含預先指定的 fraction 尺度補償，不能把差異全歸因於 fraction 單一因素。F2 與 N0 的差異亦包含內部更新規則，不能稱為只改 schedule 的 ablation；只有 F1−F2 用來辨別回饋。

所有方法在啟動前固定，不根據其他新 job 的結果選 LR、教師、checkpoint 或方法。因此可以同時啟動；代價是沒有逐方法 LR 校準。負面結果只否定這個固定 recipe，不足以否定整個 optimizer 家族。若之後要重新校準，必須成為另行記帳的新研究，不能覆寫本組。

**沿用控制組的核對結果**

N0使用 `ppl-next-20261017-B-seed42`：seed42、隨機residual-half、原intradoc遮罩、相同optimizer/LR/WD/batch/token排程，1,000,079,360 inputs／4775 updates均相符。設定差異僅study識別、arm名稱、parent與新adapter metadata；核心optimizer、attention、初始化、dataset及requirements來源檔一致。固定1k PPL=18.6994635，額外9k PPL=18.9391399，token加權合併10k PPL=**18.9155962**。9k逐文件NLL收據仍存在，可直接做配對分析，不需要重訓或重新評估控制組。

`reference-control.json`封存歷史設定、初始化、split/eval identity、環境、summary、9k結果和逐文件收據的SHA256；提交前與結果彙整時驗證。N0 config只保留為recipe／數值gate參照，不是待提交的job。歷史訓練10.7266667 H200-hours及另次共享confirmation allocation完整保留在原ledger，不再計一次新成本。新方法不讀取N0權重來訓練。

這是可用於方法初篩的歷史品質對照，未宣稱同期完全受控實驗。每個新job會先做舊native update對新wrapper的小模型數值等價gate；GPU gate仍未執行。舊成本、速度僅列為歷史參考，新增preflight、logging與cache／節點負載可能改變端到端時間。若後續要求嚴格的系統效率結論，再單獨設計短paired benchmark；本次不增加控制組重訓。額外9k早已用於歷史研究，屬共用validation，不能當作未看過的新盲測。

**共同訓練契約**

沿用已驗證的 1B 實驗幾何：Llama 3.2 1B 架構、原 tied embedding/tokenizer、8192 context、隨機 residual-half 初始化。保留 canonical seed42 document split、原 50k 保留文件與 raw training 順序。模型 seed 固定42，不改資料 split。所有學生均不讀取訓練 checkpoint。

每組終點是 **1,000,079,360 nonpadding student inputs、4,775 個更新窗口**。global BS16 在500,039,680 inputs 切換為64；完整保留為對齊 token knots 所需的 BS16/32/48 邊界窗口。每個 job 為8 H200、8 CPU總數、單一 Slurm launcher，8 ranks、每 rank 1 thread。FP32參數、BF16 autocast、FA3、activation checkpointing、global gradient clipping .5；資料與模型資產沿用現有環境，不安裝新依賴。

N0/D1/D025/A1/S1/K1/C1 使用相同 token-clock 兩階段 WSD：phase split833,355,776 inputs，第二峰值.1×；每 phase warmup170 BS16等效單位、40% tail、4:1:1 ramp/hold/final、.2 plateau。F1/F2 共用單次170單位 warmup，之後常數 LR；無第二次 warmup、restart 或 cooldown。

N0 matrix peak LR=.008614891542046706，auxiliary peak LR=.0005168934925228023。每個 trainable tensor 恰好分派一次，shared embedding/output 與 norms 走 auxiliary，hidden Linear weights 走 matrix optimizer。以 input dimension `n` 作固定名義 RMS 映射：D1/A1/F1/F2 matrix LR=`base/(.2√n)`，D025再乘2，SOAP=`base/√n`。這只是初始尺度匹配，並非已量測的最佳 LR。

hidden matrix decay 統一由外層施加 `1−.1×reference_matrix_LR(t)`，關閉 adapter 內部 decay。SF 在 fast weights z 上 decay；auxiliary 無 decay。SOAP 的第一步只建立 basis，仍計入全部已處理 inputs，且外層 decay 仍照定義執行。正式結果應讀取 `train/actual_decay_factor_*` 與實際 group LR，不能從名義 config 推算為作者原始 recipe。

**實作與來源界線**

[Dion3](https://arxiv.org/html/2608.11612v1)使用[官方套件](https://github.com/microsoft/dion/tree/7692479288f928a3d61d1cc58137b4dbe4f71000)，固定 commit、保留 license；`distributed_mesh=None`，完整 replicated DDP 狀態，自訂同一 NS5。這裡研究更新方法，不測完整作者分散式系統，也不宣稱其通訊加速。

[AdaMuon v3](https://arxiv.org/html/2507.11005v3)採作者程式的 Nesterov sign direction；本地版本使用 replicated FP32二階矩與最終方向，取代作者 BF16 通訊 buffer，屬明示的精度/通訊變體。[SOAP 官方實作](https://github.com/nikhilvyas/SOAP/tree/a1e553530fde97d0e6b307d7c82ac6d38b072340)固定 betas(.95,.95)、shampoo_beta=.95、refresh10、max_precond_dim10000、merge_dims=False，沒有用 skip 大維度冒充完整 SOAP。完整 Gram/basis 記憶體與 QR 時間可能使 S1 無法在兩小時內完成。

[Schedule-Free Spectral Optimization](https://arxiv.org/html/2605.23061v1)提供 F1 的更新原理：以 x 為平均、z 為快速權重，在 y=.9x+.1z 求梯度；matrix momentum=.95（此為本地設定）、row variance=.95，平均權重按 LR²累積。Auxiliary SF-AdamW 用目前梯度除以二階矩，**沒有額外 Adam 一階動量**；bias correction 納入有效 LR 與平均權重。F2保留相同更新及平均公式，梯度改在 fast weights 求取。評估/匯出都用x，評估後精確還原live weights；為避免轉換漂移明確保存x，記憶體不等同作者的最小狀態版本。

[C1 官方 CCE](https://github.com/apple-aiml-research/ml-cross-entropy/tree/3de376c106a1916bc5e1b619f9c77c87a461ee1c)固定 `cce_exact`、shift、sum reduction，禁用梯度過濾。數值差異必須先過 gate，不能用「exact」名稱替代測試。

K1 教師為既有 raw6B run 的凍結本地 export，對應 HF revision `13125ad091ca25c914e1528af6dce22d5f6efa3a`；全部權重與tokenizer檔案 SHA256取自已完成上傳的校驗收據，存於 `source/teacher-manifest.json`。compute node 的rank0重驗後才能載入教師。學生仍為隨機初始化；teacher只推論、不加入學生 optimizer或export。完整vocabulary KL以128個位置一塊計算，backward重算logits以控制記憶體；不做top-k近似。新job另記teacher forward inputs，保守總model inputs為2,000,158,720；歷史教師6,000,082,944 inputs與64.7066667 H200-hours保留，不能把它稱為免費從零訓練的優勢。

**每個 job 自己執行的流程**

`固定來源/設定校驗 → ≤10分鐘前置檢查 → 隨機模型訓練 → endpoint校驗/匯出 → 額外9k評估`

前置檢查沿用 FA3對SDPA的forward/backward、文件隔離、singleton/padding/continuation與activation-checkpoint測試；增加所選optimizer的12次更新、SOAP basis refresh、8 ranks相同梯度的參數一致性、state roundtrip、SF權重還原，以及KD/CCE loss/gradient reference。每rank實際讀取目標training prefix的首尾batch，驗證8192真實inputs與document mask，發生在完整模型初始化之前。這是合成數值和真實loader gate，**不是已執行完整1B模型的吞吐預測**；真實形狀首次執行仍可能失敗。

gate失敗或超時立即結束本job；沒有 automatic retry、resume、requeue、晉級控制器、artifact cleanup 或 HF upload。總allocation上限7200秒，包含compile、gate、初始化、eval、checkpoint、export與idle。trainer保留終點評估900秒及停止margin300秒；外層timeout保留剩餘流程時間。提前停下保留exit75與紀錄，不延長allocation、不宣稱達到1B。

每個 run 有獨立 configuration digest、run directory、W&B identity、token journal、optimizer/checkpoint、compile/cache/tmp與log。共用原始資料及tokenizer；canonical資料不寫入，各run保存自己的驗證receipt。實驗原始碼在本目錄 `source/` 隔離，先前封存實驗不受修改。`execution-seal.json`是完整性校驗，不代表已提交或已產生結果。

**評估與事先決定的判讀規則**

所有組固定1k評估每400更新、既有token knots、終點；若速度較慢，既有wall-time gate亦會觸發評估。主要觀測是硬標籤held-out NLL，不以訓練loss排名。每組在自己的1B終點直接評估其餘9k文件，不等其他組選出候選；因此9k只用於這批預先指定比較，之後調參應視為已看過的資料，不能再當全新盲測。

PPL=`exp(ΣNLL/Σvalid_prediction_tokens)`，不平均batch/document PPL。固定1k有1,479,302 prediction tokens，其餘9k有13,666,812。儲存 `selection-documents.json`及`confirmation-documents.json`的逐文件NLL與有效labels；只比較相同輸入終點、相同evaluation slots、相同checkpoint step。對SF用平均x，其他用最後live weights，不在9k上挑best checkpoint。完整50k OJ保留，不在這批實驗評估。

只比較seed42的8個預定對比及差值，不新增第二seed；此設計不估計訓練seed的變異。建議初篩標準預先固定為：9k NLL至少下降.005 nats，並報告計算成本；只達到固定tokens品質改善但等H200-hours沒有優勢者，記為token效率改善。D025可另以非劣性+.005 nats且allocation成本降低至少15%作效率候選。C1品質以非劣性+.002 nats作初篩。歷史控制的計時／記憶體不是同期同儀器量測，因此D025的15%成本門檻及C1的吞吐≥10%／memory≥20%門檻，在這版只作探索性參考，不能據此宣稱嚴格加速或節省記憶體已獲證實；需要這類結論時另做同作業短benchmark，不默默重跑1B控制組。

確認不確定性時，以文件為cluster、方法配對，resample時保留各文件所有windows；重算 `ΣΔNLL/Σtokens`，而非平均每文件平均loss。對8個預定對比使用Bonferroni同時區間（每個99.375%），區間只描述這次seed42已訓練模型的文件抽樣不確定性，不代表跨seed穩定性；至少20,000次bootstrap須放compute job，不能在login node執行。這個後續統計job尚未提交，也未包含在8個新訓練job的128H上限。`collect_results.py`目前只讀小型summary給point estimates，不會產生或假裝已有CI。

未達1B的run只報共同已量測token點的曲線與時限內進度，不填入1B終點表。不要外推末段loss、把超時視為整個方法無效，或將錯誤/非有限更新排除成本。成本以完成後sacct reconciliation為準。同步記錄optimizer時間、整步時間、峰值memory及各group LR/decay；KD額外記hard CE、KL、teacher inputs。原先「第一列前256元素」的embedding診斷不作支持機制的證據；這版沒有聲稱已完成前一份研究構想中的詞頻分層activation分析。

**資源與操作**

| 選擇 | 獨立job數 | 總allocation上限 | 全部同時執行的最高需求 |
|---|---:|---:|---:|
| 核心D1/D025，N0沿用，seed42 | 2 | 32 H200-hours | 16 H200 + 16 CPU |
| 全8個新方法，N0沿用，seed42 | 8 | 128 H200-hours | 64 H200 + 64 CPU |

可只選任意2組同時提交，上限32 H200-hours、16 H200/16 CPU。這些是job上限加總，不是已用量、可用餘額或queue保證。原來所有歷史正式run、失敗allocation及教師成本都保留在共享ledger。每個新job是1B方法pilot；沒有任何正式run例外或新的長程訓練承諾。

從repository root預覽，不分配資源：

```bash
python3 slurm/methods_parallel.py plan --arms all --seeds 42
```

需要實際啟動時，以下工具會逐一呼叫獨立 `sbatch`，而後讓Slurm並行排程；兩個新作業互相沒有dependency：

```bash
python3 slurm/methods_parallel.py submit --arms D1,D025 --seeds 42 --budget-h200-hours 32
```

全部8個新作業用 `--arms all --seeds 42 --budget-h200-hours 128`。`all`不包含N0；seed43與N0提交都會被拒絕。工具要求明確填寫本次提交的總上限。每個`sbatch`先hold、完成ledger綁定後release，避免job先啟動但帳本尚未登記。AGENTS要求所有allocation記帳，因此應透過wrapper使用 `slurm/methods-parallel.sbatch`，不手動繞過reservation。沒有Slurm array或跨job依賴，也沒有安裝全域併發throttle；請依可用配額選擇一次提交的組數。

```bash
# 以下都不建立新的訓練allocation。
python3 slurm/methods_parallel.py reconcile --arms all --seeds 42
python3 experiments/methods-parallel-20261019/collect_results.py
# 僅限某次sbatch結果不確定時識別並綁回原job；不重送、不自動release。
python3 slurm/methods_parallel.py recover --key D1-seed42
```

同run ID不允許重試；任何模糊提交保留reservation，不會偷偷再花一份預算。作業完成後再執行reconcile取得實際allocation成本。共享ledger使用既有檔案鎖；只在這個新study的submission目錄維護提交receipt。

**本次驗證紀錄**

11個標準函式庫契約測試通過：8個新作業相同token幾何、拒絕seed43與N0重送、歷史控制校驗、相同seed對照、SF排程、資源加總、設定鎖定、無Torch匯入、hold/bind/release順序、保留17筆模擬歷史formal身份、不確定提交保留reservation且不重送。Slurm呼叫均為mock，暫存帳本只在`/tmp`。全部Python AST與shell語法檢查另存於 `validation.json`。未執行任何GPU測試、模型載入或訓練。
