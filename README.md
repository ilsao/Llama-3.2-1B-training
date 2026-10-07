# Llama-3.2-1B-training

本專案使用 `dataset.py` 準備 `allenai/dolma3_mix-150B-1025`，並透過 `train.py`
以 **隨機初始化權重**訓練 Llama 3.2 1B。訓練採用 Accelerate、BF16 DDP、AdamW，
序列／context 長度固定為 **8192 tokens**。

作業目標是在 **2026-10-09 前**達到 held-out perplexity ≤20，並在 **2026-10-20**
完成實驗與交付；期限以 `Asia/Taipei` 為準。16 H200-hours 是 baseline 的規劃估計，
不是已量測的結果。每個邏輯訓練 run 最多處理 **6,000,000,000 個非 padding input
tokens**、使用 **64 H200-hours**，同時最多 **8 張 H200**；兩人小組最多三個正式 run。
續跑沿用原 run 的 token 與 GPU-hour 預算，所有 pilot 與驗證成本也必須記帳。

## 資料處理流程

1. 下載固定 revision 的完整原生 `train` 文件 split，保存 Hugging Face 原生 builder
   的檔案順序。資料集 viewer 的不完整內容或自動轉換的 Parquet 不能取代原生資料。
2. 逐篇核對原始 JSONL 的文件 ID／文字與原生資料集的列順序。**在任何清理或
   tokenization 之前**呼叫 map-style `Dataset.shuffle(seed=42)`，保留打亂後最後
   **50,000 篇**作為評估集。保存 permutation、inverse、評估文件索引，以及評估集
   前 **10,000 篇**的固定訓練中評估子集索引。
3. 僅清理訓練集：移除空白文件，以未修改的 UTF-8 文字 SHA256 做完全去重。
   重複文件保留 shuffle position 最小者，即使它在實體 shard 中較晚出現。
   評估文字保持原樣，評估文件不參與去重或品質推論。
4. 使用官方 DCLM OH2.5 + ELI5 **CPU fastText** 分類器，對符合條件的 Common Crawl
   文件評分。依原始浮點高品質機率排序，恰好保留 `floor(N / 10)` 篇，同分時依
   shuffle position 排序。這是全域文件數排名，不是各 shard 分別選取、固定門檻
   `0.018112`，或保留 10% tokens。其他來源中符合條件的文件全部保留為候選。
5. 按原本的 shuffle 順序遍歷候選文件，使用固定 revision 的 Llama tokenizer，
   每篇明確加入一次 BOS 與 EOS。只選取能完整放入剩餘 **6B 非 padding input token**
   預算的文件，特殊 tokens 也計入。超出剩餘預算的文件整篇跳過，不截斷、不重複；
   最終資料量可能少於 6B。
6. 將訓練文件打包為 **8192-token** 序列，以 EOS 區隔文件，使用一般 causal attention。
   結構性 BOS 與 padding 的 labels 設為 `-100`；每個序列的第一個位置也忽略，因為
   causal LM loss 會將預測位移一格。Padding 使用 EOS 並設 `attention_mask=0`，
   真正的 EOS 仍計入 token 數。尾端補 padding，資料產物不丟棄訓練 tokens。
7. 原始評估文件逐篇獨立 tokenize。長文件使用長度 8192、stride 8191 的 windows，
   使文件內每個預測目標恰好評分一次。評估序列不跨文件邊界；重疊的 input tokens
   與 prediction tokens 分開記錄。

資料產物的 6B 上限與 trainer 的實際處理量是兩個計數。Trainer 仍須累計所有 ranks、
梯度累積 microbatches、跳過 optimizer update 的 batches、重複處理及續跑中的非
padding inputs，並在下一個 batch 超過 6B 前停止。評估 perplexity 的計算方式為
`exp(sum NLL / sum valid shifted labels)`，不能平均各 batch 的 perplexity。
前處理成功本身不代表已量得 perplexity，或已驗證訓練預算合規。

## 官方實作與資料來源追溯

資料配方參考 [DataDecide](https://proceedings.mlr.press/v267/magnusson25a.html)
的來源／篩選比較；文字清理未使用其實驗結果分析 notebook，也未將作業資料集
替換為已發布的 DataDecide corpus。目前配方不宣稱是最佳篩選方式。

隨專案附帶的評分函式取自
[固定 commit 的 DCLM 原始碼](https://github.com/mlfoundations/dclm/blob/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py)。
MIT 授權保留於 `third_party/DCLM-LICENSE.txt`。只移除上游依賴 fastText 的型別註記，
讓輕量測試不用匯入 fastText。文字行的合併只用於分類器輸入，不修改保存的訓練文字；
評分保留原始機率，包括 fastText epsilon。

資料集、模型 revisions 與配方設定固定於 `configs/dataset.json`；直接依賴版本固定於
`requirements-data.txt`。環境安裝另寫入完整的 `requirements-resolved.txt`，之後的
setup jobs 重用該檔。所有解析後的套件版本均納入 preparation fingerprint；每次
preparation 都記錄程式雜湊、Git commit／工作目錄狀態與設定。修改程式、設定或環境
後須使用新的 preparation ID。來源檔名用來區分六種已確認的資料來源；遇到未知
來源路徑時停止流程，不會直接當成網頁資料。

## Slurm 環境與腳本一覽

**nano4 login node 只用於編輯、輕量檢查與提交／查詢 jobs。** 下載完整資料、
shuffle／split、清理、tokenization、packing、模型載入、訓練與評估，都須經 `sbatch`
在 compute nodes 執行。`bash slurm/prepare.sbatch` 不會取得 Slurm allocation。

現有前處理腳本沿用專案記錄的資源例外：目前帳號未確認可用的 CPU partition，
因此 CPU 前處理也配置 **1 張 H200**，但 fastText 本身在 CPU 執行。
腳本使用 `ACD115198` account、`8gpus` partition、單節點、單一 Slurm task、
**8 CPUs** 與 **200G 記憶體**。`8gpus` 是 partition 名稱，不表示每個 job 都會
取得 8 張 GPU，也不能單靠名稱判定 GPU 型號；工作啟動時會記錄並檢查實際 H200。
CPU 前處理與 GPU 訓練使用不同 jobs，前處理 allocation 成本另記於資料帳本。

資料 wrapper 對 `parallel-check` 使用 30 分鐘上限並保留 0.5 H200-hours；
其餘資料階段仍為 24 小時。這些上限包含環境啟動與驗證成本，結束後按實際 allocation 核銷。

| 腳本 | 用途 | 預設 GPU 數 | 預設 wall-time | 提交方式 |
| --- | --- | ---: | --- | --- |
| `slurm/setup.sbatch` | 安裝資料環境 | 1 | 24 小時 | 直接 `sbatch` 或資料 wrapper |
| `slurm/prepare.sbatch` | 資料各階段、`smoke`、`parallel-check` | 1 | 24 小時 | 直接 `sbatch` 或資料 wrapper |
| `slurm/score.sbatch` | DCLM CPU fastText 評分 | 1 | 24 小時 | 直接 `sbatch` 或資料 wrapper |
| `slurm/setup-training.sbatch` | 安裝獨立訓練環境 | 1 | 1 小時 | 訓練 wrapper，必須有預算 reservation |
| `slurm/train.sbatch` | runtime、validate、pilot、scaling、resume | 4 | 30 分鐘 | 訓練 wrapper，GPU／時間依 action 與 config 覆寫 |
| `slurm/template.sbatch` | 原有空白範本，只印時間 | 1 | 4 小時 | 不屬於資料或訓練流程 |

前處理預設最多 7 個 workers，保留 1 個 CPU 給協調程序；各 worker 的底層套件執行緒
限制為 1。訓練由一個 `srun` task 啟動一次 Accelerate，再建立對應 GPU 數的 ranks，
避免每個 Slurm task 各啟動一整套分散式訓練。

資料入口在 Arrow／HF 啟動執行緒前，將 `multiprocessing` 與 `multiprocess`
都設為 `spawn`，並將方法寫入 preparation identity 與 job log。Split 每 5 分鐘
輸出 Python stack，便於診斷卡住的位置。`parallel-check` 使用壓縮 JSON shards
及 threaded checksum 前置步驟，比較 1／7 workers 的原生文件順序與 shuffle。
實際資料的巢狀 metadata 與 dataset card 宣告的 string schema 不符。讀取 adapter
依 pinned builder 的 canonical 檔案順序逐行建立磁碟上的 HF map-style Dataset，
保留原始 JSON line 與 id/text；逐份核對全部欄位後才執行原生 seed-42 shuffle。
`parallel-check` 比較異質欄位的 1／7-worker 輸出、原生 id/text projection 的
shuffle indices、完整 split 入口、eval archive/resume，並在 compute node 有界抽查各來源。
此次 source 變更後使用新 preparation `dolma-dclm10-004`；保留 `002`／`003` 身分與檔案。

資料環境使用叢集既有 Miniconda 安裝，位於
`/work/asciibase64/Llama-3.2-1B-training/environments/data-py311`，Python 版本為 3.11。
原始資料、原生 Arrow cache、fastText 權重、tokenizer、SQLite 索引、packed Arrow
shards、暫存檔與 checkpoints 均放在專案的 `/work` 目錄。前處理不下載 Llama
預訓練權重。原生 Dataset loader 的 download cache 可能與 hub snapshot 同時存在；
磁碟空間須涵蓋兩者、完整原生 Arrow corpus、索引與最終 token 產物。完整建置與
文件評分可能耗時，這份說明不假設處理速度。

認證使用既有 `HF_TOKEN`／`HF_TOKEN_PATH` 或 Hugging Face login token 路徑；
搬移 cache 時保留原認證路徑。`smoke`／`select` 前須已有 gated Llama tokenizer
的存取權限。訓練 pilot／scaling 使用既有 W&B 認證。不要將任何 token 寫入 README、
設定或 Git。提交前可指定 `DATA_ROOT`／`DATA_CONFIG`，儲存根目錄須位於
`/work/asciibase64/`；自訂 `DATA_ROOT` 時使用 wrapper，才能同步調整 log 路徑。

## 完整提交順序與 sbatch 指令

一般的 4 GPU 路線為：

```text
資料 setup → smoke → parallel-check → download → split → clean
           → score → select → pack → verify
           → 訓練 setup → runtime（4 GPU）→ validate（4 GPU）→ AdamW pilot
```

`parallel-check` 是平行前處理的驗證，放在完整資料處理前；已在相同程式與環境
完成的驗證可重用。若要比較 8 GPU，在 4 GPU 驗證後另做 8 GPU runtime／validate，
再執行兩組 scaling 與比較。續跑是原階段／原 run 的延伸，不是新的下游階段。

**每個 job 成功完成、檢查結果並核銷後，才提交下一個依賴它的 job。** 下表的命令
應逐條執行；不能一次貼上全部命令，因為 `sbatch` 回傳 job ID 只代表已提交，
不代表工作成功或已完成。

### 1. 提交前準備

以下為 login node 可執行的輕量指令：

```bash
cd /home/asciibase64/Llama-3.2-1B-training
export DATA_ROOT=/work/asciibase64/Llama-3.2-1B-training
mkdir -p "$DATA_ROOT/logs"
```

Slurm 會在執行腳本前開啟 log 檔案，因此 logs 目錄必須先存在。下列 preparation／run
IDs 是範例；新實驗須使用唯一 ID，同一資料流程的正式階段則使用同一 preparation ID。

### 2. 資料準備：所有階段的 sbatch 指令

| 順序 | 提交命令 | 工作內容及前置條件 |
| --- | --- | --- |
| 1 | `sbatch slurm/setup.sbatch` | 建立資料環境、安裝依賴並保存解析後版本；後續資料 jobs 依賴此環境。 |
| 2 | `sbatch slurm/prepare.sbatch smoke dolma-smoke-001` | 小型檢查：原生 HF shuffle、官方分類器、gated tokenizer 與固定長度 Arrow／HF round trip。 |
| 3 | `sbatch slurm/prepare.sbatch parallel-check dolma-parallel-001` | 比較 1／7 workers 的原生順序、shuffle、去重、評分、tokenization、packing 與中斷續跑；檢查 worker 失敗不提交進度。 |
| 4 | `sbatch slurm/prepare.sbatch download dolma-dclm10-001` | 下載固定 revision 的完整原生資料，保存 canonical 檔案順序。 |
| 5 | `sbatch slurm/prepare.sbatch split dolma-dclm10-001` | 依賴 download；核對原生列順序、seed 42 全域 shuffle，保留最後 50,000 篇與固定 10,000 篇子集索引。 |
| 6 | `sbatch slurm/prepare.sbatch clean dolma-dclm10-001` | 依賴 split；只清理 training 空白文件及完全重複文字。 |
| 7 | `sbatch slurm/score.sbatch dolma-dclm10-001` | 依賴 clean；以官方 CPU fastText 對符合條件的 Common Crawl 文件評分。 |
| 8 | `sbatch slurm/prepare.sbatch select dolma-dclm10-001` | 依賴 score；保留 CC 全域前 10% 文件，按 shuffle 順序 tokenize 並選取最多 6B tokens。 |
| 9 | `sbatch slurm/prepare.sbatch pack dolma-dclm10-001` | 依賴 select；訓練 packing 為 8192 tokens，評估逐文件切 windows，建立 masks 與來源欄位。 |
| 10 | `sbatch slurm/prepare.sbatch verify dolma-dclm10-001` | 依賴 pack；完整驗證 split、來源、品質排名、token 上限、packing／loss masks 與 checksums，產生驗證封存紀錄。 |

`smoke` 和 `parallel-check` 使用獨立測試 ID；`download` 到 `verify` 全部使用
`dolma-dclm10-001`。`parallel-check` 的測試目錄必須是新的，失敗重跑時換新測試 ID。
正式 preparation 修改程式、設定或環境後也須換新 ID，不能混用不同 revision 的產物。

上述資料 jobs 也可改用 `slurm/submit.sh`；它會建立 logs、呼叫 `sbatch`，並把 job ID
寫入 `research-state.yaml` 與 `logs/preprocessing-*.json`。它不會自動串接下游或自動核銷。各列的對應指令如下，
同樣須逐一等待完成：

```bash
bash slurm/submit.sh setup
bash slurm/submit.sh smoke dolma-smoke-001
bash slurm/submit.sh parallel-check dolma-parallel-001
bash slurm/submit.sh download dolma-dclm10-001
bash slurm/submit.sh split dolma-dclm10-001
bash slurm/submit.sh clean dolma-dclm10-001
bash slurm/submit.sh score dolma-dclm10-001
bash slurm/submit.sh select dolma-dclm10-001
bash slurm/submit.sh pack dolma-dclm10-001
bash slurm/submit.sh verify dolma-dclm10-001
```

直接 `sbatch` 與 wrapper 是兩種替代提交方式，同一工作只選一種。

### 3. 每個 job 的狀態、logs 與核銷

將 `JOB_ID` 換成 `sbatch`／wrapper 回傳的實際編號：

```bash
squeue -j JOB_ID
sacct -X -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES
scontrol show job JOB_ID
```

| 狀態／exit code | 下一步 |
| --- | --- |
| `PENDING`／`RUNNING` | 等待；還不能開始依賴它的下一階段。 |
| `COMPLETED` 且 `ExitCode=0:0` | 檢查 `.out`／`.err` 與階段結果、完成核銷，再提交下一階段。 |
| `ExitCode=75:0` | 程式因時間／停止要求退出；查看保存進度，核銷後續跑該階段，尚不能開始下游。 |
| 其他失敗 | 查看 logs 及 job record，修正原因後再提交；若程式／環境改變，使用新的 preparation 或 run ID。 |

資料環境安裝、`smoke`、`parallel-check` 與所有資料階段結束後，用資料帳本核銷：

```bash
python3 slurm/accounting.py JOB_ID
```

訓練環境安裝、runtime、validate、pilot、scaling 與 resume 結束後，用訓練帳本核銷：

```bash
python3 slurm/submit_training.py accounting JOB_ID
```

若訓練提交的回應不確定，wrapper 會保留預算 reservation，禁止直接重送。
每次提交前保存唯一 Slurm comment、時間與命令；使用錯誤訊息提供的 ID 復原：

```bash
python3 slurm/submit_training.py recover submission-RESERVATION_ID
```

復原只查一次 queue、accounting 與 startup allocation receipts，找到唯一 job 後
綁定原 reservation；未找到或有多筆時仍保留預算。nano4 的 accounting 不保存
job comments，因此終止 jobs 可能須依 startup receipt 辨識。已確認無 job 的失敗
須記錄證據再釋放 reservation，不能因 client error 自動判定沒有提交。

兩個指令都只查詢小型 Slurm metadata，可在 login node 執行。即使 job 失敗，已配置的
GPU 時間仍須核銷。Slurm accounting 尚未提供完整紀錄時，稍後再查；不要自行填入估計。

資料 logs 為 `logs/dolma-data-env-JOB_ID.*`、`logs/dolma-prepare-JOB_ID.*` 或
`logs/dolma-fasttext-JOB_ID.*`；訓練 logs 為 `logs/llama-ACTION-JOB_ID.out`／`.err`。
以上均相對於 `DATA_ROOT`。每個 job 的 allocation、node、GPU 型號、開始／結束時間、
exit status 與 accounting snapshot 另存於 `logs/job-JOB_ID/`。

資料帳本為 `/work/asciibase64/Llama-3.2-1B-training/preprocessing-ledger.json`，
按 job ID 冪等核銷，重複執行不會重複計費。它記錄實際 GPU-hours／H200-hours；
無法從 typed TRES 或該 job 保存的硬體證據確認 H200 時會停止，不推測 GPU 型號。
它與訓練 experiment ledger 分開，不能將訓練 allocation 改記為資料前處理。

### 4. 資料階段續跑與下載重用

`bash slurm/submit.sh STAGE PREPARATION_ID [dataset arguments]` now persists a unique
submission intent in `research-state.yaml` before dispatch and tags the Slurm job.
It requests eight CPUs total and the authorized one-H200 preprocessing allocation,
with a 24-hour ceiling. Returned receipts live under `DATA_ROOT/logs/preprocessing-*.json`.
A lost, timed-out, or malformed reply retains its 24 H200-hour reservation and blocks
another stage for that preparation. Recover once with
`bash slurm/submit.sh recover preprocessing-TAG`; recovery checks queue/accounting
and startup allocation receipts without submitting or polling. Reconcile the recovered
job with `slurm/accounting.py` after termination. Stage exit/artifact checks still apply.
The wrapper updates durable state under a submission lock; invoke it only from the
single heartbeat controller (or a foreground maintenance lease), not concurrently
with state edits. The preserved `.download-continuation` remains unchanged.


前處理依實際 Slurm end time 保留 **15 分鐘**停止餘裕，在已提交的 batches／shards
保存進度。Exit **75** 時，核銷後重新提交同一階段與同一 preparation ID，例如：

```bash
# 只有前一次 split 尚未完成且設定未改變時，才重送這個命令
sbatch slurm/prepare.sbatch split dolma-dclm10-001
```

中斷的來源核對 shard 或 split-index 建置會於重啟時重新檢查；全域 SQL 排名、索引建置、
checksum 或 verify pass 可能須重新執行該操作。Verify 沒有部分成功的封存紀錄。
慢速檔案操作或單次原生 loader 操作仍可能超過停止餘裕，Slurm wall-time 是最後上限。
每個 preparation 使用檔案鎖防止同時寫入；已提交的產物不可修改。只有完成完整驗證、保存獨立 eval 全集後，才可透過
下述 cleanup 工具釋放冗餘產物；禁止手動刪除仍被引用的 caches。

如果修改前處理配方，但 dataset ID／revision 未變，可用新 preparation ID 重用舊流程
**已完成**的 download。程式會重新核對來源 manifest、檔案順序與全部 checksums，
不複製原始檔；這仍是 compute-node 工作：

```bash
sbatch slurm/prepare.sbatch download dolma-dclm10-002 \
    --reuse-download-from dolma-dclm10-001
# 成功、核銷後，split → verify 全部改用 dolma-dclm10-002
```

## 資料產物與訓練整合

每個階段在 `DATA_ROOT/preprocessed/PREPARATION_ID/STAGE/` 寫入 manifest，保存
checksums、來源關係與統計。清理階段的不可變 SQLite 索引記錄勝出的原始 row IDs、
文字雜湊、來源與 shuffle position；`empty` table 記錄空白文件，其餘原始訓練列是
重複文件。評分階段保存全部符合條件的網頁評分。選取階段保存前 10% 候選索引、
選中文件 IDs、文字雜湊、token IDs 與 evaluation slots；預算跳過行為可由候選順序
與總數重現。split 階段將全部 50,000 篇評估文件的原始欄位獨立保存於
`split/evaluation-raw/` 的 `document_json` large-string 欄位，逐行保留全部原始 JSON，
可用 `json.loads` 還原異質巢狀欄位及缺省 keys；附原始 row index、shuffle position、evaluation slot、來源及
文字 checksum；不依賴完整原生 Arrow cache。所有 eval document tokens 與 packed
sequences 均保留，固定子集仍為 reserved order 的前 10,000 篇。

Packed Arrow 包含 `input_ids`、`attention_mask`、`labels`、`document_spans`
（原始 row 與 token offsets）及 `evaluation_slot`（訓練為 -1）。完整 verify 會獨立
重建 canonical native shuffle，檢查 checksums、split 的一對一映射、所有選取文件
來源、品質排名、token 上限，並逐一重建序列與 masks，最後封存不可變的 pack manifest。

`verify` 成功後，trainer 才能在 compute node 載入資料：

```python
from dataset import load_prepared

path = '/work/asciibase64/Llama-3.2-1B-training/preprocessed/dolma-dclm10-001'
train = load_prepared(path, 'train')
evaluation = load_prepared(path, 'evaluation_subset')
# 建立模型 batch 時，移除來源追溯欄位。
train = train.select_columns(['input_ids', 'attention_mask', 'labels'])
```

Loader 拒絕未驗證或 pack manifest 已變動的資料，預設檢查 packed shard checksums，
回傳 map-style Hugging Face Dataset。評估子集選取完成前須保留 `evaluation_slot`；
可比較的 preparations 使用相同保留集前 10,000 篇文件。

## 訓練實作

第一個 pilot 預設 **4 H200 × 最多 30 分鐘 = 2 H200-hours**，最多 100 次 optimizer
updates。每 GPU microbatch 為 1、gradient accumulation 為 4，global batch 為
16 個 8192-token sequences。設定位於 `configs/train-pilot.json`；這些是配置上限，
不代表已通過 GPU runtime 驗證或已達到 perplexity ≤20。

模型只透過固定 revision 的 `AutoConfig` 與 `from_config` 隨機初始化，不載入預訓練
權重。保留原始 RoPE、架構與 tied embedding／output weights，使用既有 fast tokenizer，
訓練不重新 tokenize 或 packing。AdamW 的 decay group 為一般矩陣；embedding、norm、
bias 不做 weight decay，tied parameters 只分配一次。採用 BF16 autocast、FP32
parameters／optimizer states、SDPA 與 activation checkpointing。

資料按既有 packed 順序分給各 rank，不重複、不補樣本。目前僅處理完整的 distributed
accumulation window；尾端不足一個 window 的 sequences 留在資料集但不訓練，數量
寫入 summary。每個 window 前，以所有 ranks 的 `attention_mask` 計算實際非 padding
input tokens，先確認不會超過上限。已處理 inputs 都計入，包括 skipped updates 與
checkpoint rollback 後再次處理的資料；prediction-label count 只用於 loss normalization。

持久化 `token-journal.json` 在 forward 前先 fsync reservation，forward／backward 後
確認數量。`confirmed_tokens` 是實際確認的 inputs；中斷而無法確認的 reservation
保守保留於 `charged_tokens`，不宣稱是實測值，也不因載入舊 checkpoint 而消失。
因此可能提早停止，但不會放寬 6B 預算。

Pilot 固定使用既有 evaluation subset 前 **100 篇文件**，包含每篇所有 8192／stride-8191
windows。文件索引、checksum 與 split provenance 存入 run 目錄，同設定 runs 保持一致；
100 篇的排名抽樣不確定性較高。Evaluation forward 移除 window 尾端 padding，保留
全部真實 inputs 與原本 causal prediction targets；訓練序列仍固定 8192。正式 OJ 使用
保留的全部 50,000 篇。Perplexity 為全域 `exp(sum NLL / sum valid shifted prediction tokens)`，不平均 batch perplexities。

W&B project 固定為 **`lab5-training-llama`**；pilot／scaling 必須 online，只有 validation
可 offline。Training step 定義為 optimizer-update step，續跑保留原 step 與 W&B ID。

| 必要 metric | 本實作的語意 |
| --- | --- |
| `train/loss` | 前一個 logging interval 的 token-weighted loss，跨 microbatches／ranks 累加有效 prediction tokens。 |
| `train/grad_norm` | 當前 update clipping 前的同步 DDP gradient norm；replicated parameters 只計一次。 |
| `train/learning_rate` | 當前 update 使用的 primary `decay` group LR；另記錄各 group LR。 |
| `train/tokens_per_second` | 當前 update 的實際非 padding inputs 除以該 step 實測 duration。 |
| `train/total_tokens_seen` | 持久化 journal 的累積 `confirmed_tokens`，跨 jobs 續計；保守計費量另記 `train/budget_tokens_charged`。 |
| `eval/perplexity` | 固定 held-out 子集的全域 NLL／有效 prediction tokens 取 exponential。 |

只有 `train/loss` 是 interval average。Throughput 使用 CUDA synchronize 後各 rank 的
最大 step duration，包含 input transfer、forward、backward、clipping、同步與 optimizer
update，排除資料讀取、journal、evaluation、logging 與 checkpoint 時間。

Pilot 預設每 10 steps 評估並存檔，結束時再評估。程式依實測耗時縮短 planned steps
及 evaluation interval，維持至少每 10% planned steps 評估一次，保留至少 **3 分鐘**
停止餘裕；慢速 I/O 仍可能超過餘裕，Slurm wall-time 是最後界線。

## 訓練提交：依序執行的指令

訓練使用獨立環境
`/work/asciibase64/Llama-3.2-1B-training/environments/training-py311`，保留原資料環境。
**必須使用 `slurm/submit_training.py` 提交。** 它先建立 logs、預留 GPU-hour 預算，
再呼叫 `sbatch` 並將 job ID 綁定共享帳本。直接提交 `setup-training.sbatch` 或
`train.sbatch` 會缺少 reservation，被 `training-common.sh` 拒絕。

### 1. 四 GPU AdamW 路線

以下每一步完成後，先依前述方式檢查 `sacct`／logs，並用
`python3 slurm/submit_training.py accounting JOB_ID` 核銷，再執行下一步。
`validate`／`pilot` 提交時就要求 preparation 已有 `verify/manifest.json`。

| 順序 | 提交指令 | 用途 | 最大 allocation 成本 |
| --- | --- | --- | ---: |
| 1 | `python3 slurm/submit_training.py setup train-env-001` | 建立獨立訓練環境、安裝及檢查依賴；1 H200，最多 60 分鐘。 | 1 H200-hour |
| 2 | `python3 slurm/submit_training.py runtime train-runtime-001` | 4 GPU 小型隨機 Llama correctness checks，不需要完整資料集；最多 10 分鐘。 | 2/3 H200-hour |
| 3 | `python3 slurm/submit_training.py validate train-smoke-001 --prep-id dolma-dclm10-001` | 先做 runtime checks，再用真實 1B／8192 資料跑 2 steps，驗證 checkpoint、resume 與 HF export reload；最多 30 分鐘。 | 2 H200-hours |
| 4 | `python3 slurm/submit_training.py pilot adamw-pilot-001 --prep-id dolma-dclm10-001` | 4 H200 AdamW pilot；最多 30 分鐘／100 updates，以先到的停止條件為準。 | 2 H200-hours |

Runtime 檢查涵蓋 tied parameter coverage、padding／結構性 loss masks、token-weighted
DDP gradients、不均長度分散式評估，以及 weights／optimizer／scheduler／RNG 恢復。
真實 1B 的 `validate` 使用 `configs/train-smoke.json`，只評估 1 篇文件，offline W&B；
它是功能驗證，不可用其 perplexity 代表 pilot 或完整 held-out 結果。

### 2. 可選的四／八 GPU scaling 比較

要使用 8 GPU，先依序完成對應 runtime／真實模型驗證。每列完成後同樣檢查及核銷；
以下新 run IDs 不可與四 GPU 驗證共用。

| 順序 | 提交指令 | 用途 | 最大 allocation 成本 |
| --- | --- | --- | ---: |
| 1 | `python3 slurm/submit_training.py runtime train-runtime-8-001 --gpus 8` | 8 GPU 分散式小模型 correctness checks；最多 10 分鐘。 | 4/3 H200-hours |
| 2 | `python3 slurm/submit_training.py validate train-smoke-8-001 --prep-id dolma-dclm10-001 --config configs/train-smoke-8.json --minutes 15` | 8 GPU 真實 1B／8192、checkpoint／resume／export reload；最多 15 分鐘。 | 2 H200-hours |
| 3 | `python3 slurm/submit_training.py scaling adamw-scaling-4-001 --prep-id dolma-dclm10-001 --config configs/train-scaling-4.json` | 4 H200 × 最多 10 分鐘；microbatch 1、accumulation 4，global batch 16。 | 2/3 H200-hour |
| 4 | `python3 slurm/submit_training.py scaling adamw-scaling-8-001 --prep-id dolma-dclm10-001 --config configs/train-scaling-8.json` | 8 H200 × 最多 10 分鐘；microbatch 1、accumulation 2，global batch 同為 16。 | 4/3 H200-hours |

`train-smoke-8.json` 的 run 上限為 2 H200-hours；因此需明確指定 `--minutes 15`。
若沿用 validate 預設 30 分鐘，8 × 0.5 = 4 H200-hours，帳本會拒絕提交。
`--gpus` 只適用於 runtime；訓練／validate 的 GPU 數由 `--config` 的 `world_size` 決定。

兩組 scaling 使用相同配方、seed、split、eval 設定、100-update 上限與 token 上限。
比較期間須保持程式、環境及資料不變。兩組成功且全部核銷後，在 login node 執行：

```bash
python3 slurm/compare_scaling.py adamw-scaling-4-001 adamw-scaling-8-001
```

比較工具採用 warmup 後相同 steps 的最後 20 個 updates，並計算包含整個 allocation
的實際 H200-hours／training token 成本。只有至少 20 個可比較 updates、8 GPU
throughput 至少是 4 GPU 的 **1.8 倍**，且每 token allocation 成本比不超過 **1.25**，
才選擇 8 GPU；資料不足或門檻未達則保留 4 GPU。10 分鐘配置不保證可完成足夠 updates。

### 3. Wrapper 實際呼叫的 sbatch

訓練 wrapper 使用以下參數組合；此處用來對照 Slurm allocation，實際操作仍使用上方
wrapper 指令，讓同一個 job 在提交前建立 reservation。

| action | 腳本 | GPU 數 | `--time`（分鐘） | 腳本位置參數（依序） |
| --- | --- | ---: | ---: | --- |
| `setup` | `slurm/setup-training.sbatch` | 1 | 60 | `train-env-001` |
| `runtime`（4 GPU） | `slurm/train.sbatch` | 4 | 10 | `train-runtime-001 runtime - configs/train-pilot.json` |
| `runtime`（8 GPU） | `slurm/train.sbatch` | 8 | 10 | `train-runtime-8-001 runtime - configs/train-pilot.json` |
| `validate`（4 GPU） | `slurm/train.sbatch` | 4 | 30 | `train-smoke-001 validate dolma-dclm10-001 configs/train-smoke.json` |
| `validate`（8 GPU） | `slurm/train.sbatch` | 8 | 15 | `train-smoke-8-001 validate dolma-dclm10-001 configs/train-smoke-8.json` |
| `pilot` | `slurm/train.sbatch` | 4 | 30 | `adamw-pilot-001 pilot dolma-dclm10-001 configs/train-pilot.json` |
| `scaling`（4 GPU） | `slurm/train.sbatch` | 4 | 10 | `adamw-scaling-4-001 scaling dolma-dclm10-001 configs/train-scaling-4.json` |
| `scaling`（8 GPU） | `slurm/train.sbatch` | 8 | 10 | `adamw-scaling-8-001 scaling dolma-dclm10-001 configs/train-scaling-8.json` |
| `resume`（pilot 範例） | `slurm/train.sbatch` | 4 | 5 | `adamw-pilot-001 resume dolma-dclm10-001 configs/train-pilot.json` |

例如，4 GPU pilot 的 wrapper 在預留帳本後，實際提交等同於：

```bash
sbatch --parsable --cpus-per-task=8 --time=30 --gpus-per-node=4 \
    --job-name=llama-pilot \
    --output=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.out \
    --error=/work/asciibase64/Llama-3.2-1B-training/logs/%x-%j.err \
    /home/asciibase64/Llama-3.2-1B-training/slurm/train.sbatch \
    adamw-pilot-001 pilot dolma-dclm10-001 \
    /home/asciibase64/Llama-3.2-1B-training/configs/train-pilot.json
```

其他 action 依上表替換 script／GPU／minutes／位置參數，`--job-name` 為 `llama-ACTION`。
Config 路徑會由 wrapper 解析為絕對路徑；runtime 的 `-` 表示不需要 preparation。
`--parsable` 回傳便於程式解析的 job ID；`%x`／`%j` 分別為 job name／job ID。
Account、partition、單節點、單 task、memory 與 signal 設定由 `.sbatch` 提供，命令列的
GPU／時間／job name 覆寫腳本預設。`setup` 只傳 run ID，不傳 action／prep／config。

### 4. 訓練續跑、dry-run 與依賴

訓練 exit **75** 表示因 wall-time／signal 停止且已嘗試存檔。先查看 logs 與最新完整
checkpoint，再核銷前一 allocation，以**相同 run ID、prep ID、config、程式與環境**續跑。
目前要求相同 GPU 數、microbatch 與 accumulation；step、data cursor、scheduler、
optimizer、各 rank RNG、W&B ID、tokens 與 allocation 成本均延續。初始化中斷且沒有
checkpoint 時，重建相同 seed 的原始隨機模型，保留已占用 token／compute 預算。

```bash
# 前一次 job 已核銷，且剩餘預算至少涵蓋 4 GPU × 5 分鐘時才可提交
python3 slurm/submit_training.py resume adamw-pilot-001 \
    --prep-id dolma-dclm10-001 --config configs/train-pilot.json --minutes 5
```

原 pilot 的邏輯 run 上限固定為 **2 H200-hours**，續跑不會另獲 30 分鐘。
若首次 allocation 已耗滿 2 H200-hours，就沒有可用的續跑額度。Scaling 或自訂設定
續跑時，`--config` 必須明確指定原檔案；resume 未指定 config 時會預設使用
`train-pilot.json`。不同配方使用新 run ID／config，仍記入共享帳本。

提交前可加 `--test-only`，只進行 Slurm 資源 dry-run，不建立 reservation、不配置
compute resources。Validate／pilot 的 dry-run 仍要求 preparation 已完成 verify：

```bash
python3 slurm/submit_training.py runtime train-runtime-check-001 --test-only
python3 slurm/submit_training.py pilot adamw-pilot-check-001 \
    --prep-id dolma-dclm10-001 --test-only
```

一般建議逐步人工檢查。訓練 wrapper 也支援 `--dependency afterok:JOB_ID`，交給 Slurm
等待前置 job 成功，例如在訓練環境 setup 已提交後排入 runtime：

```bash
python3 slurm/submit_training.py runtime train-runtime-queued-001 \
    --dependency afterok:SETUP_JOB_ID
```

將 `SETUP_JOB_ID` 換成實際數字。Dependency 不會替你檢查結果或核銷；exit 75 並非
成功，`afterok` 不會繼續執行。資料 wrapper 沒有自動 dependency 串接；validate／pilot
也不能靠 dependency 跳過提交當下的資料 verify 檢查。

## 訓練帳本、產物與正式 run

共享訓練帳本是
`/work/asciibase64/Llama-3.2-1B-training/experiment-ledger.json`，記錄訓練 setup、
validation、pilot、scaling、正式 runs 與續跑。尚未結束或未核銷的 jobs 保留整個
allocation 預算；用 Slurm 實際 duration／GPU 數，加上 typed TRES 或該 job 啟動時的
H200 硬體證據計算成本，包含初始化、評估、存檔與 idle allocation。缺少型號證據時
拒絕核銷。最多三個正式 runs；pilot／validation 仍須依課程規則做總成本核算，不能
視為免費，也不能以多個接近上限的 pilot 規避正式 run 限制。

Run outputs 位於 `DATA_ROOT/runs/experiments/RUN_ID/`：

- `resolved-config.json`、`initialization.json`、`data-provenance.json`：完整設定、隨機初始化與 split 證據。
- `optimizer-groups.json`、`environment.json`、`wandb.json`、`metrics.jsonl`：參數分組、套件版本、持久 W&B run link 與 metrics。
- `token-journal.json`、`checkpoints/`、`latest.json`：不可回退的 token 計量與完整續跑狀態，保留最新兩份完整 checkpoints。
- `summary.json`、`resource-accounting.json`：實測訓練結果與 Slurm 核銷後的資源成本。
- `export.json`、`export-step-*/`：Hugging Face 相容的 model／tokenizer；程式沒有自動上傳。

小模型 runtime 結果另存於 `runs/runtime-validation/RUN_ID/result.json`；
前處理平行測試結果存於 `runs/preparation-validation/TEST_ID/result.json`。

目前儲存庫提供 validation、pilot 與 scaling configs，沒有已選定的正式 baseline config。
帳本支援 `category="full"`，但 wrapper 沒有獨立 `full` action；後續正式 run 須先準備
經驗證的 config，以 `pilot --config ... --minutes ...` 提交，帳本依 config 的 category
辨識正式 run。不能把範例 pilot 當作已完成的正式訓練，也不能在未知 throughput 下
宣稱達到 baseline 目標。

正式提交作業時，須以 Slurm job 驗證 export 載入與評估，將 weights／tokenizer 上傳到
指定 Hugging Face repository 並確認 OJ 可讀取，記錄 model ID／revision；再找到實際
OJ script，設定 `eval_model_id`。目前流程沒有上傳或 OJ 的 `.sbatch` 腳本，因此本文件
不提供不存在的命令。交付須包含完整設定、W&B link、optimizer 分組、training tokens、
實際訓練時間、GPU 型號／數量、H200-hours、split provenance 與 held-out perplexity。

## Login node 可執行的輕量檢查

```bash
python3 -m unittest discover -s tests -v
python3 dataset.py --help
python3 slurm/submit_training.py --help
python3 slurm/compare_scaling.py --help
for slurm_script in slurm/common.sh slurm/training-common.sh slurm/submit.sh slurm/*.sbatch; do
    bash -n "$slurm_script"
done
```

這些 unit tests 使用小型合成資料與 SQLite fixtures，不載入模型、完整 corpus 或
啟動 workers。`smoke`、`parallel-check`、training runtime 與真實 1B／8192 validate
則必須以本文件的 Slurm 指令提交。通過輕量測試不代表真實資料的 `verify` 已完成。

`slurm/template.sbatch` 是原始空白範本，沒有資料／訓練工作、預算紀錄與完整保護設定，
不需將它排入上述流程。若只要檢查其 Slurm 參數，可使用不配置資源的指令：

```bash
sbatch --test-only slurm/template.sbatch
```

## 本次並行實作驗證狀態

詳見 [validation-status.json](validation-status.json)。輕量測試與 shell syntax 已通過，
8 CPU／1 GPU、8 CPU／8 GPU 的 Slurm test-only 請求也通過。前處理驗證 job
511509 與環境 setup job 511510 已失敗，實際 18 秒／17 秒 allocation 成本已核銷。
`/work` 目前顯示 100 GB 用滿，曾使 ledger 更新回報 ENOSPC；GPU correctness、
真實 1B validation 和 4／8 GPU scaling 尚未完成。既有 caches／資料未刪除，
原下載 job 511170 已失敗（最後紀錄 1072／6081 shards），其 allocation 也已核銷。
恢復可用空間後以新的 validation/run IDs 重新提交；舊下載 ID 使用保留 worktree 續跑。


## Autoresearch heartbeat 與資料清理

Heartbeat 每 20 分鐘透過 cron 執行 `codex queue`，接續綁定的**目前 Codex
聊天室**。Agent 讀取 `HEARTBEAT.md`、`research-state.yaml`、`findings.md` 與
`research-log.md`，推進已授權工作並直接在聊天室回報；不再建立輸出僅寫入檔案的
獨立研究回合。排程使用專案鎖避免重複投遞，維持 login-node／Slurm 分工。
Queue client 最多等待 30 秒；投遞逾時會記錄未確定的結果，不能當成工作已完成。

```bash
python3 slurm/heartbeat.py install
python3 slurm/heartbeat.py status
python3 slurm/heartbeat.py run-once --check-only
python3 slurm/heartbeat.py disable
```

Install／disable 只修改本專案標記的 cron entry，保留其他 entries。Install 預設綁定
`CODEX_THREAD_ID`，也可使用 `install --thread UUID` 明確指定聊天室。目標與投遞
receipt 保存於 `DATA_ROOT/runs/heartbeat/target.json`／`last-run.json`。Heartbeat logs、
執行摘要與 `last-run.json` 位於 `DATA_ROOT/runs/heartbeat/`。短期 foreground lease
會讓 cron 跳過同時修改的回合，到期後自動恢復。

本次原下載以 `.download-continuation/slurm/prepare.sbatch` 接續
`dolma-dclm10-001`；完成及核銷後，主流程使用新 ID `dolma-dclm10-002`：

```bash
sbatch .download-continuation/slurm/prepare.sbatch download dolma-dclm10-001
bash slurm/submit.sh download dolma-dclm10-002 --reuse-download-from dolma-dclm10-001
```

新版 verify 封存完整 eval 原文及 packed 資料。完整驗證後、没有其他本專案 jobs
執行或排隊時，用獨立 Slurm job 預覽及清理：

```bash
sbatch slurm/cleanup.sbatch dolma-dclm10-002 --dry-run --retire-download-source
# 預覽 job 成功並完成核銷後：
sbatch slurm/cleanup.sbatch dolma-dclm10-002 --apply --retire-download-source
```

Cleanup 僅釋放本專案內由 manifests 證實的原始下載、native raw Arrow cache 與
重複的 training document tokens；其餘未知檔案保留。其他 preparation（含未完成
state）的引用、完整 eval 原文／tokens／packed、training packed、tokenizer、索引、
設定、帳本、logs 及 checkpoints 都受到保護。拒絕越界路徑、symlink 刪除路徑及
hardlinks。所有 data jobs 與 cleanup 共用 retention lock。

只有來源 preparation 尚未開始下游工作時，才可使用 `--retire-download-source`。
原始 manifests 保持不變，`retention.json` 記錄釋放的路徑／checksum／原因，
`cleanup-result.json` 記錄本次回收量與清理後離線 reload 結果。已 retired 的
preparation 保持可載入 sealed packed 資料，不能續跑已釋放的歷史階段；重做前處理
須使用新 ID 並重新取得原始資料。Exit 75 的 cleanup 可重送同一命令。
