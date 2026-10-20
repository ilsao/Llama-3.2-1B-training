# Cooldown 提早比例與平台 LR 實驗

沿用上一輪 Muon 配方、raw canonical data 與 0.4B token horizon，分別比較 cooldown 開始時間、低 LR 平台高度，以及兩者的組合。既有 E 的完整 endpoint PPL 28.6830 作參考。依使用者最新指示移除本輪純 B 候選，集中搜尋 E／BE；歷史 B 結果、checkpoint 與帳目保留。

使用者指示這輪不考慮剩餘 H200-hours，因此不以 aggregate headroom 刪減候選。所有 allocation 仍記帳，保留歷史預算及 uncertain reservations；每個候選仍維持 8 H200／8 CPUs、45 分鐘 allocation 與 6 H200-hour logical-run 上限。使用者隨後授權八組平行提交；每組獨立 job，不設相互 dependency。提交和狀態紀錄見 launch.json。

## 變因與 BE 的定義

比例分母是完整訓練 horizon，包含 warmup。cooldown tail 為最後 40%、50%、60%，分別表示在訓練進度約 60%、50%、40% 開始下降。本輪只比較 E／BE 的第一段下降、平台、第二段降零。

現有 E 已在訓練進度 60% 開始 cooldown，和原 B 一樣；兩者直接組合會重複現有 E。本輪 BE 將 cooldown 再提早到 50% 或 40%，保留平台及最後降零。所有 E／BE 在約 80% 到達指定平台，80–90% 維持，最後 10% 線性降到零；提早比例只延長第一段下降，不改平台或最後一段的長度。

| 排程 | 開始下降的進度 | 第一段下降 | 平台期間 | 終點 LR |
| --- | ---: | --- | --- | --- |
| E40 | 60% | 60–80% 降到指定平台 | 80–90% | 0 |
| BE50 | 50% | 50–80% 降到指定平台 | 80–90% | 0 |
| BE60 | 40% | 40–80% 降到指定平台 | 80–90% | 0 |

平台高度是 peak LR 的 5%、10%、20%，同步套用所有 optimizer groups：Muon 為 0.001、0.002、0.004；auxiliary AdamW 為 1.5e-5、3e-5、6e-5。peak LR 分別保持 0.02 與 3e-4；Muon WD 保持 0.1，不加入上一輪較差的 WD=0.01 候選。

## 完整矩陣

| Cooldown tail | 平台 5% 後降零 | 平台 10% 後降零 | 平台 20% 後降零 |
| --- | --- | --- | --- |
| 最後 40% | E40P05，新 run | E40P10，重用完整 E 結果 | E40P20，新 run |
| 最後 50% | BE50P05，新 run | BE50P10，新 run | BE50P20，新 run |
| 最後 60% | BE60P05，新 run | BE60P10，新 run | BE60P20，新 run |

共 9 個 cells、8 組新實驗、1 組既有參考。所有新組從 random weights 開始，既有 E40P10 直接重用完整結果。

4 組主要比較：BE50P10、E40P05、E40P20、BE60P10。這些結果分別回答 E 平台高度與 E／BE 提早程度。另外 4 組 BE50P05、BE50P20、BE60P05、BE60P20 補齊聯合矩陣，檢查提早與平台高度是否互相影響。分組只用於結果判讀；依最新指示八組全部同批提交，不等待主要比較完成。

每個新 job 的 45 分鐘包含初始化、evaluation、checkpoint 和 export。主要 4 組 allocation 上限 24 H200-hours，補矩陣 4 組上限 24，8 組共 48；既有 E 不重跑，歷史成本仍保留。若全部同時提交，需要 8 個獨立 8-GPU jobs，即最多 64 H200／64 CPUs；实际排程取決於 Slurm 配額，不保證同時開始。最新指示授權這八組平行提交；不包含額外診斷、第二 seed 或正式長程訓練。

## 固定條件與整數階段

所有新組從 seed 42 random weights 初始化，Llama 3.2 1B 架構與 tokenizer、context 8192、8 H200、8 CPUs total、microbatch 1／rank、accumulation 2。每 update 131,072 inputs，3051 updates 共 399,900,672 training input tokens；Muon／AdamW group assignments、momentum、betas、clipping、packing/masks 與資料順序保持上一輪設定。

沿用 dolma-raw-baseline-20261008 的 canonical seed-42 document split，與固定 first 1000 reserved subset documents。每 305 updates 評估，另評估最終 endpoint；使用同一 1,479,302 valid prediction tokens。每組用全新 W&B identity，project lab5-training-llama；final_only checkpoint，所有大檔放 /work/asciibase64/Llama-3.2-1B-training。

3051 updates 的 E／BE 整數階段如下：

| 名義 tail | 第一段 cooldown 起始 index | 平台起始 index | 最後降零起始 index |
| --- | ---: | ---: | ---: |
| 40% | 1831 | 2441 | 2746 |
| 50% | 1526 | 2441 | 2746 |
| 60% | 1221 | 2441 | 2746 |

indices 為 zero-based；warmup 是 0–169，最後 update 為 3050。E／BE 最後一個 LR=0 的 batch 仍計入 processed tokens。只改变排程，不重置 token accounting。

## 比較與判讀

主指標是完整 endpoint 的 sum held-out NLL / sum valid prediction tokens，PPL 用 exp(mean NLL)。只有完成 3051 successful updates、399,900,672 charged=confirmed tokens、最終 evaluation、cooldown，且無 skipped updates、Slurm 已對帳且未超限的候選可排名。未完成候選保留 partial 狀態，不插值、不拿最佳瞬間代替 endpoint、不自動 checkpoint continuation。

- E40P10／BE50P10／BE60P10：固定平台 10%，比較 cooldown 提早程度。
- E40P05／E40P10／E40P20：固定開始位置，比較平台高度。
- 聯合矩陣：在四個相關 cells 全部完成後，計算描述性 contrast，例如 L_BE50P05 − L_BE50P10 − L_E40P05 + L_E40P10；負值表示提早與降低平台的 NLL 改善超過加總。仍是單 seed 結果，沒有獨立重複樣本的 interaction 推論。

相對已完成 E40P10 改善至少 0.03 nats/token，才值得規劃後續 paired-seed 確認；這是實用門檻，不是統計顯著性。若既有 E 在最佳值 0.005 nats 內，保留既有 E；其他近似平手依 plan.json 的預先順序。只在實驗完成後報告最低 NLL，不宣稱 0.4B 排名已適用 2B 或完整 reserved-set／OJ PPL。

## 提交前檢查

執行 `python3 slurm/review_cooldown_grid.py --group main` 檢查主要 4 組，`--group interaction` 檢查補矩陣的 4 組，`--group all` 檢查全部並列出提交命令。工具只讀 configuration、source hashes、reference identity 和 ledger，不呼叫 Slurm、不保留資源、不寫共享 state。remaining aggregate H200-hours 不作篩选／阻擋條件，未結算狀態仍需在實際提交前核對。

沿用 slurm/submit_training.py 與 slurm/train.sbatch 的 verified account ACD115198／partition 8gpus、一 node／一 launcher task、200G memory、8 CPUs total、8 H200。完整保留 45 分鐘，不用縮短後續 allocation 補償先前失敗；排除本次卡在 NCCL 初始化的 25a-hgpn151。所有 jobs 檢查實際 GPU 型號與環境，初始化 cache/storage 與帳本紀錄。

實際 launch 前固定來源及配置版本，快照必須包含 plan.json pinned 的所有檔案，特別是 third_party/dclm_fasttext.py 和原 LICENSE；用 AST 核對 local import closure。來源、software environment、evaluation identity、canonical packed prefix 改變時，不能直接重用 E；需先恢復原 identity 或重規劃 matched control。

8 組配置已通過純函數／metadata 檢查：horizon、token budget、固定配方、開始位置、平台高度、非 warmup 區間 LR 單調下降、終點 LR，以及既有 E 的完成與 provenance。未額外啟動 GPU runtime check；八組 submission 的實際結果記錄於 launch.json。

## 八組平行啟動紀錄

2026-10-10 15:09:26 Asia/Taipei，八組均取得獨立 allocation 並進入 RUNNING；每組已核對 8 H200／8 CPUs、45 分鐘，合計 64 H200／64 CPUs。Job IDs 為 522849–522856，候選對應及來源快照見 launch.json。RUNNING 表示 allocation 啟動，完整 endpoint 結果仍待訓練、最終評估與 Slurm 對帳。

## 本輪結果（2026-10-10 15:56 Asia/Taipei）

四組完成 3051 updates／399,900,672 training inputs，固定 1000-document subset 的最終 PPL：BE50P10 27.9514、BE60P10 27.5064、BE50P05 28.5838、BE60P20 27.4331。四组均核對來源、環境、random initialization、evaluation identity 與 token accounting，並已 Slurm 對帳。

E40P05／E40P20／BE50P20／BE60P05 均在 model initialization 前因 NCCL socketPollConnect `No route to host` 失敗，沒有模型、checkpoint 或 training tokens；失敗 allocation 仍記帳。八組 actual allocation 共 24.831111 H200-hours，目前無 active jobs，未重送。最低觀測 NLL 為 BE60P20；與 BE60P10 的差異在 0.005 nats 內，預先 near-tie 順序偏好 BE60P10。四個缺失 cells 未完成，不能宣稱完整矩陣排名或 interaction 已確定。完整精度及 job 對應見 results.json。
