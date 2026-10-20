# Muon 訓練瓶頸實驗

固定 raw 資料，以相同 token 預算比較原配方對照 A、提早 cooldown 的 B、減弱 Muon weight decay 的 C，以及兩者的組合 BC，另加入分段降溫、低 LR 平台、最後降到零的 E。五組完成相同短期 horizon；A/B/C/BC 形成完整 2 × 2 矩陣，E 與 B 比較整體排程效果。歷史 raw 曲線保留作長程參考。

使用者於 2026-10-10 授權五組平行啟動。各組獨立提交、不互設 dependency；實際開始時間由 Slurm 排程決定。每組最多 8 H200／8 CPUs，五組同時執行最多 40 H200／40 CPUs；仍不新增自動控制器。所有新訓練從 random weights 開始。

## 證據與預算

現有 raw 2B 完整 PPL 22.4035，filtered 2B 為 26.1433；舊 raw 在 1.798B tokens、較早 cooldown 後的固定子集 PPL 為 20.9399，該 checkpoint 完整 PPL 未測。參考 measurements.json 與 training-curves.svg。

原始 raw run 的 horizon 為 2B，在 0.4B 時尚未 cooldown；新候選的 horizon 為 0.4B，endpoint 已完成 cooldown。因此歷史結果可提供長程參考，不能當作同 horizon 的 endpoint control。歷史 raw 在 0.4B 也沒有恰好對應的 evaluation point，不對相鄰 PPL 插值作 ranking。

截至 plan.json 的時間，訓練 ledger 記錄 120.140278 H200-hours，前處理 25.326111，合計 145.466389；訓練 ledger 沒有未結算 allocation，但 research-state 保留一筆 16 H200-hour uncertain preprocessing reservation，必須繼續扣住；含此筆上界為 161.466389。三個正式 run 名額已使用完畢。以 192 H200-hours 作保守合計規劃，尚餘 30.533611，這不是重新授權或對課程計帳口徑的定論。執行前仍須重查兩份 ledger、research-state 的未結算 submission intents 和 Slurm 狀態。

| 工作 | 數量及 allocation | H200-hours 上限 |
| --- | --- | ---: |
| seed 42 五組 recipe ranking | 5 × 8 H200 × 45 分鐘 | 30 |
| seed 43 對照與勝者確認 | 延後，未預留；啟用須另有足額預算 | 另需 12 |
| 既有模型與資料診斷 | 分開的 CPU 前處理與 GPU 評估 allocations | 0.5 |
| 本輪已規劃合計 | 五組加診斷，不含延後確認 | 30.5 |

五組主比較屬探索性 short ranking，不以修改 run 標籤取得額外正式 run，也不展開新的 2B 訓練。診斷的 0.5 小時是上限規劃，尚無診斷 worker；若無法在上限內完成就保留 partial 狀態，不挪用 ranking 預算。

目前剩餘上界 30.533611 無法同時涵蓋五組、診斷及兩個 seed-43 jobs（合計 42.5）。因此先只規劃 30.5，seed 確認不預留、不自動啟用。完成並對帳後，只有重新檢查的剩餘上界足以負擔兩個 jobs 與未完成診斷，才可安排確認；不能預先假設 uncertain reservation 會釋放。

## 固定條件

沿用 dolma-raw-baseline-20261008、原始 canonical seed 42 document shuffle 與 append-only packed prefix；split 與所有 50,000 reserved evaluation originals 保持一致。五組 ranking 的初始化 seed 42；若之後有預算進行確認，只改初始化 seed 為 43，不重新 shuffle 資料。

所有候選：Llama 3.2 1B 原始架構與 tied embedding/output、context 8192、8 H200、8 CPUs total、每 rank microbatch 1、accumulation 2。每 update 131,072 non-padding inputs；3051 updates 共 399,900,672 inputs，採不超過 0.4B 的完整 batches。無 padding tail、重複或跳過更新，才算完成。

Muon momentum 0.95、Nesterov、5 次 Newton–Schulz、adjust_lr_fn=original；auxiliary AdamW LR 3e-4、betas 0.9/0.95、epsilon 1e-8 不变。Embeddings/output 是同一 tied 參數，不拆成兩組。所有候選 warmup 170 updates；A/B/C/BC 的終點 LR 為 peak 的 10%，E 為零。所有 optimizer groups 共用排程 multiplier。

每 305 updates 評估固定 first 1000 subset documents，並評估 endpoint。每組以全新 W&B identity 寫入 lab5-training-llama。保留完整 required metrics；global gradient norm 在 clipping 前量測，throughput 排除 evaluation。使用 final_only checkpoint policy，ranking 採 final weights，避免用不同 token 數的 best checkpoint 比較。

## 五組主比較與一組待用配置

| 候選 | Muon LR | Muon WD | WSD decay | 驗證問題 |
| --- | ---: | ---: | ---: | --- |
| A original control | 0.02 | 0.1 | 最後 20% | 同 horizon 的原配方 baseline |
| BC combined | 0.02 | 0.01 | 最後 40% | B 加 C 是否優於個別改動 |
| B earlier decay | 0.02 | 0.1 | 最後 40% | 與 A 比較，測原 decay 下的 cooldown 效果 |
| C weaker decay | 0.02 | 0.01 | 最後 20% | 與 A 比較，測原 cooldown 下的 decay 效果 |
| E hold then zero | 0.02 | 0.1 | 20% 降到 10% LR、10% 平台、10% 降到零 | 與 B 比較分段 cooldown 整體效果 |
| D deferred lower update LR | 0.01 | 0.2 | 最後 20% | 固定每步 decay 係數後，較小 Muon 更新是否改善 |

A-control 已加入 ranking。BC 的內部 candidate ID 與配置檔沿用 A-combined，兩者是同一組；與 A-control 的 run ID 不同。D 是未納入本輪 allocation 預算的待用配置，不執行、不參與 promotion。D 將 LR 減半、WD 加倍，使其每步 lr × wd 與舊 A-control 完全相同；只改較小 LR 而不調 WD，會同時改動 decay，無法分辨原因。這個比較隔離 Muon update amplitude；auxiliary AdamW LR 保持不變。

B 同時改變兩個 optimizer 的 LR 歷程，也改變累積 decay，因此它回答整個排程的總效果，不能把差異全部歸因於 Muon 更新或單獨的 decay。

依使用者最新授權，A-control、B、C、BC、E 平行提交，不等待 A-control endpoint。每組獨立完成 accounting；任何未完成候選排除 ranking，不縮短 horizon、不自動續跑或重試。45 分鐘包含初始化、evaluation、checkpoint、export 與 stopping margin。3051 × 實測 update 約 0.673 秒，純 update 約 34.2 分鐘；這是容量估計，不是完成保證。

## E：先 cooldown、維持，再降到零

E 與 B 保持同一 peak LR、Muon WD、170-update warmup 及 stable 結束位置。以整個 3051-update horizon 為分母分配階段，百分比含 warmup，整數 rounding 如下：

| 階段 | horizon 範圍 | zero-based update indices | Muon LR |
| --- | --- | --- | --- |
| warmup 與 stable | 前約 60% | 0–1830（warmup 0–169） | warmup 後維持 0.02 |
| 第一段線性 cooldown | 約 60%–80% | 1831–2440 | 0.02 → 0.002 |
| 低 LR 平台 | 約 80%–90% | 2441–2745 | 0.002 |
| 第二段線性 cooldown | 最後約 10% | 2746–3050 | 0.002 → 0 |

Auxiliary AdamW 同步按相同比率從 3e-4 降到 3e-5、維持、再降到零。最後一個 LR=0 的 batch 仍計入實際 processed input tokens，不省略以免更改 matched endpoint。

E−B 的 endpoint NLL 比較回答這整套排程是否改善：兩者同在約 60% 開始 cooldown，但 B 在剩餘 40% 連續下降至 10% peak，E 先較快下降、維持、再歸零。低 LR 平台與歸零同時改變，累積 LR/weight decay 也不同；一組無法單獨辨識平台期的因果效果。若後續需要拆解，另設相同分段但最後維持 10% 的對照，須另有 allocation 預算。本輪 E 不納入原四組的 factorial contrasts，也不增加該拆解組。

## Ranking 與 seed 確認

主要指標為 endpoint 的 total held-out NLL / total valid prediction tokens。只納入完整 3051 successful updates、399,900,672 charged=confirmed inputs、finished cooldown、finite evaluation、zero skipped updates 且 Slurm accounting 未超限的候選。比較 endpoint，不比較最佳瞬間或不同 tokens 的 PPL。

先比較 B、C、BC、E 相對 A 的 endpoint NLL。只有相對 A 改善至少 0.03 nats/token 的候選，才值得後續 seed 確認；在合格候選中選 endpoint NLL 最低者。差距小於 0.005 nats 則依 B、C、BC、E 的預先順序處理近似平手。最後三個 evaluation points 用於觀察反彈和變異，不替代 endpoint 排名。

若有 qualifying winner，且 fresh accounting 顯示足以負擔兩個 jobs，才啟用 seed 43 的 A-control 與該勝者；缺乏預算就停在 seed-42 探索結果。勝者在兩個 seeds 都必須比 A 好，平均改善至少 0.03 nats/token，才稱為配對確認的改善。門檻是實用改善幅度，不是統計顯著性聲明。若順位反轉或達不到門檻，保留不確定，不自動添加 seeds。

A/B/C/BC 四組完整時，以每個 endpoint 的 per-token NLL 計算：

- cooldown 平均 loss 變化：((L_B − L_A) + (L_BC − L_C)) / 2。
- decay 平均 loss 變化：((L_C − L_A) + (L_BC − L_B)) / 2。
- 交互作用 contrast：L_BC − L_B − L_C + L_A。負值表示 BC 的 NLL 改善超過兩個個別改善的加總。

以上是同一初始化 seed、0.4B horizon 的描述性 contrasts，不是具重複樣本的 interaction 統計估計。B/BC 的 cooldown 會同時改變兩個 optimizer 的 LR 歷程與累積 decay，結果解釋為整體排程效果。這輪 short ranking 不替代 medium interaction validation，也不新增正式 run。

0.4B 的勝負仍可能無法轉移到 2B；短程結果不能宣稱已達完整 reserved-set PPL <20。將來若允許更長比較，必須重新規劃其 run 名額、aggregate accounting 和從 random weights 開始的 matched horizon。

## 資料與數值診斷

以下只載入既有 export 做 evaluation，不載入 checkpoint 進行訓練，且只在 Slurm compute nodes 上執行：

1. 用同一固定 first 10,000 reserved documents，比較 raw 2B 與 filtered 2B；按原始 source 聚合 NLL sum 和 prediction-token count，報告每個 source 的 NLL、tokens、documents，以及對總 NLL 差距的貢獻。按 token 聚合，不平均 source PPL。另保留既有 first 1000 的結果，避免將兩個 subset 混用。
2. 在獨立 CPU 前處理 allocation，依實際 consumed packed prefix 的 document_spans 計算 source input-token 比例、unique-document 數和 document-boundary 比例；跨 sequence 的 span 不重複計入同一段 tokens。只讀原始 split ranges 與既有 artifacts，不改 split、selection、packing 或 prefix。CPU-only 算法最多用既有一 H200 exception，8 CPUs total，不占 8 GPUs。
3. 在 GPU allocation 上計算兩個 final exports 各層 weight RMS、embedding/norm RMS 和有限性；使用固定小批次做 diagnostic forward/backward，量測分組 grad norms、QK logits 和 logits range，隨後丟棄梯度，不 optimizer.step。這些 endpoint 統計只能形成下一輪假說；要測歷程 update RMS，需要之後另行實作 logging，不能假裝現有 logs 已有此指標。

若 source-wise NLL 能解釋 filtered 的主要劣勢，再設計只去重、溫和篩選與 source mix 的 data ablation。本輪不新建資料準備，不把重新下載或 tokenization 成本當作免費。

## 執行與檢查

配置位於 configs/training-diagnosis-20261010。seed 42 的 A-control、B、C、A-combined、E-hold-zero 五份配置加入 ranking。seed 43 的檔案是延後的模板，只允許在預算充足時選 A-control 加一個合格勝者。D 不提交；十二份配置不代表十二個 jobs 都可執行。

在 repository 執行 `python3 slurm/plan_training_diagnosis.py` 可讀取計畫並檢查配置、source hashes 與 ledger，將 unresolved preprocessing 預留納入上界並列出既有 submission wrapper 的命令；此工具沒有提交、reservation 或寫入共享 state 的能力。它列出的命令供檢查，目前已有使用者的平行 launch 授權。提交前重新執行舊 uncertain reservation 的 metadata recovery；未找到對應 job，繼續完整保留 16 H200-hours，且包含此筆後仍可負擔五組；不得假設這筆預留已取消。確認階段另加 `--winner A-combined` 等已選第一名；工具驗證五組 endpoint 完整且對帳、勝者達門檻並有足額剩餘預算後，只列對照與勝者的 seed 43 命令。

沿用 slurm/submit_training.py pilot 和 slurm/train.sbatch。既有 script 使用 account ACD115198、partition 8gpus、一 node/一 launcher task、8 CPUs total、200G memory，wrapper 覆寫 8 GPUs 與 45-minute limit；training-common.sh 檢查實際 GPU 型號是 H200，初始化 software environment、cache/storage 與 accounting 記錄。logs 在 /work 下建立後才提交；所有大檔仍在 /work/asciibase64/Llama-3.2-1B-training。

執行前比較 pinned source hashes、preparation identity、calibration/environment 身分、確定 raw buffer 足夠，並重查未結算與已 dispatch 的 reservations。來源或環境變更需重新評估 validation 成本。所有 uncertainty 保留預算並 recovery，不重複 submit；所有 job 結束後用現有 accounting wrapper 對帳，沒有自動 resume、requeue、HF upload 或 cleanup。

本次 static checks 驗證所有配置與 horizon、tokens、eval interval、D 的 decay 係數，並檢查既有 shell syntax。37 個純 CPU 輕量單元測試通過，包含 E 的階段邊界、3051-update rounding、終點零 LR、無效配置及舊排程相容性；模型、來源診斷與 Slurm runtime checks 保持未執行。

## 平行提交紀錄

本輪從固定的小型程式／配置快照提交，快照不含資料、模型、憑證或大型檔案；避免等待中的不同 jobs 讀取不同程式版本。launch.json 記錄 snapshot hashes、提交命令、job IDs 和狀態；使用現有 shared-ledger wrapper 註冊每筆 allocation。seed-43 確認與診斷 jobs 均不在此次提交範圍。

### 本次啟動修正

首批 jobs 522525–522529 因快照漏掉 third_party，在模型初始化前失敗／停止；已對帳 0.353333 H200-hours，未處理 training tokens。失敗紀錄及原快照保留。

launch-source-v2 補齊原套件並靜態驗證本地 import closure。重新啟動沿用原 run IDs、configuration 和 ledger，將失敗 allocation 納入各組原 6 H200-hour 上限；A/C/BC/E 的新 allocation 各 44 分鐘，B 為 43 分鐘，token horizon、warmup、cooldown 與比較條件不变。使用現有 resume submission 入口保留 logical-run accounting；重啟前確認每組沒有 initialization、assets、checkpoint、token journal 或 W&B identity，沒有載入 checkpoint，模型仍從 random weights 建立。B 的首次作業尚未建立配置，由相同 ledger configuration 建立 resolved-config 後啟動。這次是修正提交錯誤所需的一次未初始化重啟，不授權之後的 checkpoint continuation 或自動重試。

修正後 jobs：A 522539、B 522540、C 522541、BC 522542、E 522543；均無 dependency。新 allocation 預留合計 29.2 H200-hours，加首批已測 0.353333，本輪訓練預留／已測上界為 29.553333；診斷 0.5 H200-hours 仍僅規劃，未提交。

B 的 job 522540 在 25a-hgpn151 持續等待 NCCL 初始化，未建立模型／checkpoint／journal；停止並对帳 175 秒、0.388889 H200-hours。透過正式 shared-ledger wrapper 的新增 --exclude 節點選項，將同一 B logical run 提交為 522553，排除 25a-hgpn151；訓練程式及配置與其他組完全相同，新 allocation 40 分鐘，累计保留原 6 H200-hour 上限。命令測試使用 mock subprocess，不呼叫 Slurm，既有 submission recovery 與新增 CLI 共 20 個輕量測試通過。

目前 active jobs 為 A 522539、B 522553、C 522541、BC 522542、E 522543。兩批失敗／等待成本累計 0.742222 H200-hours，active allocations 預留 28.8；本輪訓練上界 29.542222，加診斷設計預留 0.5 共 30.042222，仍在原 30.5 規劃內。所有歷史成本與 16 H200-hour uncertain preprocessing 預留繼續保留。

## 本輪觀測結果

四組完成 3051 updates／399,900,672 training input tokens；B 因收尾預留在 3023 updates／396,230,656 tokens 停止，worker exit 75 轉為 Slurm FAILED，已完成 checkpoint、evaluation、export。B 不符合 matched endpoint，排除原訂完整排名，四組 factorial contrasts 也不計算。固定 1000 documents 的 PPL：A 30.917545、B（partial）29.372632、C 46.390261、BC 42.746797、E 28.683031。E 在完成組中最低，相對 A 減少約 7.23%，NLL 改善 0.075018 nats/token；單 seed、0.4B、subset 結果不代表 2B 或完整 reserved-set 表現。全數 Slurm allocation 已對帳，本輪含所有失敗／等待成本共 24.362222 H200-hours，無 jobs 執行，無 seed-43、續跑或新訓練提交。詳見 results.json。
