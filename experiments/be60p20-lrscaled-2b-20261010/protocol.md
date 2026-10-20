# BE60P20：LR horizon scaling 到 2B tokens

使用者明確指定以 BE60P20 配方訓練新的 2B-token run，從 seed 42 隨機权重開始，parent run 只作配置與數據參考，不載入 checkpoint。資料沿用 parent 的 dolma-raw-baseline-20261008、完整 canonical seed-42 split 與 first 1000 reserved evaluation documents；不是改用 filtered training data。

[Scaling Optimal LR Across Token Horizons v3](https://arxiv.org/html/2409.19913v3) §6 建議 ≥760M 模型用 beta=0.32。採用相對縮放 lr_long = lr_short × (D_long / D_short)^(-0.32)，D_short=399900672，D_long=2000027648，factor=0.597438441121。不能由單一短 horizon 識別本模型自己的 exponent，也不宣稱 parent 的 LR 已是 optimum；套用到 Muon 與 BE hold/zero 排程屬這次實驗的 extrapolation。

Muon peak LR 0.0119487688224，auxiliary AdamW peak LR 0.000179231532336；平台分別為 0.00238975376449 與 3.58463064673e-05。保留 WD=0.1、momentum=0.95、betas=(0.9,0.95)、170 warmup updates、8192 context、8 H200/8 CPUs、microbatch 1/accumulation 2、global input batch 131072、原參數分組與 masks。WD 係數固定，沒有反向放大 WD；LR 的縮放會改變 cumulative decay，與論文所稱其他超參數固定的相對 horizon transfer 一致。

共 15259 updates，first cooldown index 6103（約40%），平台 index 12207（約80%），final cooldown index 13733（約90%），最後 update 的 LR=0。停止在完整 batch 首次跨過2B confirmed nonpadding training input tokens，通常2000027648；hard cap同值，尚未處理的 batch 先檢查剩餘額度，skipped/repeated inputs 都記帳。每1525 updates評估，另作最终評估，sum NLL / sum valid tokens 後 exp。

新 W&B identity，project lab5-training-llama。單一4小時 allocation，8 H200共32 H200-hours，300秒 stopping margin加300秒 final evaluation reserve。根據 parent 35.6分鐘/0.4B，預估約178分鐘；不是完成時間保證。原選077節點後已占用，改採同樣近期成功、提交前確認IDLE的25a-hgpn078，一 node/one launcher/8 ranks/one CPU thread per rank。模型與大檔均存 /work/asciibase64/Llama-3.2-1B-training/runs/experiments/be60p20-lrscaled-2b-20261010。

使用者在收到具體配置及自動審核拒絕說明後，明確回覆「可以」，批准這一組作為第四個正式 full run、8 H200／8 CPUs、單次4小時／32 H200-hours例外。原先拒絕紀錄保留於plan.json，現已由此批准解除；例外限精確run/config，其他run限制及歷史成本完整保留。無自動retry／resume。使用者先前不考慮剩餘H200-hours的指示不刪改歷史。此次範圍包含本地HF格式export，不包含HF upload／OJ提交或額外正式run。


51項純函數／metadata／mock submission檢查通過，包含精確第四run授權、禁止resume／改配置／超時／第五run、原三個run限制、token budget及scheduler；sbatch與embedded Python syntax通過。來源快照與job ID記錄於launch.json。

## 已提交

Job523243，2026-10-10 16:16:38 Asia/Taipei 啟動於25a-hgpn078，8 H200／8 CPUs，allocation截至2026-10-10T20:16:38。帳本category full、formal_run_number4、32 H200-hours預留已核對，所有來源hash封存於launch.json；實際cost待Slurm終態對帳。

## 完成結果

Job523243於2026-10-10 19:08:06 Asia/Taipei正常結束，8 ranks exit0。完整15259 updates與2,000,027,648 confirmed=charged non-padding inputs，最後LR=0。固定1000 reserved documents／1,479,302 prediction tokens 的最終PPL=17.7475548804，mean NLL=2.8762477532。HF export及compute-node reload verification完成，未upload；full50000 reserved/OJ尚未評估。

Slurm actual allocation 10288秒（2h51m28s），8 H200，共22.862222 H200-hours，低於批准32上限；已對帳。來源、環境、初始化signature、evaluation身份、token journal和完整endpoint核對通過。完整metadata及最後cooldown評估點見results.json。

## 使用者另行批准的 HF 公開上傳

使用者在確認既有公開 repository asciibase64/Llama-3.2-1B-dolma3_mix 與完整 15 檔案清單後回覆「傳」。上傳 job 523972 於 2026-10-10 20:30:34 Asia/Taipei 正常結束，已發布 revision 38fa239d83ee53409cec8d7b78179915fe4e1bae，保留原 revision 0b54c9d826c765cb0e3d75494b00e898d04f882e。15 個遠端 checksum、匿名 config/tokenizer 載入及三個權重分片公開存取均已驗證。此次上傳 allocation 382 秒、1 H200，共 0.106111 H200-hours，作 publication/validation 費用獨立對帳；未執行額外訓練。完整 receipt、checksums 與資源紀錄見 experiments/hf-be60p20-2b-20261010/results.json。
