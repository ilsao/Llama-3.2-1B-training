# 歷史 PPL 證據與下一輪方法實驗

> 最新執行方式已依使用者指示改為[兩階段、每階段兩個平行 sbatch](two-stage.md)：先選 causal mask，再於勝出 mask 比較 PE-only／NorMuon-only。四個 job 各120分鐘，總上限64H，包含內嵌preflight。下文原四組設計與70/72H估算僅保留為設計歷史。

本輪最值得驗證的新方法是**文件內 attention**、**獨立 Polar Express／NorMuon**與**最後階段的權重平均**，其次是**按 token 校正 weight decay**。SOAP 可作為有明確成本門檻的 optimizer 探索。歷史結果同時支持保留 Muon、residual-half 初始化與較長 cooldown，並補齊 batch × cooldown 的缺失對照。使用者要求加入獨立 PE 與 NorMuon 後，第一輪已細化為四個 training arms；見 [第一輪具體設計](first-round.md)。

這是一份研究設計與靜態診斷報告。已核對 summary、逐步 metrics、完整評估、訓練程式、實驗 controller 與 Slurm accounting；新方法尚未取得 GPU 實測結果。新配置是研究規格，未提交作業、預留算力或啟用自動控制器。

## 可比較的實測結果

下表五個 run 均從相同 seed42 residual-half 隨機初始化開始，使用相同 raw prefix、8192 context、3,000,107,008 non-padding inputs。完整評估都是原保留集 50,000 documents、76,930,208 prediction tokens；資料 fingerprint 與初始化 signature 一致。PPL 由總 NLL／prediction tokens 重算通過。

| Run／Slurm job | 配方 | 1k subset PPL | 完整 50k PPL | 實際 H200-hours |
|---|---|---:|---:|---:|
| raw-filtered-3b raw／552612 | BS16，單段 60% tail | 15.53677 | 15.57994 | 34.9178 |
| raw3b-sota／552679 | BS16，5/6＋1/6 resplit，每段 60% tail | 15.22386 | 15.27773 | 34.7133 |
| raw3b-sota-wsd20／553634 | BS16，resplit，每段 20% tail | 15.83468 | 15.91230 | 34.5844 |
| raw3b-bs64-wsd40／556710 | 50% tokens 時 BS16→64，token clock，40% tail | **15.21324** | **15.21170** | 31.9467 |
| raw3b-bs64-stepwsd40／558510 | 同 batch windows，update clock，40% tail | 15.41073 | 15.37130 | 31.7533 |

H200-hours 含各 logical run 的失敗 allocation；例如 wsd20 含 553561 的 20 秒失敗成本。Slurm 確認五個最終 job 均 COMPLETED。部分 status.md 仍寫 RUNNING，不能用作終態證據。觀測時共享 experiment ledger 有 222 筆 job、13 個 formal identity，實測與保留合計同為 **891.8678 H200-hours**；這是歷史支出，不是新實驗可用餘額。

最新最佳相對舊 SOTA 的完整 PPL 改善 **0.4322%**、allocation 成本下降 **7.9700%**；單 seed 尚不足以宣稱穩定的品質勝出。1k 子集差距只有 0.01062 PPL，而完整集差距為 0.06603。完整 NLL 改善中 **85.12% 來自長於 4096 tokens 的文件**，因此只看短文件或 aggregate train loss 會漏掉重要差異。

資料來源：[彙整證據](evidence.json)、[原始評估曲線表](eval-curves.csv)、[完整曲線與末段放大](eval-curves.svg)。每個 run 的 summary／metrics／full-evaluation 路徑與 SHA256 都記在 evidence.json。

## 歷史實驗真正支持什麼

1. **cooldown 是大效果，batch 的獨立效果仍待分解。** BS16 的 60% tail 完整 PPL 15.2777，20% tail 為 15.9123。較早 700M 實驗也是 60% 優於 40%、20%、10%。但最佳 15.2117 同時改了 batch 與 tail，缺 BS16＋40% 和 BS16→64＋60%，不能直接歸因於較大 batch。
2. **早期 PPL 排名不能代替終點排名。** update-clock run 在約 1.5B 的 subset PPL 19.7149，優於 token-clock 的 25.8295；終點卻反轉為 15.4107 對 15.2132。應完成相同 token horizon 與 cooldown 再排名。WSD 論文也將 stable 與 decay 的角色區分，支持檢查完整軌跡；它不證明本專案哪個 tail 最佳。[WSD 原論文](https://arxiv.org/abs/2410.05192)
3. **clock 比較有 warmup 混雜。** formal12 的第二段 warmup 是 170 個 BS16 等價單位，即 22,282,240 tokens；formal13 是 170 次實際 updates，BS64 區約為四倍 token。兩者也改了第一段 decay 起點。不能把 15.3713 全歸因於 decay clock。StepScheduler 目前直接以 update index 呼叫整個 schedule，改 `warmup_clock` 字串不會自動產生混合時鐘。
4. **BS 放大牽動 clipping、momentum 記憶與 decay。** 最新最佳在 1.5–2.5B 的已記錄樣本中，preclip norm 中位數 0.2942，超過 clip=.5 的比例 3.7%；BS16 舊 SOTA 分別是 0.6796、99.9%。這是每十次 update 的抽樣統計，不能當成所有 updates 的精確比例。相應吞吐約 245k 對 207k tokens/s。高 clipping 頻率本身不證明 clipping 有害，舊 700M 對照仍以 .5 優於 1、2、off。
5. **NorMuon 與 beta2 token scaling 目前沒有勝出。** 同 1,000,079,360 inputs，control PPL 20.4452、NorMuon＋Polar Express 20.6598、提高共同 LR 至 .01/.0003 為 20.5143。50% tokens 放大 batch、固定 beta2 得 19.9787，beta2=.95^4 得 20.1214；較晚切換亦輸 control。這是單 seed、1k subset、20% tails 的結果，不能排除其他 horizon 的 NorMuon，但不值得原樣重跑。
6. **獨立 AdamW LR 曾有跨 seed 證據。** 388.76M 的 short-baseline PPL 30.1530，AdamW LR×2 為 28.1088；seed43 對照 29.9137→28.1231。相反，單獨較低 WD=.03/.05 為 32.2572／31.4110，沒有改善。這些是較舊的 Gaussian／batch16 配方；新 residual-half＋動態 batch 應測轉移性，不能直接宣稱還會有相同收益。
7. **EMA 已經是 late-start，不能重新包裝成新發現。** 舊實作從 80% tokens 開始，short-baseline raw PPL 30.1530，EMA .99/.999/.9995 是 30.1864／31.3653／32.0344。另一次 EMA proxy 推出的 LR law 沒通過真 WSD 檢查。它們反對直接沿用舊 EMA，仍未回答「固定 token 時間尺度、只平均最後 phase」的問題。
8. **資料過濾尚無正向匹配證據。** 舊 388.76M filtered PPL 34.6743，raw control 30.1530；原定恢復來源 token 配比的 3B filtered arm 已取消，沒有結果。無法區分品質過濾與來源比例改變。資料設計值得研究，但不能把舊 filtered 當作有效增益。

來源：[1B batch 研究](../batch-normuon-1b-20261016/report.md)、[horizon controller](../horizon-wsd-20261014/controller-state.json)、[初始化與配方 controller](../init-recipe-20261014/controller-state.json)、[LR calibration](../lr-transfer-horizons-20261015/calibration-result.json)。新 LR law 僅用 200M／400M／600M 單段 WSD 擬合，3B 的雙階段／動態 batch 尚屬跨設定轉移；原論文支持 horizon-dependent LR，沒有給這個設定的最優值保證。[LR horizon 論文](https://arxiv.org/abs/2409.19913v3)

## 新方法 A 文件內 attention

**假說：** 訓練時無關文件的前文造成干擾，而且與獨立文件評估不一致。保留完全相同的 token stream、labels、8192 packing 與位置設定，只讓 token attend 同一文件片段內的前文，可能降低 PPL，並減少 attention 工作量。這是訓練 attention 方法改變，模型參數與 HF 架構保持相容。原研究發現 intra-document causal masking 有益；不能預設其幅度能轉移到 Dolma／本配方。[Sequence Composition 原論文](https://arxiv.org/abs/2402.13991)

**現有實作證據：** `raw_baseline.py` 明記 cross-document causal attention；完整訓練記錄的 varlen calls 為 0。`dataset.py` 已把 `document_spans`（row_index、document_start、sequence_start、length）寫入 packed rows，無需重排文件或重做 split。現有 FA3 varlen 只處理 right padding，尚未按文件 span 切分，因此這不是開一個現成設定就完成。

**驗證設計：** 1B horizon 做 dense causal vs intra-document 兩個 fresh runs。兩者從同一 seed 初始化、讀同一 packed prefix，使用 token-clock 40% tails、BS16→64 at 50%、其餘同 latest recipe；這個 1B control 必須新跑，不能借用舊 20% tail control。另以相同訓練軌跡收集方法 B 的平均權重。

**先做 correctness gate：** 在 Slurm 中以小型人工序列及真實 packed rows，把 varlen 的 forward、loss、QKV gradient 對照明確 block-diagonal causal SDPA；擾動前一文件不應改變後一文件輸出；涵蓋跨 sequence 延續片段、單 token span、padding、BOS/EOS 出現在正文與 GQA。邊界只取 document_spans，不掃描 token ID 猜邊界。保留原 labels，確認 input tokens、valid predictions 與 dense control 完全一致。位置 ID 先保持原值，以免同時改位置策略。

**判定：** matched-token NLL 改善、短／長文件分桶與文件內前 128 個 prediction 的 NLL，外加 allocation tokens/s。若僅速度增加，將它記為效率方法，後續用固定算力比較可多處理的 tokens，不能宣稱相同 tokens 的品質改善。若 1B 有訊號，再 fresh 3B 對 latest control 驗證。

## 新方法 B 最後 phase 的有限窗口平均

**假說：** 舊 EMA 的 update-based 記憶長度、跨 reheating 的權重混合及滯後，可能掩蓋平均的好處。LAWA 在 LLM pretraining 中使用同一軌跡的多個權重，且高 LR 情境收益更明顯；本案 low-LR 終段未必受益，應直接與同一軌跡 raw endpoint 對照。[LAWA 原論文](https://arxiv.org/html/2306.03241v2)

在方法 A 的兩個 fresh 1B runs 中，各輸出三個 endpoint：raw；從 95% tokens 初始化的 token-EMA，半衰期為 horizon 的 1%；最後 2% horizon 內、等 token 間隔的五個權重之等權平均。EMA 每次更新用 `beta=2**(-delta_input_tokens/half_life_tokens)`；平均不回寫訓練權重。LAWA 取 98%、98.5%、99%、99.5%、100% 最近的已完成 batch，記錄實際 tokens，保持 batch windows 相同。兩種平均都只含最後 phase。

兩個 FP32 shadows 約增加 9.9 GB／rank，需實測峰值及 checkpoint overhead；用 streaming mean 避免存五份完整模型。所有 tied parameters 只平均一次；輸出本身仍是標準 Llama。驗證 raw trajectory 不因平均而改變、state 能精確保存、HF reload 的 eval 一致。

既有兩個最佳 3B run 都只留終點 checkpoint，現在無法離線重建其平均軌跡。此方案的節省是附掛在已需要的 fresh 對照上，不是從現有 checkpoint 再訓練。平均後進一步訓練也不屬於本方案。

## 新方法 C 按 token 對齊 decoupled weight decay

**已完成的靜態診斷：** 用 formal12 的實際 scheduler 與 batch-window 函式重建 14,312 次 updates，endpoint 為 3,000,107,008 inputs，並通過 1,432 個已記錄 LR 的比對。Muon `prod(1-lr*wd)` 為 0.001030945；相同 token LR 曲線、全程 BS16 的對應乘積是 0.000216375。前者的累積 log-shrink 只有後者的 81.50%。這是 decay 項單獨的乘積，**不是模型實際 weight norm 的預測**，因為每次更新還有梯度項。計算與來源 hash 見 [decay audit](decay-audit.json)，可用 `python3 experiments/ppl-next-20261016/audit_decay.py` 重現。

**假說：** batch 的優勢部分來自每 token 的 decay 較少；因此固定名義 WD 的 batch sweep 並未隔離梯度噪聲機制。對一個包含 k 個 BS16 單位的大 batch，定義該區間虛擬 BS16 衰減乘積 `d=prod_j(1-lr_j*wd)`，大 batch 直接對權重施加 d，再做原梯度更新。若當步 LR 非零，可等價為 `wd_eff=(1-d)/lr_current`；零 LR 仍使用獨立乘法 d，不做除零。這只對齊 decay，不能讓不同 batch 的 optimizer trajectory 相同。

固定 token40 schedule，做三個 1B arms：全程 BS16／原 WD；BS16→64／原 WD；BS16→64／token-normalized WD。第二臂復用方法 A 的 dense control，僅補第一、第三臂。比較 batch gain 是否在對齊 decay 後保留，並記錄每層 norm、完整 cumulative log-decay、update/weight ratio。固定 AdamW beta2=.95、Muon momentum=.95；不在同一 arm 再改 momentum。原 embedding/head/norm 無 decay，保持如此。

此方法是由本專案診斷導出的實驗假說，不是論文已證明的改進。Muon 文獻強調 weight decay 與更新尺度的重要性，可作研究動機，不能當成本方案結果。[Muon 原論文](https://arxiv.org/abs/2502.16982)

## 新方法 D SOAP 作為次優先 optimizer 探索

SOAP 在 Shampoo eigenbasis 裡做 Adam 類更新，原論文有 360M／660M LM pretraining 證據；主要對照是 AdamW／Shampoo，不能推論會勝過此處已調整過的 Muon。[SOAP 論文](https://arxiv.org/html/2409.11321v2)、[作者實作](https://github.com/nikhilvyas/SOAP)

先做包含單卡數值檢查與八 rank 更新一致性的 runtime gate，再用與主對照相同 parameter routing：只把 hidden matrices 換 SOAP，tied embedding/head 與 norms 保持原 AdamW。初始驗證設定為 betas=(.9,.95)、shampoo_beta=.95、eps=1e-8、precondition frequency=100、max_precond_dim=2048、merge_dims=False、precondition_1d=False、normalize_grads=False。2048 cutoff 會跳過 8192 維方向，屬於部分／單側 preconditioning，不能稱為完整 SOAP 重現；作者程式確有該 cutoff 行為。固定作者 commit 與檔案 hash 後才執行。

用矩陣 LR {3e-4, 1e-3, 3e-3} 做各 20 分鐘穩定性 screen，auxiliary LR 固定。screen 只排除 NaN、OOM 與明顯無法執行的設定，不以未完成 cooldown 的 loss 宣稱勝者。最多兩個存活候選進 1B 完整 horizon；若吞吐不足以在 120 分鐘內完成，改為所有 comparator 共同較短的預先固定 horizon，或停止，不能讓 SOAP 少看資料還硬排名。proposal 的效用門檻為 NLL 降低至少 .01，並且 matched-token allocation 成本不增加超過 20%；門檻是本研究決策，不是文獻保證。

## 其他方向與優先序

| 方向 | 現有證據與下一步 | 優先度 |
|---|---|---|
| Batch × tail | 3B 的 2×2 表缺 BS16＋40%、BS16→64＋60%；補兩個 fresh runs 即可分解主效果與交互作用，token warmup 固定 | 高，作為方法研究的配方基線 |
| 獨立 auxiliary LR | 新配方固定 Muon，AdamW LR×{.5,1,2}；1B 三臂，完成 cooldown，再以 seed43 重複 control／winner | 中高，有歷史跨 seed 證據 |
| 混合 warmup／decay clock | 補 token decay＋update warmup、update decay＋token warmup；凍結相同 batch windows，形成 2×2 | 中，偏因果解釋 |
| 增加 fresh tokens 至 4B | raw buffer 已發布約 6B unique tokens；先 4B、同一最佳方法、重新隨機初始化。LR 表僅作起點 | 中高，偏最終分數 |
| 來源配比保持的過濾 | 先在 canonical train 內建立 unique-token availability 表，分來源等 token 配額過濾；樣本不足就縮短共同 horizon。用 raw、原 filtered、配比恢復 filtered 三臂分解效果 | 中低，準備成本高 |
| DoReMi | proxy/reference 額外成本與 mixture 變更較大，先完成來源分桶診斷；不直接借用別的模型學得的 weights | 低 |
| MTP、QK-norm、untied head、短 context curriculum | 涉及架構或固定 8192 限制；不放入本輪合規主線 | 暫不採用 |
| 原樣重跑舊 NorMuon＋PE 組合、全域降 WD、取消 clipping | 現有對照沒有正訊號 | 低 |

DoReMi 的證據支持資料混合比例值得學習，但其 proxy Group-DRO 流程需要額外模型，並非免費替換 sampler。[DoReMi 原論文](https://arxiv.org/abs/2305.10429) DCLM 支持品質過濾研究，但其 downstream 成績不等於原 Dolma 保留集 PPL 的保證。[DCLM 原論文](https://arxiv.org/abs/2406.11794)

## 最小可判讀的第一輪

建議首先執行四個 1B fresh runs：dense＋NS5 control、intra-document＋NS5、dense＋PE5、dense＋NS5＋NorMuon normalization；每個同時計算 raw、token-EMA、有限窗口平均，共十二個可比較 endpoints。PE 只替換正交化子程序；NorMuon-only 只加入逐 output-row 的二階矩 normalization 並保持原更新 norm，兩者均保留 native Muon 外層更新。舊 NorMuon＋PE 沒有隔離各自效果，也不能充當目前40% tail配方的組合對照。先看三個 raw 主效果，再看平均相對各自 raw；需要組合時才補對應交互作用格。詳細 PE／NorMuon 公式、parity gate 與決策規則見 [第一輪具體設計](first-round.md)。

| 項目 | 配置與上限 | 保留上限 H200-hours |
|---|---|---:|
| correctness／memory／throughput gate | 8 H200，8 CPU，15 分鐘，含八 rank 更新核對 | 2 |
| 獨立 PE correctness gate | 8 H200，8 CPU，15 分鐘，含 native NS parity | 2 |
| 獨立 NorMuon correctness gate | 8 H200，8 CPU，15 分鐘，含 row-axis／norm／state 核對 | 2 |
| 1B dense control | 8 H200，8 CPU，120 分鐘，fresh seed42 | 16 |
| 1B intra-document | 同上 | 16 |
| 1B PE-only | 同上，保持 dense attention | 16 |
| 1B NorMuon-only | 同上，保持 dense attention 與 NS5 | 16 |
| 9k 確認評估，條件式 | 8 H200，8 CPU，15 分鐘；鎖定 candidate 與 control | 2 |
| seed43 確認，條件式 | control 與已鎖定 winner，各 120 分鐘 | 32 |
| 3B 轉移，條件式 | control 與 winner，各 6 小時、48H，fresh | 96 |

第一個篩選決策點需預留 **70H**，加條件式 9k 確認為 **72H**；seed43／3B 不自動啟動。兩種平均及 1k endpoint 評估都包含在 training allocation 上限內。每 training allocation 預留最後 20 分鐘給評估／輸出／停止餘裕，gate 驗證餘下時間能完成共同 horizon。若第一輪沒有正訊號，保留結果並轉向 token-normalized WD 或已缺的 batch × tail 對照，而不是自動跑全部方法。

第一輪共通設計：1,000,079,360 inputs（7630 BS16 等價單位）、phase split=6358／7630、第二 peak=.1×、170 等價單位 warmup、40% tail 的 ramp/hold/final 比例 4:1:1、hold LR=.2×；Muon peak=.008614891542046706、AdamW peak=.00025844674626140116、WD=.1、clip=.5。BS16 用 8 ranks×microbatch1×accum2，於 500,039,680 inputs 後改 accum8。LR 是既有 horizon 表的起點，並非本新設定的已驗證最優值。所有 event 與 endpoint 用整個 BS16 等價單位截短 accumulation，實際 update 數由新 planner 計算並鎖定，不沿用舊 20% tail 的 update 數。

其他模組各自的額外保留上限：C 補兩臂 32H；3B batch×tail 補兩臂 96H；auxiliary LR 三臂 48H（若 control 完全相同可復用）；SOAP runtime 2H＋三個 20min screen 8H＋最多兩個 1B arms 32H，合計 42H；4B 單 run 建議 7h／56H 上限。這些是可選研究模組，不是整套已授權預算。4B 成本以現有 3B 實測線性估計約 42.6H，仍須加 overhead 與停機餘裕；6B 在 64H 內過於緊，不宜直接承諾。

## 如何判定改善與避免假陽性

- 主指標固定為相同 token endpoint 的 held-out mean NLL，PPL=exp(mean NLL)；每個方法也報實際 allocation H200-hours。使用現有固定 first1000 documents 作開發集，不能拿當前 train loss 或不同 token checkpoint 排名。
- 第一次篩選要求 ΔNLL≤−.005（約 .5% PPL 改善），且在原固定 10k subset 的其餘 9000 documents 確認同方向。若該 9000 過去曾用於選擇，需標記為重用 validation，不能稱為全新 holdout。小於門檻者記為接近，主要看效率與 seed43，不硬選最小數字。
- 每 400 actual updates 及共同 10% token knots 評估，另加 phase／batch 邊界和終點。planner 必須再確認 eval interval 不超過實際 planned updates 的 10%。輸出每文件總 NLL／valid tokens，以文件為單位 paired bootstrap；不可把相依的 token 當獨立樣本。分桶看 document length、來源、文件開頭與後段；最終指標維持原 token-weighted aggregation。
- seed43 配對重跑 control／winner，保留 seed42 的完整結果；兩 seed 同方向是重現訊號，不是充分估計所有訓練隨機性的信賴區間。文檔 bootstrap 也不能取代 seed variation。
- 完整 50k 評估只用於鎖定的最終候選，報 final 而非事後 best checkpoint。既有 50k 已被反覆查看，需坦承 adaptive model-selection 的限制；不能改動 canonical split 來製造新的「官方測試集」。
- 新 run 均從隨機權重開始，獨立 W&B identity；不沿用舊 branch checkpoint 訓練。每次 processed batch、skipped update 均計 input token 預算。保留失敗 allocation，無自動 retry／resume。大型處理與 GPU correctness 均經 Slurm，metadata-only 分析才在 login node。

## 可執行性與交付狀態

已完成：五個 3B run 的來源封存、PPL 重算、一致性核對、逐步曲線及 clipping／throughput 摘要、歷史消融重讀、論文與作者實作 survey、weight-decay 算術診斷。重跑 `python3 experiments/ppl-next-20261016/collect_evidence.py` 只讀小型 metadata，更新本目錄證據與圖表。

尚待方法實作與 GPU 驗證：document-span FA3 adapter、phase-local averaging、PE-only／NorMuon-only Muon adapter、token-normalized WD、SOAP adapter，以及它們的獨立 source seal、planner、sbatch 與 submission guard。現有 formal12／13 contracts 鎖定 run/config，不可直接改 JSON 後套用既有授權。這次只完成研究設計；新的正式 run 與超過 120 分鐘 allocation 必須在具體配置完成後確認適用的新增授權。參見 [機器可讀研究規格](plan.json)。
