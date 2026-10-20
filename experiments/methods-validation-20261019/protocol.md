**Llama 3.2 1B 訓練方法驗證實驗設計**

日期：2026-10-19，Asia/Taipei。第一輪比較現有 NorMuon、Dion3 全量更新與 Dion3 部分更新，辨別「移植差異」和「部分更新」的效果。AdaMuon／SOAP、蒸餾、schedule-free 與 CCE 分成可選模組；單項成立後才研究組合。

交付包括 [機器可讀方案](plan.json)、[18 個實驗組藍圖](arms.json)、[逐項預算](budget.csv)與[靜態檢查結果](design-check.json)。藍圖包含待量測的 LR 單位映射及待實作 adapter，不能直接當作訓練 config 執行。此次設計沒有預留資源或提交作業。

**共同控制與比較單位**

品質初篩沿用已驗證的 1B 實驗批次計畫。它讓每組在同一批次邊界、相同訓練量與完整 cooldown 下結束，避免把穩定學習期和已衰減的模型混比。

| 項目 | 固定設定 |
|---|---|
| 模型 | Llama 3.2 1B，1,235,814,400 參數，原 tokenizer、tied embedding，8192 context |
| 初始化 | residual-half，隨機權重；初篩 seed42，確認 seed43；每個 run 重新開始 |
| 資料 | `dolma-raw-baseline-20261008`，原 canonical split、文件順序與 packed prefix |
| 注意力與 loss | 原 document-span intradoc causal mask；原 shifted labels 與 ignored-label 規則 |
| 輸入終點 | 1,000,079,360 個 nonpadding student inputs，4,775 個更新窗口 |
| 批次 | 8 H200／8 CPU；microbatch1；global BS16 → 64，切換點 500,039,680 inputs |
| 排程 | token clock；phase split 833,355,776；第二峰值 0.1×；各 phase warmup170個 BS16 等效單位、40% tail、4:1:1 ramp／hold／final、plateau0.2 |
| 精度 | FP32 參數／狀態，BF16 autocast；FA3；activation checkpointing |
| 梯度 | 以全 rank 有效 labels 正規化，global clipping0.5，記錄 clipping 前 norm |
| 基準 LR | matrix0.008614891542046706；輔助 AdamW0.0005168934925228023 |
| 參數組 | hidden matrices973,078,528；shared embedding＋norms262,735,872；每個 tensor 恰好一次 |
| 評估 | 固定1k 每400更新、既有 token／phase／batch knots 及 endpoint；候選鎖定後額外9k |
| 每組 allocation | 8 H200／8 CPU、120分鐘上限；包含初始化、評估、輸出與等待；無自動重跑 |

批次計畫中有為精確對齊 knots 而使用的 BS16／32／48 邊界窗口，必須完整沿用，不能只照「16→64」重新估算成另一組步數。[原始批次計畫](../ppl-next-20261017/batch-plan.json)及[控制設定](../ppl-next-20261017/configs/B-seed42.json)的 SHA256 已記錄。

scheduled 方法的 matrix decay 在外層統一施加：`q(t)=1−reference_peak_lr×0.1×schedule(t)`，使用更新前權重，停用 adapter 內的 WD。這使不同 LR 數值單位或校準倍率不會偷偷改變 shrinkage；N0 中心設定仍恢復原本 ordinary WD。此選擇是本研究的匹配控制，不稱為逐字複製作者 recipe。SOAP 的作者實作有不同的 decay 次序，因此要做 wrapper 對照。SF 模組依其演算法在 z 上 decay，獨立說明，不強稱與 scheduled 軌跡等價。

**先驗證程式與成本，才進入品質實驗**

V0 使用小型張量 reference 與新隨機初始化的完整模型，全部透過 Slurm。歷史模型只可用於教師推論／唯讀分析；不載入訓練 checkpoint 作學生初始點。合成張量上的 optimizer state 序列化與 gradient replay，不作模型續訓。

| 檢查 | 做法 | 預先指定通過條件 |
|---|---|---|
| 分組與計數 | 檢查 tied tensor、每 rank 資料 prefix、loss 分母、實際 inputs | tensor 無重複／遺漏；分母和 tokens 對得上；無 eval document 進入訓練 |
| 小張量 reference | 正方形與高／寬矩陣、zero／低 rank／病態梯度；FP64 loss 或高精度 update reference | tiny loss 絕對誤差≤1e−6；適用的梯度 relative L2≤1e−5；zero norm 另以絕對誤差判斷 |
| 八 rank 等價 | 同一確定性梯度序列，比較所有 ranks 狀態與更新；DDP 全模型使用相同 global batch | 結果在預登記容忍度內；不得將 replicated norm 重複計算 |
| 真實尺寸與 BS 切換 | 8192 context，觀察 BS16 與64；含一次跨 boundary 窗口 | loss／gradient／weights／state 全部有限；切換後 state 未重置 |
| 成本 | 初始化、compile、穩態更新與 evaluation 分別計時；CPU/GPU peak memory | 使用 p95 step time 推估完整1B＋固定 overhead＋15%裕量≤110分鐘；另留10分鐘 allocation margin |
| 記憶體 | 所有方法完成 state initialization，SOAP 至少兩次 basis refresh，KD 含師生同時常駐 | 最忙 rank 保留≥10%裝置記憶體；OOM 記為當前實作不可行 |
| 輸出 | 最終權重／token／step 對齊；export 後唯讀 evaluation | 匯出模型的固定 subset loss 在數值誤差內一致 |

完整模型的短期驗證也依實際步數設定每10%至少一次評估，極短檢查每步評估，終點必評；不能沿用1B的400步間隔。

對 CCE 的大尺寸 BF16 檢查，預設 loss 差≤2e−4 nats、梯度 relative L2≤1%、cosine≥0.999。Gram NS 的 BF16 direction 初步門檻為 relative L2≤5%、cosine≥0.995，還需後續長軌跡品質驗證；這不是宣稱更新逐 bit 相等。門檻失敗先檢查實作，不依看到的 PPL 放寬「exact」定義。

新環境、adapter source 與 upstream commit 必須先封存；目前來源仍以論文版本和 URL 指定。若依賴需要升級，全組控制也在同一環境重跑，不將舊結果當作新的隨機對照。原封存 source、資料 preparation 與歷史結果保持不變。

**量測修正納入共同控制**

embedding 更新診斷改為固定 seed 的分層 rows：按 training-only prefix 的詞頻劃為高／中／低頻與未見四層，各64 rows；記錄整列2048維的 norm，不再固定抽第一列前256元素。分層與 row IDs 在校準控制 run 後鎖定，所有方法共用。若某層不足64 rows，取全體並記錄實際數量。

另記錄 hidden-matrix 完整 update RMS／weight RMS、列內變異，以及第0／7／15層固定64維 projection 的 activation covariance 摘要。projection 只用作相關性診斷，不能冒稱完整 Hessian。hook 僅在原 forward 收集，避免 activation checkpointing 重算重複計數。診斷 overhead 必須≤3%；超過時降低所有組的共同頻率，修改後重新封存設定。

W&B 保留 required metrics；KD 的 `train/loss` 是實際混合 objective 的有效-token加權區間平均，同時記錄 `train/ce_loss` 與 `train/kl_loss`。跨方法主要看 held-out hard-label NLL，不直接比較 KD objective 和普通 CE。額外記錄每個群組 LR／decay、optimizer時間、teacher時間、峰值記憶體、sampled rows 與方法特有狀態。

**第一輪 核心三組**

| ID | 組別 | 主要對比 | 要回答的問題 |
|---|---|---|---|
| N0 | 現有 native NS5 NorMuon＋AdamW×2 | 新的共同基準 | 現有方法在新封存環境的結果 |
| D1 | Dion3 NorMuon flavor，f=1 | D1−N0 | 作者實作、momentum、scale 或 kernel 移植造成多少差異？ |
| D025 | 同一 Dion3，f=1/4，error feedback 開啟 | D025−D1，以及 D025−N0 | 部分更新是否有效，而且能否勝過目前方法？ |

關鍵分解是：`D025−N0 = (D025−D1) + (D1−N0)`。D025 只勝過 N0、卻沒勝過 D1，不能把收益歸給部分更新。D1 不必然等同本地 Nesterov 實作；差異須公開保留。f=1 首先對照作者 NorMuon reference，以確認 adapter 正確；另做只替換 Gram NS 的局部更新測試。D1／D025共同固定momentum0.95、row second moment0.95、NS5及原NS係數；API中的momentum語義須對照Dion3，而非沿用Dion2同名參數。[Dion3 方法](https://arxiv.org/html/2608.11612v1)、[作者 API](https://github.com/microsoft/dion)。

LR calibration 每組至多三個倍率 `[0.7,1,1.4]`，每次20分鐘上限、fresh seed42、200,015,872 inputs／1,526步、BS16、warmup50步、單 phase 完整40% tail，固定1k 每152步與終點。新方法先用 training-only gradients 換算 LR 中心；D025 在 D1 單位映射後使用 `1/sqrt(f)` 作初始比例。記錄每類矩陣的實際 update RMS，不能只比較數字 LR。

校準只決定數值穩定與可用尺度，不作最終方法排名；它與1B的 horizon 不同。完整且有限的校準 run 中選最小 endpoint NLL，差距≤0.002時選靠近倍率1的設定。最佳值落在邊界時標記「校準範圍可能不足」，不得用負面結果否定整個方法；也不自動增加試跑。1B 開始前鎖定設定。若最慢方法在20分鐘內連共同校準 horizon 都無法完成，須在看到任何校準 NLL 前統一縮短該模組的 horizon，或判為此預算不可行。

**第二輪 幾何模組**

A0 為新的 NorMuon 控制；A1 為 AdaMuon v3；S1 為完整 SOAP，皆只替換 hidden matrices，aux AdamW 保持一致。每組各三次 calibration 後跑匹配1B。AdaMuon 固定 sign transformation、逐元素二階矩與 RMS alignment，momentum／second moment先取0.95。SOAP 固定 betas(0.95,0.95)、precondition頻率10、max dimension10000、merge_dims=false、normalize_grads=false、bias correction=true。[AdaMuon v3](https://arxiv.org/html/2507.11005v3)、[SOAP 實作](https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py)。

SOAP 第一次梯度若只初始化 basis，該批輸入照計，不能補跑一批。若完整 SOAP OOM 或無法在兩小時完成1B，結論是此配置在本預算不可行；分塊、單側或只改 MLP 是新的變體，另立組別，不悄悄替換。

只有 A1 勝出，才追加拆解 sign transformation 與逐元素二階矩；只有 S1 勝出，才測單側／稀疏 refresh 等降成本變體。activation covariance 診斷若突出，可將 Newton–Muon 作下一個 geometry 候選，使用同一對照與成本門檻。

**第三輪 蒸餾模組**

K0 是 fresh hard-CE 控制；K1 的學生仍由隨機初始化，以固定現有6B教師提供 `L=0.75 CE+0.25 KL(p_teacher || p_student)`，T=1。教師固定於本地6B export／已發佈 revision `13125ad091ca25c914e1528af6dce22d5f6efa3a`。師生使用相同有效 context，教師不看下一詞之外的未來位置。[蒸餾成本與規模研究](https://arxiv.org/abs/2502.08606)。

先驗證 λ=0 恢復 hard-CE、相同分布 KL≈0、teacher gradients 全為空、shift／ignored labels／document spans 完全一致。先做 full-vocabulary chunked exact KL，不直接用截斷 top-k。若後續壓縮教師訊號，須另對完整 KL 測偏差。

學生 schedule 與匹配 token 指標使用學生 inputs；教師 forward、額外重算及合計 inputs 另列。為保守遵守 hard cap，logical charged inputs 使用「學生＋教師＋額外模型 forward」總和，≤6B；因此1B學生＋1B教師約收費2B inputs。`train/total_tokens_seen` 清楚定義為學生實際訓練 inputs，另列 teacher 與 conservative budget counters。教師的歷史6B訓練／64.7067 H200-hours另報，不能宣稱整體資訊來源只有1B。

K0／K1 各以三個 matrix-LR倍率做同額 calibration，固定 λ與T。初篩同學生 tokens 的 NLL；若成立，才做同新增 GPU budget 的比較。K1 正面後可追加 K2，ε=0.1 的 uniform label smoothing，檢查一般平滑是否也有收益；此對照沒有匹配教師 entropy，不足以單獨證明整個因果機制。

**第四輪 Schedule-free 模組**

F0 是 scheduled NorMuon；F1 使用 SF-NorMuon hidden matrices＋SF-AdamW aux；F2 使用與 F1 對齊的 inner update、相同 warmup／constant LR，但梯度在快速權重計算，另維護只觀察、不回饋訓練的平均權重。F2 同時計 live 與 observer NLL，從同一個 run 取得，不額外訓練。

F1 初始設計採 β插值0.9、matrix momentum0.95、second moment0.95、eta_scale0.2；這是本地 adaptation，不冒稱論文最佳值。warmup170個BS16等效單位後維持 constant LR，無第二 phase 重啟／cooldown；資料 batch windows仍共用原計畫，平均 clock 使用 optimizer updates，權重∝LR²，batch 切換不重設。[SF-NorMuon reference](https://arxiv.org/html/2605.23061v1)。

F1 的 evaluation／export 使用 x，訓練使用 y；切換前後必須核對 y 可恢復，WD 施於 z，F1／F2共同採固定參考shrinkage `1−reference_peak_lr×0.1×warmup(t)`，不隨LR校準倍率改變；F2則施於fast weights。SF-AdamW 的 aux state 同步切換。除了1B endpoint，記錄既有 token knots 的 anytime NLL；F0 未完成 cooldown 的中途值只代表可用 checkpoint，不代表該較短 horizon 的最佳 scheduled 結果。

成功可分兩種：固定1B品質勝出；或 endpoint regret≤0.02 nats 且多個事先指定停止點較好，取得「anytime 候選」資格。第二種不等於本專案最佳PPL。若要主張對多 horizon 最佳，須另跑各 horizon 完整 cooldown 的 fresh scheduled 控制，另行估算成本。

**第五輪 CCE 系統模組**

C0／C1 只差 native CE 與 `cce_exact`，保留同一 optimizer、microbatch1、global batch與資料。先用小張量及完整模型比較 loss、hidden gradients、tied classifier／lookup 梯度；包括 masked labels、跨文件邊界與不同有效 label 數。預設不啟用 gradient filtering。[CCE exact 模式](https://github.com/apple-aiml-research/ml-cross-entropy)。

通過後在同一 allocation 做 ABBA／BAAB 計時：每 block暖機20步、測40步，BS16和64各測一組；記錄compile與初始化總成本，穩態統計另列。這些效能測試的輸入也記帳，不能視為免費。時限不足即記未完成，不延長。

C1 若整步耗時改善≥5%或峰值記憶體改善≥10%，才做 fresh 1B C0／C1 品質等價檢查。更大 microbatch或較少 checkpointing是後續單獨對照；不得和 CE kernel 同時改，否則無法定位收益。

**分析與晉級規則**

主要指標為 `NLL=Σdoc_nll/Σdoc_valid_prediction_tokens`，PPL為exp(NLL)，Δ為candidate−control。1k用於校準與初篩；勝出設定鎖定後再用固定額外9k確認。該9k曾供過往驗證，是共用 validation，不是新獨立 holdout；不從剩餘40k另取資料反覆選方法。

對完整文件保存NLL和token count，以文件為單位配對bootstrap10,000次，每次重新算總NLL／總token差；不可平均文件PPL。core的兩個主要對比D025−N0、D025−D1各給97.5%區間，作Bonferroni同時95%控制；geometry的兩個對照同樣處理。單一事前鎖定對比用95%區間。其他分層分析標為探索，除非事前登記不作額外勝出宣稱。

| 結果類型 | 晉級條件 | 可作的結論 |
|---|---|---|
| 品質候選 | 匹配1B ΔNLL≤−0.005，9k配對區間上界<0 | 值得進第二 seed，尚未證明長 horizon 排序 |
| 品質確認 | fresh seed43方向相同；兩seed平均ΔNLL≤−0.005；確認評估方向一致 | 在兩個初始化下支持改善 |
| 效率候選 | ΔNLL區間上界<+0.002，且可比完整 allocation成本下降≥5% | 品質非劣的效率收益，不能稱品質勝出 |
| 品質提高但更貴 | 成本增加>5% | 記錄品質／成本取捨，進獨立同預算確認才宣稱效率改善 |
| 正常但無差異 | 區間跨0或效果小於門檻 | 證據不足；不是證明方法永遠無效 |
| 數值／成本失敗 | NaN／Inf、OOM、來源不符、不能完成共同horizon | 排除當前配置；保留耗費及失敗原因 |

−0.005 nats約等於0.5% PPL改善，是本次實用門檻。文件bootstrap只涵蓋給定模型的資料抽樣不確定性，不能代替seed variation；兩seed仍不足以作廣泛統計保證。若兩seed方向相反，停止晉級並列為不確定；第三seed需另列成本，非自動加跑。

分解詞頻、文件長度、文件內位置，以及可回溯來源的NLL。沿用全體有效token加權；來源metadata缺失時不猜來源類別。提前停止僅依數值、資源或身分錯誤；一般loss波動不作跨方法早停，避免系統性偏袒快收斂方法。

**同 GPU 預算確認與組合實驗**

同token比較通過後，另設兩個fresh run，每個90分鐘／12H200-hours上限。先只用鎖定的速度模型選各方法能在相同預算內完成的token horizon與完整cooldown，預留10分鐘停止margin；不得觀察NLL後再改horizon。每個方法的target由整批planner取不超cap的endpoint，BS切換與phase依相同比例重新規劃。若處理量含teacher，仍受保守logical6B輸入上限。

此項衡量「同資源上限內可完成的recipe」，實際成本照實列出；不能透過idle補時宣稱算力嚴格相等。中途曲線的time-to-target可輔助觀察，但不能將未cooldown checkpoint與已完成endpoint當作純optimizer因果證據。完整pair加確認預算上限26H200-hours。

只有optimizer與蒸餾各自確認，才考慮fresh 2×2：baseline、optimizer-only、KD-only、兩者合併，共四組，各1B。先鎖定設定，不在組合中額外調參。交互作用為 `NLL11−NLL10−NLL01+NLL00`；負值支持額外互補，零附近表示近似可加，正值表示互相抵銷。此輪是新的交互作用實驗，不能僅重用不同source／environment／time的舊對照。

**資源預算與執行順序**

以下均為H200-hours，包括失敗、compile、evaluation、輸出及allocation等待；每job8GPU／8CPU，同時預設一job。不是wall-clock小時，也不是已授權餘額。

| 模組 | 正確性／計時 | 校準 | 1B初篩 | 9k確認 | 上限 |
|---|---:|---:|---:|---:|---:|
| 核心 N0／D1／D025 | 6 | 24 | 48 | 2 | **80** |
| AdaMuon／SOAP＋新控制 | 4 | 24 | 48 | 2 | **78** |
| 蒸餾＋新控制 | 4 | 16 | 32 | 2 | **54** |
| SF＋constant橋接＋新控制 | 4 | 24 | 48 | 2 | **78** |
| CCE＋新控制 | 4 | 0 | 32 | 2 | **38** |
| 每個勝出方法的seed43配對 | 0 | 0 | 32 | 2 | **34** |

核心的三個1B run若與歷史控制同成本，約32.18H；加上驗證／校準／確認的完整預留後約64.18H。這是以現有10.7267H／1B外推的新方法估算，實際未知；80H是硬上限。核心加一次seed43確認最多114H，相當於8GPU連續使用14.25小時，另有程式實作與排隊時間。其他模組不自動加入；沒有把所有模組相加當成一次活動。

目前可引用的帳本快照已有1194.6611H歷史實測支出，這不是可用餘額。本設計未建立新的授權額度；既有formal17及其startup replacement例外只適用原run。任何啟動前需在當時帳本下確認新活動的明確上限、source與設定。現有slurm/ppl-factorial-train.sbatch提供account／partition／H200 constraint參考，但新的adapter尚未實作，因此本次不產生可誤提交的訓練腳本。

最有資訊價值的第一個決策是完成core後決定是否做seed43；若Dion3無優勢，再依量測選geometry或KD。10/20截止前是否足夠，須包含adapter驗證與排隊，不能只看訓練時數；未完成確認的候選保留為研究結果，不替代已驗證交付模型。

**暫緩方法的具體驗證問題**

DoReMi／DoGE：用training-only proxy產生凍結的domain weights，再比較相同tokens的原mix學生與新mix學生。保持split不變、另建overlay；proxy、重新packing與教師等資訊成本另記。此方法改變資料分配，不能擅自加入固定prefix核心組，也不能用官方50k梯度學權重。

MARS：先比較一梯度近似版與N0；精確版的額外forward／backward均收費。Cautious optimizer：只改aux Adam的方向遮罩，保持ordinary WD，避免混同已做過的cautious decay。多token預測：需要增加訓練heads，超出目前固定架構核心，只有在此範圍明確改變後才設新組。這些候選不占用上述第一輪預算。
