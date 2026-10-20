# 兩階段、每階段兩個平行 sbatch

此文件依使用者最新指示取代 first-round.md 的四組獨立對照提交順序與70/72H估算。方法定義維持不變；新的執行規格是 execution-plan.json、configs/、batch-plan.json 與 execution-seal.json。

## 第一階段：causal mask

同時提交兩個獨立 job，不設彼此的 Slurm dependency：

| Run suffix | Mask | Matrix optimizer |
|---|---|---|
| s1-dense | 原 packed cross-document causal | native Muon NS5 |
| s1-intradoc | document_spans 分塊 causal | native Muon NS5 |

兩者都必須到達1,000,079,360 non-padding inputs、4,775 successful/attempted updates，完成固定1k documents／1,479,302 prediction tokens 的 raw endpoint 評估、observers評估、checkpoint及HF reload驗證，並由sacct確認COMPLETED。失敗、不完整、來源不一致或非有限NLL都阻止第二階段；不自動retry/resume。

選擇 **raw endpoint mean NLL最低者**（等價於PPL最低），不是best checkpoint、train loss或EMA最低者。完全相同則選dense；差距小於.005 NLL仍按較低者選，但標註差距很小、不宣稱顯著。將兩個結果、來源與決策固定到stage1-selection.json。這裡的選擇規則依使用者要求，不以先前.005改善門檻阻止進入第二階段。

## 第二階段：選中mask上的PE-only與NorMuon-only

第一階段兩個allocation均結束且核對後，再同時提交：

| Run suffix | Mask | Matrix optimizer |
|---|---|---|
| s2-pe | 第一階段勝出者 | 原Muon外層更新＋PE5；無NorMuon |
| s2-nor | 第一階段勝出者 | 原NS5＋NorMuon neuron-wise normalization；無PE |

兩者均重新從seed42隨機初始化，使用同一canonical raw prefix與完整1B horizon。只繼承mask設定，不載入第一階段checkpoint，不改LR/batch/scheduler/clip/decay。第一階段勝出的NS5 run可作第三個比較點，因此總共四次training就能比較選中mask下的NS5、PE與NorMuon。

保留比較的限制：mask是依同一開發集選出的，單seed排名有adaptive selection偏差；這不是完整mask×optimizer factorial，不能推論落選mask上的optimizer排名。兩個新方法若都輸NS5，保留NS5；不強制從PE與NorMuon挑一個宣稱改進。PE+NorMuon、不同seed、3B及額外9k/50k評估都不自動啟動。

## 共通配方與讀出

- 原Llama-3.2-1B架構、residual-half random initialization、8192 context、同樣tokens/labels/packing/positions。
- phase boundaries為833,355,776與1,000,079,360inputs；second peak=.1×；每phase170個BS16等價warmup，40%tail，ramp/hold/final=4:1:1、hold=.2×phase peak。
- Muon peak=.008614891542046706、AdamW peak=.00025844674626140116；Muon momentum=.95、Nesterov；AdamW(.9,.95)；Muon WD=.1、embedding/head/norm無decay；clip=.5。
- 8 ranks×micro1×accum2，在500,039,680inputs之後改accum8；token事件邊界縮短accumulation以精確完成。實際更新4775次。
- 每400updates、共同10%tokens節點、batch/phase邊界與終點評估；W&B project為lab5-training-llama，四個fresh identities。
- 每run保留raw、token-EMA與finite-window mean。EMA從950,140,928inputs開始，half-life=10,000,793.6inputs；mean snapshots為979,894,272、985,137,152、989,986,816、995,229,696、1,000,079,360inputs。observers不回寫訓練，且不參與mask選擇。
- PE使用固定五組係數及1.02 normalization，native Muon外層更新；NorMuon沿原output-row累積FP32二階矩、beta2=.95、保留更新Frobenius norm，batch/phase切換不重置。見first-round.md的算法細節。

## 每個sbatch與預算

每job：已驗證nano4 account ACD115198、partition8gpus、constraintH200、單node/launcher、8H200、8CPU、200G RAM、120分鐘、no-requeue。兩job可併行，峰值16H200/16CPU；實際同時開跑取決於Slurm排程。未修改其他job或cron。

各job先用獨立進程做最多600秒preflight：native NS adapter parity、FA3 block-mask forward/backward對SDPA、GQA/padding/singleton/continuation、tiny Llama＋activation checkpoint transport、observer state/RNG檢查，並按PE/NorMuon方法增加數值檢查。這是合成數值驗證，沒有載入或續訓歷史模型。通過後才啟動真正fresh訓練進程。preflight時間完整計入該run allocation。

訓練停止點保留900秒end-evaluation/export＋至少300秒stop margin；總共仍120分鐘。若時間不足，保存並標記不完整，不延長、不自動重跑，也不拿少tokens結果晉級。

| 項目 | sbatch數 | H200-hours上限 |
|---|---:|---:|
| Stage1 mask | 2 | 32 |
| Stage2 PE／NorMuon | 2 | 32 |
| 總計，已含preflight、evaluation、checkpoint、export與idle | 4 | **64** |

所有四個run是1B bounded pilots，每個logical run上限16H；不新增2B+正式run。每筆分配及失敗成本都寫入既有shared ledger。沒有新9k confirmation allocation、cleanup或HF upload。大型產物寫/work，source/config/log receipts保持獨立。

## 控制器與檢查

唯一submission owner為source/slurm/ppl_controller.py，使用跨進程file lock、先reserve再submit、唯一Slurm comment tag及uncertain-submission恢復；不重試不確定的submission。minute cron只負責輕量查詢、收據驗證及第二階段提交，不執行任何模型／資料工作。

`python3 experiments/ppl-next-20261016/source/slurm/ppl_controller.py status`查看狀態；在研究目錄建立disabled檔只停止新提交，已分配jobs仍照常記帳。source與六份預先固定config（stage1兩份，stage2兩種mask各兩份）在execution-seal.json封存；第二階段僅選中一對，不會提交另外一對。

已完成的login-node檢查：配置load/AST/shell語法、精確token/schedule終點、observer knots、兩階段selector鎖定、平行提交分支及失敗停止測試。GPU preflight與正式訓練狀態請以controller-state.json、receipts/和Slurm為準。
