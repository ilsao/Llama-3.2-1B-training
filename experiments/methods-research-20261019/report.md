**Llama 3.2 1B：超越調參的訓練方法研究與實驗判斷**

研究日期：2026-10-19，Asia/Taipei。結論基於已完成實驗、6B 執行封存程式、論文及作者實作。本文提出的方法尚未在本專案驗證；本次工作產出研究與證據整理，沒有新增 GPU allocation。

建議優先驗證 Dion3；以詞彙層與梯度結構診斷，選擇 AdaMuon 或 SOAP 作為第二條優化器路線；另立一條使用現有模型的蒸餾路線。Schedule-free、資料混合與多 token 預測具有不同目的，應依目標與限制分開評估。完整詞表交叉熵的記憶體優化可以支援上述研究，但其速度收益不能直接當作相同 token 下的品質收益。

**1．重新定義問題：我們究竟缺少什麼？**

目前基準已有文件內因果注意力、NS5 NorMuon、AdamW 輔助群組與完整 cooldown。這不是一個只需從 AdamW 換成 Muon 的弱基準。最新 6B run 的固定 1k PPL 為 12.77394，額外 9k 為 12.92243；以總 NLL／總有效預測 token 合併的 10k PPL 是 12.90785。4B AdamW ×2 為 13.85047；4B step-tail30 為 13.98182。6B 與 4B 的差異包含訓練量與對應設定，不能當作某個方法的受控增益。這些 10k 數字也不能直接與其他實驗的全 50k PPL 排名。[可重建的數據與來源雜湊](evidence.json)、[6B 結果](../raw6b-normuon-adam2x-20261018/results.json)。

實際模型有 1,235,814,400 個參數，6B 輸入只有約 **4.855 tokens／parameter**。這表示文獻中經過 50B、100B 甚至更長訓練的排序，不一定適合此處的早期學習階段。系統比較論文也發現優化器的相對優勢會隨模型大小和資料量改變；勝過 AdamW 的結果不能直接推論勝過目前 NorMuon。[Fantastic Pretraining Optimizers](https://arxiv.org/abs/2509.02046)。

我將研究問題拆成四個可區分的假說：

| 假說 | 現有線索 | 能否證實的觀測 | 對應方法 |
|---|---|---|---|
| 更新方向或尺度不適合局部幾何 | NorMuon 只存每列二階矩，沒有逐元素或歷史旋轉基底 | 列內更新變異、輸入 covariance、更新方向與後續 NLL 的關係 | AdaMuon、SOAP、Newton–Muon |
| 每個輸入提供的監督訊號不足 | 只用真實下一詞的 hard-label CE，訓練量很短 | 蒸餾能否降低相同學生 tokens 的 NLL，且抵銷教師成本 | 使用現有教師的 soft-target 蒸餾 |
| 固定 horizon 的軌跡過度依賴 cooldown | 3B 附近 batch 切換與第一階段 decay 起點重疊 | 完整 horizon 下 scheduled 與 schedule-free 的比較 | SF-NorMuon |
| 計算分配浪費在不必要的中間張量或更新 | 128k 詞表；optimizer 占訓練計時約 13.5% | 實測峰值記憶體與整個 update 耗時 | CCE、Gram NS、Dion3 |

這些目前是待驗證假說。梯度 covariance 的各向異性不等於已量得 Hessian；單一 loss 曲線也無法識別是哪個假說成立。

**2．最值得先驗證的優化器：Dion3**

Dion3 將 Gram Newton–Schulz、對稱矩陣 kernels、部分列／行的更新與誤差回饋結合。未被選中的動量資訊會保留供之後更新。其 NorMuon 版本直接對照調過的 NorMuon：1B 模型／100B tokens 的驗證 CE 約低 0.01，另有較大模型／10B tokens 的支持。這是它優先於單純「勝過 AdamW」方法的主要原因。論文的最高約 6 倍是 optimizer step 的速度，排除 forward／backward。[Dion3 論文 §6–8](https://arxiv.org/html/2608.11612v1)、[作者實作](https://github.com/microsoft/dion)。

本專案建議的辨因實驗：先比較現有 NorMuon 與只替換 Gram NS 的等價實作，再比較 Dion3 的部分更新。前者主要測系統收益，後者才測更新規則的品質差異；每次只推進一個有證據支持的變更。可先以 f=1/4 作方法候選，f=1/8 留待前者成立後測試，這是研究設計而非已知最佳值。

移植時必須處理本地 native `original` shape scaling、Nesterov 與作者 RMS scaling 的差別。以實際每個矩陣的 update RMS、momentum 定義和 WD shrinkage 核對語義，不能照貼作者的數字 LR。f=1 也應先驗證，不能假設直接等於現有程式。保持輔助 AdamW 群組、輸入順序、mask 與 token-clock schedule 作共同控制。

本地計時提供更保守的效率預期：optimizer 3,659.04 秒／training 27,076.35 秒 ≈ 13.51%。假設整個 optimizer 都快 6 倍、其餘不變，Amdahl 推算整個訓練約快 **1.127 倍**；即使 optimizer 完全免費，上限也約 **1.156 倍**。這只是計時模型，不是 Dion3 benchmark；包含初始化與輸出的整個 allocation 收益還會更小。真正值得投入的理由是可能同時降低 NLL 與成本。

**3．第二條優化器路線：由診斷選擇 AdaMuon 或 SOAP**

目前 `neuron_normalize` 對正交化後更新計算 `mean(u², dim=-1)`，儲存每列一個二階矩，再恢復整體 Frobenius norm。它能平衡列尺度，但同一列不同座標共用縮放；也沒有利用歷史梯度的旋轉基底。這是程式能直接證實的結構，不代表它一定不如更複雜的方法。[本地優化器](../raw6b-normuon-adam2x-20261018/source-v3/ppl_optimizer.py)。

**AdaMuon 適合檢驗列內尺度是否重要。** 本次採用 v3：sign transformation、正交化、逐元素二階矩與 RMS 對齊共同構成方法。Qwen 實驗的 time-to-target 改善相對 Adam 為 30.8%，相對 Muon 則是 2.9%；尚不能推論勝過 NorMuon。[AdaMuon v3](https://arxiv.org/html/2507.11005v3)、[作者程式](https://github.com/Chongjie-Si/AdaMuon)。

移植後若有效，再拆開 sign transformation 與逐元素適應，以免把整套方法的收益歸給其中一項。依目前 hidden-matrix 參數數量估算，多一份 FP32 全矩陣二階矩約 **3.625 GiB／DDP rank**，尚未包含暫存張量。若列內變異不明顯、品質差異也不足，沒有理由只為形式上的精細度付出成本。

**SOAP 適合檢驗座標相關性是否重要。** 它在 Shampoo 建立的旋轉座標中使用 Adam，再將更新轉回去。原始語言模型實驗有相對 AdamW 的迭代與 wall-time 改善，但比較模型、context 與基準不同，並非本地 NorMuon 的勝出證據。[SOAP](https://arxiv.org/abs/2409.11321)、[作者實作](https://github.com/nikhilvyas/SOAP)。

本地 8192×2048 MLP 矩陣會使完整左右 Gram 矩陣及旋轉基底很昂貴：若四個矩陣皆 FP32，單個權重的這部分約 **544 MiB**，48 個此形狀 MLP 權重合計約 **25.5 GiB／rank**，尚未計動量、逐元素二階矩與 eigensolver 暫存。應先測完整方法的實際可行性；若選擇單邊、分塊或只套部分層，需明確稱為變體。官方 `max_precond_dim` 的限制會跳過過大維度，不代表自動實作了分塊 SOAP。[實作細節](https://raw.githubusercontent.com/nikhilvyas/SOAP/main/soap.py)。

另一個機制更直接的備選是 **Newton–Muon**：利用輸入 activation 的二階矩做右側預條件化，再進行矩陣方向更新；原文在 modded-nanoGPT 的改善幅度約為 6% 迭代、4% 時間，證據規模較小。[Newton–Muon](https://arxiv.org/html/2604.01472v1)。若診斷顯示 activation covariance 強烈偏斜，可用它回答「輸入幾何是否才是瓶頸」。與本地 NorMuon 合併將是新變體，不能直接引用原文成績；activation checkpointing 的重算 hook 也不能重複累積統計。

**4．比換 optimizer 更根本的訓練方法：利用現有模型蒸餾**

現有 6B 教師可提供一個 hard label 以外的詞彙分布。提案是讓學生保持相同架構與隨機初始化，在原始訓練資料上最小化：

`L = (1 − λ) CE(y, p_student) + λ T² KL(p_teacher,T || p_student,T)`

教師固定，師生使用相同的文件內 context。學生不載入教師權重；教師 checkpoint 僅用於產生監督。初步研究目標是「1–2B 學生 tokens 能否更有效率」，而不是預設同架構學生的 6B 結果必然勝過教師。

蒸餾 scaling law 顯示，其經濟性依教師是否已存在、教師品質和學生規模而改變；教師越強也不必然越好。專為單一學生從頭訓練教師，與利用已有教師是不同的成本問題。[Distillation Scaling Laws](https://arxiv.org/abs/2502.08606)。

這裡必須同時報告三項：學生實際處理 tokens、教師推論成本、新增 allocation 的總成本；教師的歷史訓練成本另列而不抹除。線上教師會增加 forward 與記憶體，收益需以相同總 GPU-hours 再比較一次。現有教師本身訓練不足，也可能把偏差教給學生，因此保留 hard-label 項並分析不同詞頻的 NLL。

不應把全資料完整 logits 寫入磁碟：按 6B×128,256×FP32 推算約 **3.08 PB**。可考慮線上分塊 KL；若用 top-k 加剩餘機率質量作近似，需先在小樣本與完整 KL 對照，承認它改變了目標。只儲存 top-k 並重新正規化會遺失尾部分布，不能稱為完整蒸餾。

**5．Schedule-free 值得研究，但目標是可隨時停止的軌跡**

SF-NorMuon 維護快速點 z、平均點 x，並在插值點 y 計算梯度，平均結果會回饋後續訓練。因此，先前末段 EMA observer 沒改善，不能否定這種方法。論文強調 weight decay 作用點；其附錄與 scheduled NorMuon 的差距仍約 0.02–0.03 nats，主要價值是未知 horizon 時的競爭力，不能宣稱必然改善已知 6B endpoint。[Schedule-Free Spectral Optimization](https://arxiv.org/html/2605.23061v1)。

此方法較適合未來需要多個停止點、算力常變動的研究。若測試，evaluation 必須使用 x，之後恢復 y，保存完整狀態，並明確處理 batch 切換後平均權重的時鐘。現有「約 3B 後 PPL 大降」同時包含 batch 增大與 cooldown，不能只拿該段斜率證明任一機制；要以各自完整訓練 horizon 的 endpoint 作比較。

**6．先補足詞彙層的可觀測性，再決定是否改變其更新幾何**

共享 embedding／LM head 是 128,256×2048，共 262,668,288 個參數，占全模型 **21.25%**。它與 normalization 參數一起由無 WD 的 AdamW 更新；AdamW decay group 實際為空。因此先前 AdamW LR ×2 的成功主要影響這一大組，不能概括成「所有參數都需要更大 LR」。現有紀錄沒有把 embedding 與 norms 的因果效果分離。

現有逐層診斷並非完全缺席，但更新量用 `p.flatten()[:256]` 抽樣；對 embedding 而言，這只涵蓋 token row 0 的前 256 維，無法代表 128k 詞表。[抽樣程式](../raw6b-normuon-adam2x-20261018/source-v3/recipe_runtime.py)。這是量測代表性的限制，並不是已證實的訓練錯誤。

建議先在小型 compute-node 診斷加入：依訓練詞頻分層的固定隨機 rows、每組完整 update RMS／weight RMS、列內更新差異、輸出 logit 尺度；並將 held-out NLL 依詞頻、來源、文件位置分解。要以有效 prediction tokens 加權，並保留整體 NLL。若品質只在高頻詞改善，才有理由研究詞彙層專用更新；若差異主要是信心尺度，可先用獨立 development probe 的溫度診斷區分 calibration 與 representation，不將事後校正混入既有結果。

輸出層 norm 與 scaling 的研究可提供診斷動機，但其 untied embedding 與額外正規化設定不同；不能直接搬到這裡的 shared tensor，也不能推論固定 norm 即保證最佳訓練。[Optimal Scaling Needs Optimal Norm](https://arxiv.org/html/2510.03871v2)。任何新更新規則都須保持 tied tensor 只更新一次，同時承受 lookup 與輸出兩路梯度。

**7．完整詞表 CE 是獨立的系統改進機會**

目前仍透過 HF CausalLM forward 取得 loss，沒有 CCE 路徑。[loss 呼叫](../raw6b-normuon-adam2x-20261018/source-v3/train.py)。單序列的 8192×128,256 logits 一個張量即約 BF16 **1.957 GiB**／FP32 **3.914 GiB**；這是形狀估算，尚未量測實際峰值與瓶頸。

Cut Cross-Entropy 可避免完整 logits 常駐。應先用無梯度過濾的 `cce_exact` 驗證；預設 CCE 的梯度過濾是另一個近似選擇。需核對 shift、ignored labels、sum reduction、跨 rank 的有效 token 分母，以及 tied classifier／lookup 梯度。[Cut Your Losses](https://arxiv.org/abs/2411.09009)、[作者實作與 exact 模式](https://github.com/apple-aiml-research/ml-cross-entropy)。

驗證後再測能否在相同 global batch 增大 microbatch，或減少 activation recomputation。預期收益是可測的資源效率；在相同 tokens 下不能預設 PPL 改善。蒸餾的 KL 也不能假設直接由 hard-label CCE 提供，需要另驗證實作。

**8．保留的研究候選與暫緩理由**

| 候選 | 要回答的問題 | 本次排序與理由 |
|---|---|---|
| Dion3 部分更新＋誤差回饋 | 部分更新能否兼顧品質與成本？ | 第一優先，有直接 NorMuon 比較 |
| Gram NS 等價替換 | 正交化開銷能否降低？ | Dion3 前的系統控制 |
| AdaMuon | 列內自適應是否有價值？ | 第二路線候選，需診斷支持 |
| SOAP | 歷史旋轉座標是否改善更新？ | 第二路線候選，先過成本門檻 |
| Newton–Muon | activation 幾何是否比更新尺度重要？ | 探索候選，外部證據規模較小 |
| 現有教師蒸餾 | 每 token 更豐富的訊號能否抵銷教師 forward？ | 獨立訓練方法路線 |
| SF-NorMuon | 未知停止點能否保有好 endpoint？ | 中期，固定 6B PPL 的證據較弱 |
| 詞彙層專用更新 | shared embedding 是否學得不均衡？ | 先補量測，暫不指定新演算法 |
| 完整詞表 CCE | 大詞表中間張量是否限制吞吐？ | 支援性系統改進 |
| DoReMi／DoGE | 哪些來源能帶來跨來源泛化？ | 長期，需要新的資料混合協定 |
| MARS | 梯度方差縮減能否提高 token 效率？ | 次要，精確形式的額外計算需計費 |
| Cautious momentum update | 梯度與動量符號不一致的更新是否有害？ | 可作小型備選，不能與 cautious WD 混為一談 |
| 多 token 預測 | 輔助未來詞任務是否改善共享表徵？ | 架構限制及目標匹配問題，暫緩 |
| 更換 tokenizer／縮詞表／改模型尺寸 | 架構是否更適合短訓練？ | 不符合本題固定架構與 tokenizer 的比較範圍 |

DoReMi 利用 proxy 的相對 excess loss 學習 domain mixture；DoGE 利用跨 domain 的泛化訊號。這與先前固定閾值的品質過濾不同，過濾失敗不能否定混合最佳化。資料權重必須從 training-only probe 學習，不能用官方保留 50k 的梯度；保留 canonical split，另建可追溯 mixture overlay，proxy 成本也記帳。DoGE 的資料重抽樣與只改 loss 權重也不是等價方法。[DoReMi](https://arxiv.org/abs/2305.10429)、[DoGE](https://arxiv.org/html/2310.15393v2)。

MARS 的精確方差縮減涉及同批資料在不同參數下的梯度；額外重算要計入實際輸入處理與 GPU 成本，不能當作免費資訊。近似版本與精確版分開報告。[MARS](https://arxiv.org/html/2411.10438v1)。Cautious optimizer 的方向遮罩與本地已試的 cautious weight decay 是不同機制，後者失敗不足以否定前者。[Cautious Optimizers](https://arxiv.org/abs/2411.16085)。

多 token 預測以額外 heads 學未來位置，原文有 downstream 尤其程式任務的支持；這不等於已證明下一詞 PPL 一定下降。即使匯出時移除額外 heads，訓練架構仍已改動，因此不列入目前固定架構的首輪；若日後研究，future labels 也必須遮掉跨文件位置。[Multi-token Prediction](https://arxiv.org/html/2404.19737v1)。

**9．下一輪實驗應如何產生知識，而不只再產生一個數字？**

第一步是修正量測代表性，並在小型 Slurm 驗證中完成新方法的數值與成本檢查。此步只回答「程式正確、成本可接受嗎」，不以 warmup 中的 PPL 宣稱方法勝出。確認每個 trainable tensor 恰好分組一次、跨 rank 梯度與真實輸入 token 計數一致，保持 8192 與 intradoc mask。

第二步先比較 **現有 NorMuon 控制組與 Dion3**。由診斷選出的 AdaMuon 或 SOAP，以及蒸餾，屬下一輪候選，避免一次啟動全部方法。每個方法有相同的小型校準預算，因不同更新尺度不能共用數值 LR；校準是公平比較的必要條件，研究主體仍是更新規則／監督訊號。先前完整 1B pilot 每臂約需 10.9 H200-hours，可作粗略規劃參考，新方法要重新實測，不能視為已授權預算。

第三步以完整 cooldown 的匹配 1B endpoint 篩選，保留相同 seed、資料 prefix、global batch、evaluation 與初始化；方法本身若更改 schedule，需事先定義公平 horizon。選出候選後再做第二 seed 與固定額外 9k 的文件配對 NLL。這 9k 曾用於過往確認，是共用 validation，不稱為全新 holdout；文件 bootstrap 衡量資料抽樣不確定性，不能替代 seed replication。1B 排序只作暫定，通過後才考慮較長的匹配 token horizon。

建議的預先登記判準如下，門檻是研究設計，不是測得的結論：

| 面向 | 判斷方式 | 失敗時學到什麼 |
|---|---|---|
| 品質 | 匹配 tokens 的 ΔNLL；可用 −0.005 nats（約 0.5% PPL）作值得追蹤門檻，並看配對區間與 seeds | 小收益可能不足以值得移植，或短 horizon 尚無優勢 |
| 計算效率 | 到相同 NLL 的完整 allocation GPU-hours，以及相同 GPU-hours 的 endpoint NLL | 更好梯度方向未必有更好的成本效益 |
| 系統方法 | exact loss／gradient 對照、峰值記憶體、整步耗時，品質在預定等價範圍內 | 加速來源或數值近似與預期不同 |
| 蒸餾 | 同學生 tokens 與同總新增 GPU-hours 兩份比較；另列教師歷史成本 | 可區分監督訊號收益與算力堆疊 |
| 泛化 | 詞頻、來源、文件位置 NLL 加上整體加權 NLL | 可區分普遍改善與只服務一小群資料 |

如果新方法每步慢 20%，需要約少 16.7% 的步數才在同 target loss 的訓練時間上打平；單次 endpoint PPL 不能代替這個計算。不要將 overhead 藏在準備工作，或以新的 run ID 重置既有成本。

考慮 10/20 lab 截止日，現有最佳模型已完成交付驗證，新方法需要依上述順序累積證據；沒有足夠時間完成數值、品質與資源驗證的候選，應保留為後續研究，不能用未完成的 pilot 取代正式結果。本報告不推算新方法一定能達到某個 PPL，也不把文獻的速度百分比直接套到本專案。

**證據重建**：`python experiments/methods-research-20261019/collect_evidence.py` 只讀既有小型 JSON 與封存原始碼，輸出本目錄的 `evidence.json`；不載入權重、不處理資料集、不提交 Slurm。所有新方法的模型級驗證仍需在 compute node 執行。
