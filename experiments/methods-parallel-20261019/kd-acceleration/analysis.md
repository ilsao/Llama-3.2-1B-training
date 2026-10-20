教師選擇已修正：舊 K1 使用本專案只訓練約 6B tokens 的 1.24B 模型，測到的是早期同架構教師的知識轉移，沒有回答「成熟預訓練教師能提供多少幫助」。不再把舊 K1 當作成熟教師蒸餾的代表結果。

**後續pipeline驗證已完成：** 官方3B教師現已下載並通過完整tokenizer對齊；8卡pipeline與串行在確定性backward下完整梯度一致。短測每步1.04255→1.02843秒，吞吐提升約1.37%，顯存峰值33.636→37.503GiB／卡。收益有限，正式訓練仍未啟動。以下保留前期設計與測試歷程；最新實作、注意力後端修正、計時與計帳請見 [pipeline驗證紀錄](pipeline/README.md)。

修正版採官方 **meta-llama/Llama-3.2-3B base**，revision `13afe5124825b4f3751f836b40dafda64c1ed062`。官方模型卡記載 Llama 3.2 系列預訓練最多 9T tokens，且使用過 Llama 3.1 8B/70B 的 logits。優先考慮過 Llama 3.1 8B base（官方記載 15T+），但目前憑證讀取其 config 回傳 HTTP 403；3B 的 config、權重索引與第一個權重 shard 的 HEAD 存取已成功。尚未下載權重，尚未啟動新訓練。[Meta 3B 模型卡](https://huggingface.co/meta-llama/Llama-3.2-3B)、[Meta 8B 模型卡](https://huggingface.co/meta-llama/Llama-3.1-8B)。

選 base 的理由是本任務評估原始文本的下一 token NLL；先避免將對話後訓練造成的分布改變混入教師效果。教師參數量或預訓練量本身不保證這份 held-out 資料上的優勢，仍需先量測教師 NLL。學生繼續隨機初始化，不載入教師或其他訓練 checkpoint 的權重。外部預訓練教師引入外部知識，結果應與純 optimizer 比較分開解讀。

使用者已選定 **online 蒸餾**：每個學生 microbatch 以 `torch.no_grad()` 即時計算凍結教師輸出；完整詞彙 logits 僅在 GPU 上分塊暫存並立即用於 loss，不寫入持久快取，也不建立離線 logits／top-k 資料集。教師與學生使用相同 input IDs、文件邊界與預測位置。加速工作聚焦避免投影重算、分塊計算及 batch／activation checkpointing 的效能驗證。

具體配置已保存於 `pretrained-teacher-proposal.json`；原始 K1 的不可變配置位於 `../configs/K1-seed42.json`。

| 項目 | 修正版配置 |
|---|---|
| 教師 | 官方 Llama 3.2 3B base；28 層、hidden 3072、vocab 128256；BF16、eval、凍結、無 KV cache |
| 學生 | Llama 3.2 1B 架構，1,235,814,400 參數；16 層、hidden 2048；residual-half 隨機初始化、seed 42 |
| 目標 | `0.75 CE(y,p_s) + 0.25 KL(p_t || p_s)`，T=1，完整 vocabulary |
| 遮罩與位置 | 相同訓練文本、8192 context、document-span causal mask；各模型保留原生 RoPE；忽略 padding 與無效 shifted labels |
| Vocabulary | 必須核對完整 token-ID 對應、normalizer、特殊 token 與實際編碼；詞彙大小相同不足以證明可直接對齊 |
| Batch | 8 H200／8 CPU；microbatch 1；累積 2 次形成 global BS16；500,039,680 inputs 後 BS64 |
| Matrix optimizer | NorMuon，NS5，momentum .95、Nesterov、post-NS row normalization beta2 .95；peak LR .008614891542046706；ordinary WD .1 |
| 其他參數 | tied embedding/head 與 norms 用 AdamW；peak LR .0005168934925228023；betas (.9,.95)、eps 1e-8、WD 0 |
| 梯度 | global clip .5；按所有 rank 與累積 microbatch 的有效 labels 統一正規化 |
| Schedule | 保持 K1 token-clock WSD：833,355,776 inputs 分相；第二峰 .1×；每相 warmup 170 個 BS16 等效單位；40% tail，ramp/hold/final=4:1:1，plateau .2 |
| 評估 | 相同固定 1k；每 ≤400 updates 及原定 token knots；完成目標才做 additional 9k |
| Pilot 目標 | 1,000,079,360 student input tokens；實際 wall-time 須依新教師速度重新訂定，尚未提交 |

α=.25、T=1 保留作為第一個成熟教師實驗的控制變因，不宣稱是最佳值。這裡使用的是 forward KL `KL(teacher || student)`；其 logits 梯度是 `p_s - .75*one_hot(y) - .25*p_t`。不同教師 hidden size 不影響這個公式，兩者只需各自投影到一致的 vocabulary。

已取消 D025、SOAP、F1、F2 及依使用者明確要求取消的 K1。前四組採「最近兩個相同 token 的固定 1k 評估皆比 N0 差 >5%」作資源裁剪規則，並非統計顯著性檢定；只有 seed42。保留接近 N0 的 D1、AdaMuon、CCE。證據與 job ID 在 `cancellation-intent.json`；所有取消 allocation 已保留並對帳。

舊 K1 在 step3052／400,031,744 inputs 的固定 1k PPL=29.2867，N0=32.4394，低 9.72%；取消時處理到 408,158,208 inputs。這只能證明舊教師在該位置有幫助，沒有最終 1B／9k 結果，也沒有相同時間的勝出證據。

既有計時取相同 BS16、steps3015–3114 各100步中位數。N0 為歷史紀錄，並非同期同節點的配對效能實驗。

| 秒／update | 舊 K1 | N0 | 同期 CCE |
|---|---:|---:|---:|
| 整步 | 1.2142 | .6707 | .7994 |
| 計算（含 optimizer） | 1.1830 | .6386 | .7705 |
| Optimizer | .1353 | .1263 | .1275 |

舊 K1 整步耗時是 N0 的 1.81 倍；額外計算約 .544 秒，而 optimizer 只增加約 .009 秒。兩者計算以外差距皆約 .03 秒，因此現有證據指向計算路徑，沒有支持優先擴充 dataloader workers。這些粗粒度紀錄無法再分解教師 backbone、head 與學生 backward 的精確占比。

程式層面的主要浪費已定位：每個序列有 8191 個 shifted positions，舊 loss 以128為一塊，共64塊。Forward 計算學生與教師的完整128256維 logits；backward 又重算兩者 logits 與 softmax。除此之外，每塊都累加約1.05GB的學生 head FP32梯度，造成多次大矩陣讀寫。這些是可消除的實作成本；教師 backbone 推論則是 online KD 的必要成本。舊 K1 的 peak allocated 約33.91GiB、reserved約41.75GiB，當時顯存不是首要限制。換3B／8B教師後必須重新測量。

`fast_kd.py` 實作的改進是：forward 得到 logits 時立即算出並保存 `dHidden`、`dWeight`，backward 只乘上游梯度。以解析梯度避免重複學生／教師投影，保留完整 KL，不做 top-k 近似；另以1024 positions 為候選分塊，將每序列分塊數從64降至8。前者改變計算流程，後者減少小 GEMM 與巨大 dWeight 的反覆累加。Liger 的公開 fused linear distillation 也採用在 forward 產生梯度、backward 套用上游梯度的結構；本模組直接實作上述 KL 解析梯度，未安裝新依賴。[Liger 官方原始碼](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/chunked_loss/fused_linear_distillation.py)。

驗證狀態必須分開看：job573035 在 GPU 通過10個小型 FP32／BF16數值案例（α=0/.25/1、ignored labels、全部 ignored、相同教師、非單位上游比例），另通過 FP32 tied-embedding 梯度檢查。FP32 最大相對L2誤差約3.4e-7；BF16約0.00323。完整模型初始化後，profiling harness 因 `inspect.getclosurevars` 遇到未啟用 CCE 分支的空 closure cell 而失敗，尚未產生完整梯度或端到端速度結果。實耗49秒×8H200=0.108889 H200-hours，已入 shared ledger。原始程式與seal保存於 `attempt1/`。已修正為只讀取教師 cell，並通過輕量回歸檢查；未自動重交舊教師 benchmark。

後續 prototype 也已允許教師與學生有不同 hidden size，並提供 `pretrained_teacher.py` 顯式 adapter：保留教師自身深度／RoPE，教師不加入學生 optimizer／export，按配置讀取 teacher_weight、chunk，非T=1明確拒絕。不同 hidden size 的新版本與3B整模型尚未GPU驗證。此自訂 autograd 僅支援一階梯度；BF16將縮放移到GEMM之後會改變捨入，數學目標相同不等於bitwise等價。

新教師的加速順序：先做3B真實形狀的梯度、完整8卡forward/backward、teacher NLL與配對吞吐檢查；通過後比較 recompute128、recompute1024、pregrad128、pregrad1024。再評估在相同global BS16下以microbatch2／accum1攤薄開銷，以及用顯存換取關閉學生activation checkpointing。這兩項都要保持token順序、遮罩、loss normalization不變，並測量記憶體峰值。

依使用者最新決定，離線完整 logits 與 top-k 快取均移出目前方案；保留完整詞彙的 online KL。新教師更大，舊1.24B教師的時間不能按比例直接外推。現階段結論是已找出並實作可避免的重算，但**尚無端到端加速倍數，也尚無成熟教師蒸餾的PPL結果**。
