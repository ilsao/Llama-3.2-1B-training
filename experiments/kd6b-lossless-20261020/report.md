# 6B-token lossless 蒸餾：瓶頸、實測與達標條件

2026-10-20，Asia/Taipei。此處 lossless 依使用者確認，指數學等價，容許 BF16 浮點運算順序造成的差異。教師、完整詞表 KL、資料／document mask、global batch、optimizer 更新公式均保留。

**目前尚未驗證出 64 H200-hours 內完成的配置。** 已完成一次實機配對驗證，實際花費 **0.8 H200-hours**。最好的已測方案是八張 GPU 同時放教師與學生、移除學生 activation checkpointing、融合完整詞表 CE/KL：吞吐比同期同卡控制高 **19.3%**；6B 純訓練估算 **80.66H**，加10%執行餘量與30分鐘初始化／評估／存檔預留後為 **92.72H**。不能把此結果宣稱為64H達標。

## 原始3B訓練的實際成本

| 角色 | Job | Allocation | H200-hours |
|---|---|---:|---:|
| 教師8卡 | 575376 | 17,073秒 | 37.9400 |
| 學生8卡 | 575377 | 17,879秒 | 39.7311 |
| 合計 | | | **77.6711** |

已完成3,000,107,008學生inputs／14,312更新，教師另計相同inputs。固定1k PPL13.73196，合併10k PPL13.94690。按實際成本與token數線性外推6B約 **155.34H**，需將每token成本降為41.2%。學生單獨外推也要 **79.46H**；只優化教師不夠。

來源：[Slurm accounting](../kd3b-hidden8-20261019/accounting.json)、[完成狀態](../kd3b-hidden8-20261019/status.json)、[可重算的分析資料](analysis.json)。跨6B的外推保留兩種global batch各占約一半tokens，尚未執行新6B排程。

## 實際瓶頸

1. **額外教師allocation的等待成本很大。** 教師完成全3B的service time（計算＋D2H＋寫入、最慢rank累計）10,319.64秒，allocation17,073秒。rank0寫入4,547.68秒、等待學生騰出spool4,961.40秒。三者的rank／計時範圍不同，不能直接相減後稱為精確GPU利用率。生產程式把teacher forward、D2H同步與write/fsync串行執行；可用雙buffer＋writer thread重疊，但這無法解決學生運算本身。

2. **hidden transport已經大量隱藏讀取時間。** 學生rank0累計讀取約2,006.88秒，實際暴露的等待656.86秒；所有rank中最大的logged等待約699秒，占約17,000秒計算區間4.1%。消除所有讀取等待也不是倍數加速。6B BF16 hidden payload約36.86TB，寫入＋讀取約73.72TB；1TB只容納約1.63億inputs，僅6B的2.7%。它是緩衝，不能消除整次教師的成本。

3. **學生有可移除的重算。** 舊run學生峰值allocated約34.78GB、reserved約46.19GB，卻啟用整層checkpoint。checkpoint省顯存的代價是backward重跑部分forward。已實測關閉後可容納：教師與學生同卡、fused loss的峰值50.73GB，遠低於H200容量。[PyTorch checkpointing說明](https://pytorch.org/blog/activation-checkpointing-techniques/)

4. **完整詞表loss存在HBM搬運浪費，但不是全部瓶頸。** 舊實作每1024位置產生兩份FP32 log-softmax及多個128256詞表中間張量，另做教師／學生projection、hidden gradient及head gradient GEMM。新kernel在FP32暫存器完成normalizer、CE、KL與dlogits，將梯度寫回BF16 student-logits buffer；全部詞彙仍參與。實測head/KD區間縮短約38.5%，端到端只再快約7.0%，符合其時間占比。[Liger的同類融合實作](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_kl_div.py)

5. **NorMuon更新被所有DDP ranks重複計算。** 舊學生optimizer累計1,854秒，占compute區間10.9%。同卡最佳方案每update仍約127ms。不過把整個optimizer假設成免費，以本次rank0事件從最慢rank時間扣除的樂觀敏感度估算，6B純訓練仍約72.56H。因此它值得優化，卻無法單獨達標。

另外，既有同GPU雙stream pipeline試驗只有約1.6%速度改善：teacher事件從154ms延長到208ms，student從368ms延長到419ms。時間軸重疊不代表GPU運算資源免費；不採用「兩邊時間取max即可」的估算。

## 本次GPU配對測試

Job577467，8H200／8CPU，25a-hgpn069，01:00:39–01:06:39，360秒／**0.8H**。只有seed42，隨機學生，固定官方pretrained3B教師；四組使用相同真實canonical訓練windows，context8192、intradoc mask、micro1、global BS16／64。每組每batch3次warmup＋15次計時。optimizer執行原運算但LR與decay LR置零，避免測速本身產生要交付的訓練模型；初始化／編譯也記入allocation。

以下四組均為同一8卡node上，每卡先跑teacher backbone再跑student，hidden留GPU；不是跨node spool，也不是歷史logits-prefetch pipeline。

| 組合 | BS16 tokens/s | BS64 tokens/s | token加權吞吐 | 峰值allocated |
|---|---:|---:|---:|---:|
| checkpoint＋原loss | 126,654 | 152,859 | 138,528 | 35.92GB |
| 不checkpoint＋原loss | 141,136 | 170,682 | 154,509 | 52.31GB |
| **不checkpoint＋fused loss** | **149,981** | **184,135** | **165,312** | **50.73GB** |
| checkpoint＋fused loss | 135,089 | 163,324 | 147,870 | 34.34GB |

吞吐使用兩階段token權重的調和平均，不平均兩個tokens/s，也不只取較快的BS64。

數值gate通過：FP32/BF16、lambda0/.25/1、忽略部分或全部labels、非整除chunk、不同teacher/student hidden維度；同時比對sealed loss與autograd。完整8192／128256實例：reference loss11.7756404877，candidate11.7756395340；最大rank全模型gradient相對L2差0.002511（0.2511%）。這是BF16等價性的數值篩檢，還不是長程收斂或非零LR多步軌跡的證明。

來源：[彙整數值／timing](analysis.json)、[實際allocation](accounting.json)、[執行契約](protocol.md)、[融合kernel](fused_kd.py)、[benchmark](benchmark.py)。47,316,992學生inputs與47,251,456教師inputs均已計帳，包含重複與warmup。

## 剩餘成本集中在哪裡

最佳方案每個update的rank0 CUDA事件，與max-rank wall time分開報告；事件不能直接推導所有rank的占用率。

| 區段 | BS16／update | BS64／update |
|---|---:|---:|
| 教師backbone | 282.5ms | 1016.0ms |
| 學生body forward | 103.6ms | 401.8ms |
| 完整詞表head＋KD＋head梯度 | 93.6ms | 374.7ms |
| 學生backward，含DDP | 261.0ms | 919.5ms |
| clipping | 4.4ms | 4.3ms |
| optimizer | 127.5ms | 127.0ms |
| **完整wall time** | **873.9ms** | **2847.3ms** |

移除I/O和checkpoint後，主要剩下**教師backbone與學生body forward/backward**。後續工作應針對這些區段；不能再把「換一個KL kernel」當作足夠解法。

## 下一輪具體改進與判定

建議以目前最佳配置為控制，只用seed42，先做下列獨立改動，再合併有收益者。以下尚未執行，不計為已取得的加速。

| 順序 | 改動 | 保留的數學定義 | 必須量測／查核 |
|---|---|---|---|
| 1 | micro1→2、accum2→1與8→4 | 每update仍是同一16／64序列；位置、mask與loss-token全局正規化相同 | 峰值顯存、相同global batch的梯度、教師／學生GEMM時間；不能把擴大global batch當lossless |
| 2 | 分別compile teacher/student的RMSNorm、RoPE、SwiGLU／MLP | 相同BF16／FP32精度規則和attention mask | prefill與backward實際收益、編譯時間、變動document spans是否反覆recompile；不直接compile整個含journal的train loop |
| 3 | NorMuon按**完整矩陣**分配owner；owner更新後all-gather FP32權重 | 同一全局梯度、NS5、動量、逐neuron EMA、WD、LR；矩陣不能切片後各自做NS | 非零LR多步權重／optimizer state比對；通信與packing是否吃掉約127ms的省時 |
| 4 | head-gradient用BF16 GEMM直接累積FP32 buffer，移除逐chunk大型cast/add暫存 | 相同矩陣乘積、FP32累加；不刪除任何梯度 | 逐chunk捨入差、梯度相對誤差與head區間時間；不能假設新版API在已pin環境一定可用 |
| 5 | 編譯支援teacher head-dim128的BF16 FA3 | 保留document-span causal與GQA | 本地現有FA3只支援student head-dim64，需新build與teacher attention數值gate；比較整個teacher，而非只報attention kernel加速 |

第3項採用分散optimizer工作的原理，不能直接把會展平／切開矩陣的ZeRO配置套進NorMuon。需要保留完整矩陣形狀，並將每個state與parameter恰好分配一次。[ZeRO官方說明](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/zero.md)

第4項可參考`addmm(...,out_dtype=torch.float32)`的CUDA支援與Liger累加路徑。[PyTorch addmm](https://docs.pytorch.org/docs/main/generated/torch.addmm.html) 預設CCE有gradient filtering；`cce_exact`才移除相關filter，而且hard-label CE本身不能直接替代完整soft-target KL。[Apple CCE](https://github.com/apple-aiml-research/ml-cross-entropy)

數學契約始終為：

`L = .75 CE(y,p_s) + .25 KL(p_t || p_s)`

`dL/dz_s = p_s - .25 p_t - .75 one_hot(y)`

可把運算融合、重排與分散；仍計算完整normalizer和KL常數項以維持loss日誌。FP8／INT8、top-k KL、換小教師、減少NS步數、擴大global batch、少跑tokens均不列入lossless方案。EAGLE3的autoregressive decoding加速也不能直接加速這種對已知訓練序列的一次teacher-forced forward。

## 64H的硬門檻與停止條件

全部teacher＋student配置若共8GPU，64H=8小時；不含任何overhead時至少需208,333 tokens/s。16GPU配置則要416,667 tokens/s，不能因wall time縮短就稱GPU-hours改善。

採保守預算：`8 × [1.10 × (3e9/r_BS16 + 3e9/r_BS64) + 1800] / 3600`。
30分鐘用於初始化、固定評估、最終額外9k與存檔／export；10%吸收真實資料供應、journal和較長執行的變動。這些是規劃餘量，仍需以正式job的allocation與實際token計數核算。

只算full run需至少244,444 token加權tokens/s；連本次實際0.8H驗證也算入64H則需約247,748。**下一輪目標至少250,000 tokens/s**，對目前165,312還需約51.2%額外提升。250k對應full run62.67H，加這次驗證為63.47H。任何新增驗證成本也要從總額再扣，不能沿用此餘額。

進入正式6B前，必須完成非零LR多步optimizer／資料順序／mask與固定eval比對、長一些的不重複真實資料吞吐、各phase邊界與完整收尾時間檢查。若達不到此門檻，應如實維持「64H未驗證可行」，不靠刪教師成本、縮短token數或省略評估宣稱成功。

本次只完成效能驗證；正式6B尚未提交。18個既有formal identities與全部歷史成本保留。研究目前實際使用173.889722H（含本次0.8H）。新正式6B需另有對應configuration與預算授權，不能挪用先前限定3B或其他6B run的例外。
