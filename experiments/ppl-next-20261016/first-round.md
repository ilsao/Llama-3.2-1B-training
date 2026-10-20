# 第一輪：文件內 attention、獨立 Polar Express／NorMuon、權重平均

> 最新執行方式已依使用者指示改為[兩階段、每階段兩個平行 sbatch](two-stage.md)：先選 causal mask，再於勝出 mask 比較 PE-only／NorMuon-only。四個 job 各120分鐘，總上限64H，包含內嵌preflight。下文原四組設計與70/72H估算僅保留為設計歷史。

本文件細化第一輪，並依使用者要求加入獨立 Polar Express 與獨立 NorMuon，形成四組。狀態仍為 design-only；未提交 GPU 作業。第一輪回答「在同一已選配方下，單一方法是否改善」，不等同於對每種方法各自搜尋最優超參數。

## 四個 fresh training arms

| Arm | 訓練 attention | Muon 正交化 | 相對 A 唯一方法改變 |
|---|---|---|---|
| A dense_control | 原 packed cross-document causal | 原 native NS5 | 對照 |
| B intra_document | 按 document_spans 分塊 causal | 原 native NS5 | attention 可見範圍 |
| C polar_express_only | 與 A 相同 | PE5 | 正交化子程序，含其數值正規化 |
| D normuon_only | 與 A 相同 | 原 native NS5 | 正交化後的 neuron-wise 二階矩正規化 |

四組均從相同 seed42 隨機初始化開始，使用相同 raw prefix、token 順序、loss masks、batch windows 和 scheduler。B 仍使用 8192 packed context，不重新 tokenize、重排文件或更改位置 ID。C 保持原 Muon 外層更新，不引入 NorMuon 二階矩、cautious decay、row equilibration 或額外 RMS normalization。D 保留原 NS5，只加入下述 NorMuon 正規化；不呼叫 PE。

先比較 B.raw−A.raw、C.raw−A.raw 與 D.raw−A.raw。每組另輸出兩個平均權重，形成十二個 endpoints；平均是同一軌跡的讀出，不能稱為十二個獨立訓練樣本。第一輪不自動加入組合。若需要確認 attention×PE，可補 E=intra-document+PE5，交互作用為 NLL(E)−NLL(B)−NLL(C)+NLL(A)。若需要確認 NorMuon×PE，可補 F=dense+NorMuon+PE5，交互作用為 NLL(F)−NLL(C)−NLL(D)+NLL(A)。每個補充 arm 另需16H；兩項各自改善不保證組合改善。歷史 NorMuon+PE 用舊20% tail與不同外層實作，不能充當 F 的匹配格。

## 固定配方與時間軸

| 項目 | 預先固定的值 |
|---|---|
| 訓練輸入 | 1,000,079,360 non-padding tokens，7630 個 BS16 等價單位 |
| 架構／初始化 | Llama-3.2-1B，residual-half 隨機初始化，seed42 |
| 資料 | dolma-raw-baseline-20261008，同 canonical seed42 split 與 training prefix |
| Context／精度 | 8192；既有 FP32 parameters、BF16 autocast、FA3 |
| 分配 | 每組 8 H200、8 CPU，1 launcher，8 ranks；依序執行 |
| Batch | 每 rank microbatch1；accum2→8，即 global sequences16→64 |
| Batch 切換 | 500,039,680 inputs 後的下一個 batch |
| Phase | 0→833,355,776→1,000,079,360 inputs；第二 peak 為第一的 .1 倍 |
| Warmup | 每 phase 都是 170 個 BS16 等價單位＝22,282,240 inputs |
| Tail | 每 phase 的末 40%；ramp／hold／final=4:1:1，hold LR=.2×phase peak，末端歸零 |
| Peak LR | Muon .008614891542046706；AdamW .00025844674626140116 |
| Optimizer | Muon momentum=.95、Nesterov；auxiliary AdamW betas=(.9,.95) |
| Decay／clip | Muon WD=.1、embedding/head/norm 無 decay；global clip=.5 |

所有 schedule 都按已處理 non-padding inputs 前進。第一 phase 的 40% tail 約從 0.5B 開始；第二 phase 在約 0.833B 重新 warmup，再完成自己的 cooldown。LR 來自既有 horizon calibration，是一致的起點，並非新配方的已知最優值。

event 邊界只允許以完整 BS16 等價單位縮短 accumulation。新的 planner 要預先列出每次 batch 的 tokens、LR、phase 與評估事件，鎖定實際 update 數，再確認 eval_steps=400 不超過總 updates 的 10%。尚未產生這份執行級 batch plan，不把規格 JSON 當成可直接提交的 config。

## 為何獨立 Polar Express 值得加入

歷史 control PPL=20.4452、NorMuon+PE=20.6598，但後者同時改了二階矩／更新尺度等規則，而且用的是舊 20% tail 配方。它只能判定該組合當時沒有勝出，不能判定 PE-only 無效。

Polar Express 的原論文針對 Muon 內的 polar approximation，報告 GPT-2／FineWeb 1–10B tokens 的 validation-loss 改善。這是加入實驗的理由，並非 Llama／Dolma 會改善的保證。[原論文 v5](https://arxiv.org/abs/2505.16932v5)、[作者程式](https://github.com/NoahAmsel/PolarExpress)

第一輪選定已存在的 muonium 0.2.0 **PE5 變體**作為對照，便於與舊組合對應；不混用不同實作的係數與 safety factors。其係數表為：

```text
(8.156554524902461, -22.48329292557795, 15.878769915207462)
(4.042929935166739, -2.808917465908714, 0.5000178451051316)
(3.8916678022926607, -2.772484153217685, 0.5060648178503393)
(3.285753657755655, -2.3681294933425376, 0.46449024233003106)
(2.3465413258596377, -1.7097828382687081, 0.42323551169305323)
```

先將輸入 momentum update 轉為與既有 NS 相同的運算 dtype，按寬高決定轉置；正規化為 `X/(1.02*||X||F+1e-6)`。每輪依序用其 (a,b,c) 做 `A=XXᵀ; X=aX+(bA+cA²)X`，總共五輪。原 NS5 使用同一組 (3.4445,−4.775,2.0315) 重複五次。兩者矩陣乘法結構相近，但總 optimizer／training step 的速度必須實測。

這個 PE5 表與 normalization 是目前本地套件的具體變體，並非宣稱與作者目前 main 的預設（10 組生成係數、另一 safety setting）完全一致。參考檔案：`/work/asciibase64/Llama-3.2-1B-training/environments/batch-normuon-1b-20261016/muonium-0.2.0/muonium/orthogonalize/polar_express.py`。執行前需封存實際 adapter 與參考檔案 SHA256，不改已封存的歷史實驗。

實作邊界是沿用 native torch 2.9.1 Muon 的 momentum、Nesterov、`adjust_lr_fn=original`、decay 次序與 state semantics，僅替換正交化子程序。不要直接換整個 muonium optimizer，否則又引入額外差異。先讓 adapter 的 NS 模式逐步重現 native Muon，再切 PE 模式。

PE 將改變輸出的奇異值與 update RMS；這是方法效果的一部分。首輪不額外調 LR 或強制做 RMS 對齊，避免再改第二個因素，但必須記錄 update/weight ratio、矩陣 update RMS 與 norm。若結果顯示差異主要是更新尺度，第二輪再做預先鎖定的 LR／RMS-matched 對照。PE 輸出更接近 polar factor 本身不算 PPL 勝出。

## D：獨立 NorMuon 的精確定義

NorMuon 的核心是對已正交化的更新加入逐神經元二階矩正規化，再恢復原更新的 Frobenius norm。此步驟可以搭配原 NS5，並不依賴 PE。[論文](https://arxiv.org/abs/2510.05491)、[作者實作說明](https://github.com/zichongli5/NorMuon)

令 U 為原 NS5 輸出，已轉回原 weight matrix 的形狀。每個 output row 維護一個 FP32 state，初始化為零：

```text
v = 0.95*v + 0.05*mean(U**2, dim=input_dimension)
V = U / (sqrt(v) + 1e-8)
U_nor = V * (norm_F(U) / max(norm_F(V), 1e-8))
```

這固定為本地 muonium 0.2.0 `_normuon_normalize` 的數值規則，不加 variance bias correction；只移植此 normalization，保留 native Muon 的外層 momentum、Nesterov、LR shape scaling 與 decoupled decay。不能直接換整個套件 optimizer：本地套件對 decay 與 adjusted LR 的結合方式也是潛在額外變因。因此稱為「NorMuon 核心機制的單因素消融」，不聲稱逐行重現整套第三方 optimizer。

beta2 固定每次 update 為 .95，不隨 BS16→64 改成 .95^4；也不重置 v 或 momentum。原 AdamW beta2=.95 是另一個獨立 state，兩者不可混淆。首輪只測這個固定方案，不同時搜尋 beta2/LR。每層記錄正規化前後的 row-norm 分布、Frobenius norm ratio、update/weight ratio，以及實際步耗時。

Correctness gate 要驗證：關閉 normalization 時重現 native Muon；開啟時匹配 FP32 reference；非退化輸入保持矩陣更新 norm；零／極小輸入有限；NS 為計算效率做的轉置必須先還原，v 才能沿真正的 output-neuron 維度累積；八 rank 一致；batch/phase 邊界與 state save/load 都保持 v。這些尚未做 GPU 實測。

## 每組的三種終點讀出

1. Raw：完整 horizon 的最後訓練權重。
2. Token-EMA：從 95% tokens 的完成 batch 初始化 shadow，半衰期為總 horizon 的 1%；之後 `beta=2**(-delta_tokens/half_life_tokens)`，只覆蓋最後 phase。
3. 有限窗口平均：98%、98.5%、99%、99.5%、100% 附近的五個預先鎖定完成 batch，做等權 streaming mean。四組採同一批 token 位置。

平均不回寫 training weights、不多跑 backward，兩個 FP32 shadows 約 9.9GB/rank；更新、匯出與額外評估仍有成本。確認 tied parameters 只平均一次，observer 不消耗 training RNG，不改原軌跡。不得從平均權重開始新訓練。

## Correctness gate、評估與決策

- A/B/averaging gate：文件 span FA3 與明確 block-causal SDPA 的 forward/backward、padding/GQA、擾動前文件不影響後文件、相同 input/label counts、八 rank 一致性、observer trajectory/state/export 檢查。
- PE gate：adapter NS 模式與 native Muon 多步參數／momentum 等價；PE 與固定參考實作比對；正方形／高／寬／零／低秩矩陣；BF16 finite values、eager/compiled parity、八 rank 一致性。小矩陣用 FP64 SVD 作診斷，低秩矩陣不能要求零 singular directions 變成正交。所有矩陣/model runtime checks 經 Slurm。
- NorMuon gate：上述 normalization reference、native parity、原 output-row 軸、norm preservation、零／極小輸入、state 保存及八 rank 一致性。
- 四組每 400 actual updates、共同 10% token knots、phase/batch 邊界及終點評估固定 1k documents。主比較是終點 raw NLL，不以中途最低點選贏家；同軌跡 averaging 另作配對比較。
- 先看三個主效果，再看每組平均相對自己的 raw，避免把 optimizer 效果與 averaging 混為一談。十二個 endpoints 的選擇有多重比較偏差，完整揭露全部結果。
- 選擇規則：候選相對 A.raw 的 ΔNLL≤−.005，約 .5% PPL；若約 PPL20，約為下降 .10。在 1k 上鎖定最多一個完整候選，再於原 10k 子集中其餘 9k 比較該候選與 A.raw；確認同方向、報 per-document paired-bootstrap CI。若 9k 曾參與過去選擇，標記重用 validation。不得看完 9k 後反覆換候選。
- 只有吞吐改善時，記為效率結果，後續另設固定算力比較。orthogonality residual、train loss 或 throughput 不能替代 held-out NLL。
- 有訊號才 fresh seed43 重跑 control／鎖定候選，之後再考慮 3B。1B 勝出不是 3B 必勝；兩 seed 也不是充分的顯著性證明。

## 配額與順序

| 項目 | 每 job 上限 | H200-hours 上限 |
|---|---|---:|
| A/B/averaging gate | 8 H200、8 CPU、15min | 2 |
| 獨立 PE gate | 8 H200、8 CPU、15min | 2 |
| 獨立 NorMuon gate | 8 H200、8 CPU、15min | 2 |
| A、B、C、D 四個 fresh 1B runs | 每組 8 H200、8 CPU、120min | 64 |
| 第一個篩選決策點 | 四組各三個 1k endpoints | **70** |
| 條件式 9k 確認評估 | 8 H200、8 CPU、15min；只讀已鎖定 candidate/control 權重做 eval | 2 |
| 篩選＋確認設計上限 | 不包含 seed43／3B | **72** |

相對前版 PE 三組方案，增加 NorMuon fresh training16H＋獨立 gate2H，篩選上限由52H升至70H。9k 確認仍明列2H，不假設已結束的 training allocation 可以免費重用；若不進確認就不使用該 allocation。可選 E／F 各額外16H；seed43 配對額外32H，均不自動啟動。

每 training allocation 預留最後20分鐘給三種終點評估／輸出／停止餘裕，總長仍不超過120分鐘。gate 先驗證 1B 能在剩餘時間完成；若不能，先改全部 arms 的共同計畫或停止，不拿不同 tokens 的終點硬排名。gate／training 失敗不自動重試；每筆失敗與閒置成本照記帳。這些是設計上限，不是實際成本或新增授權記錄。
