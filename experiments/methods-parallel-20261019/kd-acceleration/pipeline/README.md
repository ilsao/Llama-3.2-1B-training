Online 蒸餾 pipeline 已實作在 `online_pipeline.py`，使用官方預訓練 Llama 3.2 3B 教師與隨機初始化的 Llama 3.2 1B 學生。每張 H200 保留一份教師、一份學生；學生維持8-rank DDP、microbatch1、global BS16／accumulation2。沒有啟動正式訓練。

**完成結果：** job573794 通過完整模型驗證。在確定性FA3 backward下，pipeline與inline的8卡完整梯度relative L2=0、cosine=1，重複inline也完全一致。吞吐計時恢復原本的非確定性backward，所有模式設定相同。兩個反向排序區塊的結果如下，包含pipeline填充與排空：

| 模式 | 秒／update | peak allocated／卡 |
|---|---:|---:|
| Inline串行即時計算 | 1.04255 | 33.636 GiB |
| 串行雙緩衝 | 1.04489 | 37.503 GiB |
| Pipeline雙緩衝 | 1.02843 | 37.503 GiB |

Pipeline相對inline的短測吞吐提升1.37%，相對串行緩衝提升1.60%；峰值顯存增加3.867 GiB。每個模式只有兩個6-update區塊，未證明長時間收益或統計顯著性。功能與數值正確，但效能改善有限，不能宣稱接近2倍加速。原始量測在 `results.json`、`timings.json`；完整梯度結果在 `full-gradient-check.json`。

Rank0 CUDA事件顯示next-teacher與current-student的stage區間平均重疊123.3ms。但教師stage從串行的154.3ms增加至207.7ms，學生stage從368.0ms增加至419.5ms，符合共用運算／HBM資源時的競爭；這是根據stage計時的推論，不是SM級profiler證明。相互重疊所省的時間，大多被階段耗時增加抵銷。詳見 `stage-timing-analysis.json`。

| 階段 | 教師 stream | 學生 stream |
|---|---|---|
| 填充 | 計算 batch0 的 hidden 與完整 logits | 等待 batch0 ready |
| 穩態 | 預取 batch n+1 的 hidden 與完整 logits | batch n 的 forward、KD loss、backward，依原累積邊界更新 |
| 排空 | 不再預取新資料 | 完成最後一個 batch，釋放緩衝 |

這裡 batch 指每卡 microbatch。教師凍結，因此學生參數更新不會使下一批的教師輸出失效。尚未處理的訓練資料也不能改變學生當前 batch 的輸入、遮罩或梯度。

兩個固定 BF16 緩衝各為 `8191 × 128256`，合計每卡4,202,179,584 bytes，約3.914 GiB。這是短暫 GPU 記憶體，不是離線 logits 資料集。教師 projection 以1024 positions為塊寫入緩衝；學生使用完整詞彙 `.75 CE + .25 KL(teacher || student)`、T=1，在FP32計算softmax／loss，並在forward產生head梯度以避免重算。

同步約束：教師先等待輸入ready；學生只等待當前ticket的teacher-ready；重用slot前教師等待上一個consumer-release。`record_stream`處理跨stream張量生命週期。以遞增batch key、tensor identity／version檢查防止亂序、重複使用與輸入被修改；緩衝上限固定2。這依循PyTorch對非預設streams的同步與生命週期要求。[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)、[record_stream](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html)。

訓練整合應先按原token cap、batch schedule與資料cursor規劃完整窗口，再prefetch；只有學生實際消費才能前移學生cursor。教師每次prefetch都計入teacher inputs；停止時未消費的預取也保留成本。評估、checkpoint與停止前drain；若丟棄pending輸出，下次重新生成仍要再記教師成本，不將佇列當作已完成訓練。維持原DDP `no_sync`範圍與全域有效labels正規化。

每張卡同時使用兩條streams只能提供重疊的機會，兩個模型仍共用SM、Tensor Cores與HBM，不能假定時間會由 `T_teacher + T_student` 自動降成 `max(T_teacher,T_student)`。後者只適用於有足夠獨立資源的理想情況。若改成4張教師卡＋4張學生卡，學生DDP大小、梯度累積及GPU間logits傳輸都會改變；不能直接當成同配置的免費加速。本次先保持原8張學生卡比較。

官方3B教師資產已由job573586下載，固定revision `13afe5124825b4f3751f836b40dafda64c1ed062`；所有權重檔案hash、token-ID完整映射、tokenizer處理元件、特殊token與多語言編碼探針皆驗證。大型檔案只放在 `/work/asciibase64/Llama-3.2-1B-training/caches/kd-teachers/llama32-3b-13afe512-pipeline-20261019`。資產準備175秒×1H200=0.048611 H200-hours，列為preprocessing。

首次完整驗證job573687先通過GPU數值／佇列測試，但目前安裝的FA3僅編譯head dimension64，無法執行3B教師的head128。該次54秒／0.12 H200-hours與保守input reservations保存在 `attempt1/` 及ledger。修正採用目前固定PyTorch2.9.1自帶的flash varlen ATen operator，僅用於凍結教師，保留document-span causal mask；學生仍用原FA3。此私有介面必須與環境版本一起固定，並先對照FP32 dense causal SDPA測試GQA、padding與跨文件隔離。[PyTorch2.9.1實作](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)。

配對驗證比較 `inline`（原online串行pregrad）、`serial_buffer`（同樣緩衝但不重疊）與 `pipeline`。同一組初始權重、固定canonical資料窗口、相同global batch與loss；LR及外部decay LR均為0，沒有更新模型權重。先比較完整8卡梯度（relative L2<1e-4、cosine>.999999、loss absolute error<1e-5），通過後做兩個順序相反的區塊，每個模式每區塊6個updates，計入fill/drain並記錄峰值顯存。這只能驗證數值／效能，不是PPL或收斂實驗。

修正驗證job573750仍屬同一logical validation run，600秒上限；連同失敗的54秒，最壞654秒／1.453333 H200-hours，低於原720秒／1.6 H200-hours。未載入任何學生checkpoint。量測結果完成後保存於本目錄。

job573750實際56秒後在嚴格梯度門檻失敗，relative L2=.00220964、cosine=.99999762。後續job573794保留門檻，增加repeat-inline控制並切換確定性FA3 backward，結果repeat及pipeline差異均為0；效能測試再恢復原設定。這支持先前差異來自非確定性執行，不能將其誤當作pipeline的loss／batch錯配。所有失敗紀錄、原始source與counter保留於 `attempt1/`、`attempt2/`；最終執行輸入於 `completed-execution/`。

最終Slurm accounting：assets175秒×1H200；驗證54、56、131秒各8H200（包含Slurm收尾等待）。本次總成本 **0.584167 H200-hours**，其中GPU驗證0.535556，低於1.6上限。全部allocation已terminal並對帳。成功allocation完整處理5,898,240個學生inputs與同量教師inputs；包括前次失敗預約窗口的保守累積上限為6,291,456，原始results中的input欄位採此上限，不應當作精確實耗tokens。沒有任何optimizer權重更新，沒有正式訓練。詳見 `accounting-summary.json`、`result-summary.json`。
