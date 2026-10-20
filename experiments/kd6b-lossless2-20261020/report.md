# 第二輪 lossless 蒸餾加速實測

2026-10-20，作業577962完成。最快的合併組達219,104加權tokens/s，比本輪原始實作快57.5%，並通過本次短程PPL不升高的篩選。**尚未證明最終6B PPL不退步，也尚未驗證完整6B能在64 H200-hours內完成。**

六組均使用seed42相同隨機初始化、相同canonical資料順序、相同global batch和原3B學習率排程前綴，跑128次非零LR更新／16,777,216學生inputs。終點使用相同eager evaluator、固定1,000文件／1,479,302 prediction tokens。仍處於warmup；這些PPL不能與已完成3B訓練的PPL直接比較。沒有第二個seed。

| 組別 | 改動 | 短程1k PPL | 相對R0 | 加權tokens/s | 6B預估H，含餘量 |
|---|---|---:|---:|---:|---:|
| R0 | 原始checkpoint、eager loss | 806.217433 | — | 139,117 | 109.43 |
| R1 | 關閉checkpoint＋融合完整詞表CE/KL | 806.085866 | −0.131567 | 168,129 | 91.23 |
| R2 | R1＋microbatch 2 | 806.294757 | +0.077324 | 171,064 | 89.74 |
| R3 | R1＋局部compile | 807.289076 | +1.071643 | 191,956 | 80.41 |
| R4 | R1＋按完整矩陣分散NorMuon更新 | 805.791045 | −0.426388 | 182,481 | 84.37 |
| R5 | R1＋microbatch 2＋compile＋分散更新 | 805.836726 | −0.380707 | 219,104 | 70.94 |

R2與R3沒有通過預先設定的嚴格短程PPL門檻。R1、R4、R5通過，但不宣稱它們在品質上有顯著改善，也不把單一seed短程結果當成最終PPL保證。R5相對R1另快30.3%；相對R0的PPL差異僅−0.0472%。

## 保留的數學定義與數值檢查

教師仍為官方預訓練Llama3.2-3B，revision `13afe5124825b4f3751f836b40dafda64c1ed062`。學生仍為隨機residual-half Llama3.2-1B。完整128256詞表的`.75 CE + .25 KL(teacher || student)`、T=1、BF16、8192 context、document-span causal mask、全域BS16/64、梯度正規化／clipping、NS5 NorMuon、EMA、普通WD與AdamW LR×2均保留。教師與學生共同使用同一組8張H200，無磁碟hidden-state spool。

microbatch 2只改變每rank分組；accumulation從2/8改成1/4，同一更新仍處理同一組全域16/64序列。compile僅包住RMSNorm、MLP與RoPE，attention不變，evaluation全部用eager。完整矩陣分配給owner執行原NorMuon步驟，再all-gather FP32權重；不將NS矩陣切片。仍對所有學生參數清梯度和做全域clipping，AdamW保持replicated。

12步非零LR的分散optimizer檢查中，權重、momentum及variance state與原實作逐bit相同。這是給定相同梯度的合成矩陣測試，不是全模型訓練軌跡逐bit相同的證明。六組真實全模型梯度門檻均通過；R5相對R0的梯度relative L2約0.552%，loss差約−0.0000364。這些檢查與BF16運算順序容許值相容，仍須以訓練後PPL驗證品質。

逐文件NLL重算與原始eval紀錄一致；六組每一文件的prediction-token數一致，定期evaluation步數均為12、24、…、120、128。[品質核對](quality-audit.json)保存結果。R5的1k文件中580份NLL較低、420份較高，不代表跨seed或長程非劣性證據。

## 時間與剩餘瓶頸

R5 BS16實測209,677 tokens/s，BS64實測229,417 tokens/s；依前後各3B tokens取調和平均得到219,104 tokens/s。BS16來自非零LR訓練步驟；BS64在品質評估後以13個後續真實窗口測速，前三個warmup、後十個計時，LR與decay LR為零但完整optimizer計算仍執行。計時包含線上教師、學生前後向、clipping、optimizer與權重同步；排除資料準備、token journal及evaluation。

R5 optimizer在BS16/64約31.66/31.38ms，R1約127.73/127.70ms，包含owner通信後仍省約75%。它在R5總step時間的占比僅約5.1%／1.4%，剩餘計算主要在教師與學生前後向。峰值allocated memory為70.24GB／65.42GiB。編譯收據記錄12個unique graphs，完整作業正常完成。[完整速度數據](status.json)

6B純訓練外推：`8 × (3e9 / 209677.1068 + 3e9 / 229417.4096) / 3600 = 60.8540 H200-hours`。

64H若只計正式作業，裸計算只留下約23.6分鐘給初始化、完整loop開銷、固定評估、最終額外9k評估、checkpoint及export。現有保守模型保留10% runtime餘量和30分鐘初始化／收尾，故完整作業預估70.9394H。這是規劃假設，**不是實測完整6B耗時，也不是不可達64H的證明**。

本輪2.288889H，前輪0.8H，兩輪實測驗證合計3.088889H；若全部包含在64H內，完整方案保守預估74.0283H，吞吐需達257,712 tokens/s，比R5再快17.6%。只計正式作業且沿用上述餘量時，需244,444 tokens/s，比R5再快11.6%。不能僅以60.85H的純訓練估計宣稱達標。

下一項最有判別力的驗證是：對R5量測完整訓練loop、journal／資料供應與實際評估、存檔開銷，釐清剩餘23.6分鐘是否足夠。若需更高kernel吞吐，再考慮teacher head-dim128 FA3及head梯度FP32累積；兩者本輪均未實作或量測。任何後續方案都須保留完整詞表KD並重做數值與品質檢查。

## 執行與記帳

Slurm577962於Asia/Taipei 01:30:45–01:47:55執行，共1030秒，8 H200／8 CPU，實際2.288889 H200-hours，狀態COMPLETED；低於本輪4H上限。教師與學生各處理142,344,192 inputs，含全部重複組別、數值檢查與warmup，兩個journal皆無pending transaction。共享study累計176.178611H，保留18個formal identities與所有歷史成本，低於191.419H。

沒有啟動正式6B訓練、載入訓練checkpoint、重試、清理舊artifacts或上傳HF。執行來源seal核對通過。程式與契約：[benchmark.py](benchmark.py)、[accelerations.py](accelerations.py)、[protocol.md](protocol.md)；收據：[accounting.json](accounting.json)、[verification.json](verification.json)。

R5 W&B：[kd6b-lossless2-R5_combined](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/7b8f713dc72a)。各組W&B及全部原始輸出位置記錄於[status.json](status.json)。
