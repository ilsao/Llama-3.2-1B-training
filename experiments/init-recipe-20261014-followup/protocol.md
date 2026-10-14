# 並行的 residual-half＋resplit 2B 對照

使用者於2026-10-15要求：本輪最終結果出來後，不加入EMA，加入residual-half與resplit再測一次。
使用者隨後要求無訓練依賴就並行啟動，取代先前等結果的順序安排。
這裡安排一個額外的完整2B run，測兩項合用效果；不宣稱已拆分單項交互作用。

- Run：init-recipe-20261014-followup-rh-resplit-2b，正式run8的限定例外。
- 沿用已提交且正在訓練的formal7不可變配置，與其並行；不需要formal7 checkpoint或結果來決定本run配方。
- 從seed42隨機初始化；不載入任何training checkpoint，不自動retry/resume。
- 沿用當輪最終實際選定的AdamW/Muon峰值LR、optimizer groups、WD、clipping、8192 context、batch、tokenizer、raw data和固定evaluation indices。
- 唯一數值設定差異：o/down projection初始化std減半；WSD改為兩段，分界12715/15259，第二段峰值為第一段0.1倍。兩段保留60%尾段/20%LR平台與本輪warmup token數。
- 第一段峰值完全沿用control，不因第一段horizon較短而調高。EMA固定關閉。
- 同樣停止於跨2B的完整batch。8H200/8CPU，單次4h/32H allocation，包含初始化、訓練、eval、checkpoint、export、full evaluation。
- 32H納入原init-recipe-20261014的256H總預算，不另開免費budget。保留所有舊成本與失敗費用。
- 限定解除此run的正式run數與120分鐘提交門檻；其他gates保持。
- 排除001/027/062/071/083/151節點。提交結果不明保留reservation並recover，不重送。
- 提交前驗證control配置SHA、ledger正式run7登記與256H總成本（包含尚未結算allocation的完整reservation）。兩個run均完成後才產生full-set比較。
- 輸出同一full50k上的NLL/PPL差、W&B、checkpoint與實際GPU成本。不清理artifacts，不上傳HF。

獨立封存source保留原數值訓練payload；training_state僅新增run8的精確配置授權。
原controller與原訓練source不修改。後續controller每分鐘結算各自allocation；不設Slurm dependency。比較報告等兩個run的完整結果與結算再更新。
