# 完成當輪結果後的 residual-half＋resplit 對照

使用者於2026-10-15要求：本輪最終結果出來後，不加入EMA，加入residual-half與resplit再測一次。
這裡安排一個額外的完整2B run，測兩項合用效果；不宣稱已拆分單項交互作用。

- Run：init-recipe-20261014-followup-rh-resplit-2b，正式run8的限定例外。
- 等候init-recipe-20261014-final-2b完成full50000-document evaluation、HF reload verification和Slurm費用結算。
- 從seed42隨機初始化；不載入任何training checkpoint，不自動retry/resume。
- 沿用當輪最終實際選定的AdamW/Muon峰值LR、optimizer groups、WD、clipping、8192 context、batch、tokenizer、raw data和固定evaluation indices。
- 唯一數值設定差異：o/down projection初始化std減半；WSD改為兩段，分界12715/15259，第二段峰值為第一段0.1倍。兩段保留60%尾段/20%LR平台與本輪warmup token數。
- 第一段峰值完全沿用control，不因第一段horizon較短而調高。EMA固定關閉。
- 同樣停止於跨2B的完整batch。8H200/8CPU，單次4h/32H allocation，包含初始化、訓練、eval、checkpoint、export、full evaluation。
- 32H納入原init-recipe-20261014的256H總預算，不另開免費budget。保留所有舊成本與失敗費用。
- 限定解除此run的正式run數與120分鐘提交門檻；其他gates保持。
- 排除001/027/062/071/083/151節點。提交結果不明保留reservation並recover，不重送。
- 只在control有完整有限full-set PPL、正確2B token count、eval token count和全allocation結算後提交。Control失敗則不提交。
- 輸出同一full50k上的NLL/PPL差、W&B、checkpoint與實際GPU成本。不清理artifacts，不上傳HF。

獨立封存source保留原數值訓練payload；training_state僅新增run8的精確配置授權。
原controller與原訓練source不修改。後續controller每分鐘讀取原結果，GPU提交時段不重疊。
