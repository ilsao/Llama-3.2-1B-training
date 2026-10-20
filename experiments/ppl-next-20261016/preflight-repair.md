# Preflight reference repair r1

Stage1 jobs559836(dense)與559837(intradoc)分別於23秒、22秒停止，兩者都在合成attention gate，沒有開始完整Llama初始化／資料訓練。shared ledger已保留FAILED狀態及0.0511111／0.0488889 H200-hours，合計0.10H。原source/、execution-seal.json、controller-state.json與failure-initial.json保持不變。

第一次gate在FA3 varlen輸出與預設SDPA reference的padding query比較失敗；最大誤差位於第28個位置（有效輸入只有0..22）。FA3 unpack明確補零，而該reference的全遮罩padding query回傳非零。修正版使用FP32 math SDPA，將padding query的reference輸出與梯度明確歸零；有效token仍逐元素比較原.03 forward／.05 backward容許誤差，並獨立斷言FA3 padding輸出為零。tiny Llama reference也固定math backend。原生NS adapter八rank數值等價檢查在首次gate失敗前已通過；完整修正版GPU gate仍待新allocation驗證。

source-r1/是獨立修正副本，不覆寫已封存source。修正後使用同一logical run ID、相同配置與seed，沒有checkpoint讀取。一次性的修復分配僅允許這兩個確定FAILED、未建立training run目錄的job，要求先核銷成本；每個後續allocation最多7140秒(119min)。累計最大8*(23+7140)/3600與8*(22+7140)/3600，均小於原16H／7200 allocation-seconds上限。加上stage2各7200秒，整個study最壞累計63.8333H，仍低於原64H。

這是本次實作錯誤的有界修復，不啟用generic automatic retry。r1若失敗，controller停止新提交，不再重跑。實際allocation筆數會從原四筆增加至最多六筆，logical training runs仍為四個；不得丟棄兩筆失敗成本。新submission owner為source-r1/slurm/ppl_controller.py，狀態寫controller-state-r1.json，仍每階段最多兩個active jobs。所有候選設定、研究目標與stage1選擇規則保持一致。
