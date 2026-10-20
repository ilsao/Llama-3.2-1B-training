Dion3 數值檢查失敗後的重試，2026-10-19。使用者已要求「重試失敗的」，本次僅D1、D025各一次，Job572264、572265。

原編譯路徑在12步合成更新後出現跨rank差異：D1最大3.72529e-9，D025最大1.86265e-9，兩者均在完整模型初始化前停止。這些數值只能證明原逐位相等gate失敗；尚不能把根因確定為編譯器，也不能評價方法的PPL。

本次保留官方vendor、NS5、optimizer/LR/WD、seed42、data、mask、token endpoint與兩組原config digest。以隔離source-dion-retry1啟動，TORCH_COMPILE_DISABLE=1使用官方eager運算。gate不放寬容差：先驗證初始參數逐位一致，合成梯度顯式廣播以模擬DDP已歸約梯度；保留原12步、狀態roundtrip與mask/data檢查，增加(2048,2048)、(512,2048)、(8192,2048)、(2048,8192)實際layer矩陣的5步更新、參數及optimizer state跨rank逐位一致性與state roundtrip。只有全部通過才隨機初始化並訓練完整模型。結果記錄於各job的dion-eager-real-shapes.json和ppl-gate.json。

原source與6個進行中作業不變；新source改動限parallel_runtime.py、parallel_gate.py、train.py及slurm/training-common.sh。train將eager模式寫入來源provenance；worker的帳本綁定等待延長為有上限的120秒，以容忍Slurm回覆延遲。提交逾時只按唯一tag尋回原job，不重送sbatch。

每組8 H200/8 CPU、單一launcher、7080秒，排除25a-hgpn001。D1先前累計68秒、D02565秒，因此各logical run仍低於7200秒/16H200-hours；整批包含所有歷史失敗成本與本次最大allocation為127.2533333H200-hours，小於128。相同logical run只延續資源記帳；學生不讀取checkpoint，無自動後續重試。

提交前完成Python AST、shell語法、舊新source seal、原始config和已記帳失敗allocation檢查；重送/未記帳allocation的拒絕條件已用標準函式庫測試。GPU驗證僅在上述兩個Slurm作業內執行。

實測結果：兩組全部嚴格gate通過，包含4種真實矩陣形狀與optimizer state逐位一致。已確認D1至少40個更新、D025至少50個更新並正常記錄W&B；全部8組目前均在訓練。這是eager模式加上明確共同測試梯度的聯合驗證，未單獨隔離原編譯路徑或測試RNG對舊誤差的貢獻。詳見validation.json與../launch-status.json。
