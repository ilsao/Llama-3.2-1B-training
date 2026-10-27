# 6B 結果與選擇依據

[6b-comparison.json](6b-comparison.json) 比較三個已完成的6B run，選定 formal19 R5 KD。
以同一固定10k的 `exp(sum NLL / sum prediction tokens)` 排名；不是平均 PPL，
也不是用最新 run、training loss 或不同 token budget 排名。全保留50k尚未評估。

`evidence/` 保存從 `/work/.../runs/experiments/<run>/summary.json` 複製的小型原始 metadata，
來源路徑及 SHA256 記在比較檔。非 KD 的 additional9k 引用既有
[results.json](../experiments/raw6b-normuon-adam2x-20261018/results.json)。
這讓一般 Git checkout 不依賴 /work 掛載也能核對選擇；不包含模型權重或新增成本帳本。

在 repository root 執行 `make check-sota` 可驗證 evidence、PPL、endpoint、SOTA source
seal、configuration digest、symlink 目標、checkpoint 與已發布模型 revision。
此驗證只核對保存的證據，不重新跑模型評估或遠端模型下載。
