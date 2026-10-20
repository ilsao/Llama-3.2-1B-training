# Residual-half＋resplit 2B 對照

狀態：complete；這個 run 最多32 H200-hours，包含在原256H總預算。

EMA 關閉，從隨機權重開始。沿用當輪最終 LR、Muon、seed、data 和 batch。

完整50k PPL：原配方 16.854164；residual-half＋resplit 16.606456。

[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/9e4b50ad065648cdbef83e0e17e5fbc3)
