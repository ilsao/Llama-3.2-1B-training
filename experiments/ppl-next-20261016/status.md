# 兩階段實驗：已完成

核對時間：2026-10-16T17:27:36.758768+00:00。四個最終Slurm作業均COMPLETED、exit0；目前沒有執行中的本研究job。

所有run均從seed42隨機權重開始，完成1,000,079,360 non-padding inputs、4,775次更新；評估為固定1,000文件／1,479,302 prediction tokens。四組初始化、來源、環境與資料provenance由controller核對一致；第二階段使用第一階段勝出的文件內causal mask。

| 階段／方法 | Job | Raw PPL | Token EMA | 末段平均 | 累積H200-hours | W&B |
|---|---|---:|---:|---:|---:|---|
| 1／跨文件 causal＋NS5 | 559883 | 20.20459 | 20.21980 | 20.21729 | 10.8867 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/b55f54803c094cf2a28df1c44a3aa03f) |
| 1／文件內 causal＋NS5 | 559884 | 19.42737 | 19.44173 | 19.43929 | 10.8444 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/7b22c06800d7441785030ef9b2ecf575) |
| 2／文件內 causal＋PE-only | 560141 | 19.44875 | 19.46288 | 19.45999 | 10.8356 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/68a0cd1a7ff84e98b34f749be73f0679) |
| 2／文件內 causal＋NorMuon-only | 560143 | 19.14916 | 19.16341 | 19.16072 | 10.8156 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/5129ddc7b2194999b26712f6b1b46550) |

文件內mask相對跨文件mask改善3.85%；NorMuon相對文件內NS5再改善1.43%，相對最初dense control合計改善5.22%。PE相對文件內NS5增加0.11% PPL，未勝出。兩種平均在四個run都略遜raw。

實際合計**43.382222 H200-hours**，包含最初四筆失敗allocation的0.315556H，低於64H上限。每個完成allocation約80–81分鐘。第一階段於2026-10-17 00:01(Taipei)完成，第二階段00:02開始，01:23完成。

這些是單seed、固定1k開發子集的結果；尚未做額外9k確認、seed43或完整50k評估，不與歷史3B full-set PPL直接排名。未自動啟動新訓練。

[機器可讀結果](results.json) · [第一階段選擇](stage1-selection.json) · [控制器終態](controller-state-r2.json) · [執行設計](two-stage.md)
