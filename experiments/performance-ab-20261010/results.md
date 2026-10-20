# 原版與整合優化版實測

| 版本 | 端到端 tokens/s | median update 秒 | PPL |
|---|---:|---:|---:|
| A | 197109.3 | 0.6524 | 912.852739 |
| B | 262407.7 | 0.4985 | 912.201729 |

端到端加速：1.331×；PPL 相對差異：0.071%。
2B 投影與 cold-start excess 是估計；compiler categories 為巢狀計時，不互相加總。
Optimizer GPU event 時間和原版 host wall 時間口徑不同；主要比較完整 update 時間。
評估範圍：固定 1000 reserved documents，非完整 OJ。
