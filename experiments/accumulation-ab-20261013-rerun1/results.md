# 梯度累積開關比較

| 梯度累積 | 每 GPU batch | 累積次數 | tokens/s | median update 秒 | 峰值 allocated GiB | PPL |
|---|---:|---:|---:|---:|---:|---:|
| on | 1 | 2 | 260641 | 0.5012 | 30.71 | 912.544943 |
| off | 2 | 1 | 266794 | 0.4894 | 36.92 | 912.926994 |

不累積 / 累積吞吐比：1.0236×；PPL 相對差 0.0419%。
數值檢查：True；建議：inconclusive。
兩組實際 allocation：2.164444 H200-hours。
獨立並行 allocation、每組8 H200 / 8 CPU、8192 context、global batch 16、相同初始化與每 update 的 global sequences。
21–120 updates 的完整 update 計時含資料、token journal、training、logging，排除 evaluation；另記錄 compute throughput。
獨立冷編譯 cache，前20 updates排除；allocation成本仍包含暖機、編譯、評估與checkpoint。
峰值顯存包含 initialization/evaluation/checkpoint；每 rank 原始觀測留在 job logs。
固定1000 reserved documents，未評估完整OJ。120 updates仍在170 warmup內，只能比較效能及短期數值一致性。
