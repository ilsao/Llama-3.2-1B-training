# 4B訓練與完整評估完成；共享ledger核銷受阻

目標4,000,055,296非padding inputs／19,080次更新均已完成，八rank訓練exit0。

| 指標 | 實測 |
|---|---:|
| 固定1k subset PPL | 13.86035745 |
| 完整50k PPL | 13.83802096 |
| 相對原3B完整PPL改善 | 9.03% |
| Allocation時間 | 5h16m45s |
| 實際H200-hours | 42.233333 |

Slurm job560415保留FAILED/exit1：錯誤發生在所有訓練、HF export/reload、完整50k評估與W&B同步完成後的receipt歷史ledger檢查。共享experiment-ledger.json目前jobs/runs均空，無法核銷；未修改共享ledger，也未重跑訓練。

封存source/config hash、實際初始化/資料provenance、完整4B訓練計數、所有rank退出碼、HF reload與exp(NLL/tokens)=PPL均獨立驗證通過。不同token budgets與單seed限制仍適用。

[獨立審核結果](audited-results.json) · [ledger異常證據](ledger-anomaly.json) · [W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/302d493de0854a3b816a2bee7b5d2a4f)
