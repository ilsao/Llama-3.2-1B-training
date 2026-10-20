# LR horizon transfer

階段：wsd_check_failed；study 138.262/79.469 H200-hours；共享pool 314.449/256。

固定1.236B dense model、residual-half、raw、8192 context、batch16、seed42。只測固定AdamW/Muon比例的共同LR倍率。
先15分鐘GPU正確性驗證（16篇），再4個15分鐘穩定性screen、4個94分鐘1B stable proxy、3個45分鐘400M完整單段WSD。舊驗證失敗成本保留。
EMA beta0.6每32updates更新一次，只供proxy量測；單段WSD validation不使用EMA。

- decay0: job 550805，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/ad3594be016d4f90a45aa492cdc53998)
- decay1: job 550806，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/4f2e3ee24f334e3fa8f325615c3719c6)
- decay2: job 550807，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/ed135ecedf4247a1bca00f292130226c)
- proxy0: job 549715，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/517f1c6078314dd0906ce35e18201905)
- proxy1: job 549716，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/d08d9170d33449c386626dda61a789df)
- proxy2: job 549717，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/cfc76e7ef7ff4aac9900c4661b0c03fd)
- proxy3: job 549718，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/2d2bf861a8c74280897724f690d164fb)
- screen0: job 549594，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/e210a566df434a138e7e6ca3a2344b80)
- screen1: job 549595，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/9f4f3ef033524833b03e5d8edf1bd8c1)
- screen2: job 549596，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/1778f3bbc11c4bf0971cd606df669595)
- screen3: job 549597，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8aa49d8881a847e6afdfa32e61541748)
- validation: job 549160，TIMEOUT
- validation-r1: job 549507，COMPLETED；[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/09a1262b21524a59ab7aa197dcbf7ade)

# 1–6B token LR transfer

狀態：尚未通過所有驗證。固定模型與AdamW/Muon比例，不是二維全域最佳值。

| Token horizon | Muon peak LR | AdamW peak LR | 狀態 |
|---|---:|---:|---|
| 1B | 0.00315254264 | 9.45762792e-05 | unvalidated_estimate |
| 2B | 0.00197262839 | 5.91788517e-05 | unvalidated_estimate |
| 3B | 0.00149946984 | 4.49840953e-05 | unvalidated_estimate |
| 4B | 0.00123432518 | 3.70297554e-05 | unvalidated_estimate |
| 5B | 0.00106140218 | 3.18420655e-05 | unvalidated_estimate |
| 6B | 0.000938257501 | 2.8147725e-05 | unvalidated_estimate |

log LR = -0.67639668 log(tokens/1B) + -5.759546

1B附近為proxy實測範圍；2–6B為外推。WSD檢查只在400M，不宣稱已逐一實測1–6B。單seed；EMA只用於proxy量測。
fit-window range為擬合視窗敏感度，不是統計95%信賴區間。

