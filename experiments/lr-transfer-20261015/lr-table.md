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
