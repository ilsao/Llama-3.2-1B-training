# 梯度累積開關修正後並行重跑

使用者明確授權：「修好他們，然後並行的重跑。」
本次是原 accumulation A/B 小實驗的兩個 fresh random-initialization validation runs，
各自使用新 run ID、新 W&B identity、新 compiler cache。原始配置、source seal、失敗 artifacts
及成本保留；historical-allocations.json 記錄已對帳的 536902 / 536909。
歷史實際成本為 8 × 376 / 3600 = 0.835555556 H200-hours，不移除或重新分類。

| 組別 | 每 job GPU / CPU | microbatch / accumulation | updates | non-padding inputs |
|---|---|---|---:|---:|
| on | 8 H200 / 8 CPU | 1 / 2 | 120 | 15,728,640 |
| off | 8 H200 / 8 CPU | 2 / 1 | 120 | 15,728,640 |

兩個 sbatch 無 afterok 依賴、無同一節點綁定；Slurm 分別排程，允許同時執行。
總同時資源最多16 H200 /16 CPU；每 job符合8 H200 /8 CPU限制。
使用已驗證 ACD115198 /8gpus、H200 feature constraint、單一node/launcher，
每個job15分鐘/最多2 H200-hours，新增總上限4 H200-hours；含歷史總上限4.835555556。
180秒 stopping margin及30秒final evaluation reserve保留，不自動retry/resume。

修正 overlay 使用 grad-enabled 且 module.training 辨識訓練 forward，
允許評估傳入 eager forward callable，不讀取 function.training，不將eval計入observer。
只將首個compiled update的耗時移出後續step的rolling maximum，
初始下步估計最多30秒，第二步起採recent20步的最大值；若後續重編譯仍計入估計。
首次編譯耗時留在原始timing與allocation成本，observer記錄cold_start_update_seconds。
硬allocation上限、訊號、final overhead reserve與token cap不變。

沿用 performance-ab-20261010/source-b封存training code，FA3/bf16/activation checkpointing、
Muon/AdamW、seed42、8192 context、canonical raw split/packing/masks、固定1000 held-out documents。
兩組global batch16、每update131072 inputs，配置只允許microbatch/accumulation差別；
scheduler horizon15259、warmup170、每12updates與結尾evaluation。
暖機1–20排除於穩態吞吐，21–120測量，檢查graph穩定、初始化、資料/optimizer/token/LR一致性。
沿用原數值與5%吞吐門檻。節點可不同，report需列allocated_nodes；單次跨node結果可能含硬體差異，
不作最終品質結論。保留final checkpoint；不做export、upload或cleanup。

本次5項純Python stub回歸測試及bash語法檢查通過；GPU/runtime檢查於這兩個sbatch內執行。
依使用者先前「等我叫你看再看」要求，提交後停止持續監控；不建立自動retry或監控排程。

```bash
python3 slurm/accumulation_parallel.py check
python3 slurm/accumulation_parallel.py on
python3 slurm/accumulation_parallel.py off
# 使用者下次要求查看時：
python3 slurm/accumulation_parallel.py status
python3 slurm/accumulation_parallel.py report
```
