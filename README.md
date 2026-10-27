# Llama 3.2 1B — 6B-token SOTA

目前 main 的最佳已完成配方是 **R5 online KD（formal 19）**：隨機初始化的
Llama 3.2 **1B 參數** student，訓練 **6,000,082,944 個非 padding input tokens**。
這裡的「6B」指訓練 tokens。正式入口集中在 [recipes/sota6b/](recipes/sota6b/README.md)。

| 6B 配方 | 固定 1k PPL | 合併 10k PPL | Updates |
| --- | ---: | ---: | ---: |
| 非 KD / NorMuon + AdamW ×2 | 12.7739 | 12.9078 | 28,619 |
| **R5 online KD（選定）** | **12.2678** | **12.3989** | **28,619** |
| C2 adaptive batch + early decay | 12.3953 | 12.5380 | 22,060 |

來源：[可驗證比較與原始 metadata](results/6b-comparison.json)。三組為相同 seed42、
6B endpoint 與固定評估範圍；10k PPL 由合併 NLL / prediction tokens 計算。
這些 checkpoint 尚未完成全保留 50k 評估；C2 同時更動 batch 與 decay。

## 從這裡開始

```bash
make sota          # 顯示最佳配方、checkpoint 與固定 HF revision
make check-sota    # 唯讀驗證來源 hash、配置、結果與正式入口
make test-sota     # 純 Python 配方／索引測試，不載入模型
```

以上可在 login node 執行。模型載入、訓練與評估須透過 Slurm。
R5 的歷史單次 allocation 已完成；新的訓練須另建 run、配置與預算授權。

已發布模型（依本地完成紀錄固定版本）：
[asciibase64/Llama-3.2-1B-dolma3_mix @ 3f9196e](https://huggingface.co/asciibase64/Llama-3.2-1B-dolma3_mix/tree/3f9196e11f990ed2cb60fcf558d5f1f07a7a36d6)。
[W&B 訓練曲線](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/58970e70c88c)。

## 專案導覽

| 位置 | 內容 |
| --- | --- |
| [recipes/sota6b/](recipes/sota6b/README.md) | 選定配置、訓練實作、runtime、KD kernels 與 Slurm 入口 |
| [results/](results/README.md) | 6B 比較、選擇依據與小型原始結果 metadata |
| [experiments/](experiments/README.md) | 歷史實驗、授權、成本、來源封存；依主題索引 |
| [slurm/](slurm/README.md) | 叢集執行腳本與控制器 |
| [configs/](configs/README.md) | 早期 baseline / sweep 配置；SOTA 配置由 recipe 指向封存版本 |
| [requirements/](requirements/README.md) | 資料、訓練與 FA3 環境的 pinned dependencies |
| [docs/](docs/README.md) | 結構說明、研究紀錄與歷史操作手冊 |
| tests/ | 輕量測試與需 Slurm 的 runtime checks |
| literature/、to_human/ | 研究參考資料與既有交付資料 |

根目錄 `train.py`、`dataset.py` 及其 helper modules 是早期 baseline 的相容入口。
SOTA 執行的是 `recipes/sota6b/implementation/runner.py` 與封存 runtime；不能只把
SOTA config 傳給根目錄 `train.py` 就視為相同配方。
大型資料、checkpoint、cache 皆位於 `/work/asciibase64/Llama-3.2-1B-training/`。
`AGENTS.md` 保留全部實驗授權；`research-state.yaml` 保留既有控制器狀態。
