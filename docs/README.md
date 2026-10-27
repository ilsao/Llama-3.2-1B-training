# 專案文件

| 文件 | 用途 |
| --- | --- |
| [目前最佳配方](../recipes/sota6b/README.md) | 正式 6B R5 KD 入口與數值配方 |
| [6B 比較](../results/README.md) | 選擇依據及原始 metadata |
| [研究發現](research/findings.md) | 既有實驗結論，保留原文 |
| [研究日誌](research/research-log.md) | 既有時間序列紀錄，保留原文 |
| [歷史 baseline 操作手冊](legacy/baseline.md) | 原 README 的完整內容，包含早期資料、AdamW、sweep 與控制器操作 |

## 目錄原則

`recipes/` 放目前選定配方的入口；`results/` 放小型結果及來源索引；
`experiments/` 保存原始實驗、成本、授權、Git worktrees 與 sealed source。
`slurm/` 保留叢集腳本；`requirements/` 集中環境清單。

歷史實驗路徑參與 hash、bootstrap、Slurm source-root 與 preparation fingerprint，
因此保留原址。recipe 使用相對 symlinks，避免複製出第二份會漂移的實作。
根目錄的 baseline Python API 仍供舊 controllers / tests 使用。
舊 `findings.md`、`research-log.md`、`requirements-*.txt` 以 symlink 保持相容。
所有文件中的 shell 指令預設從 repository root 執行。

`research-state.yaml`、controller state、現有 Report/ 與 worktrees 不由本次整理接管。
Report/ 是工作區既有的未追蹤稿件，結果索引直接引用原始 run metadata。
模型權重、optimizer states、dataset、cache 與大型 logs 留在 `/work/asciibase64/`。
歷史單次訓練授權不是可重複使用的執行許可；新訓練沿用 AGENTS.md 的授權流程。
