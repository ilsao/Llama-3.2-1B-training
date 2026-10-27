# 6B-token R5 KD — selected recipe

本目錄是 main 的最佳配方入口，選定 `kd6b-r5-20261020-seed42`（formal19）。
[manifest.json](manifest.json) 固定實作、配置、checkpoint、HF revision 與比較依據。

| 入口 | 實際用途 |
| --- | --- |
| [config.json](config.json) | 原封不動的 resolved configuration |
| [implementation/runner.py](../../experiments/kd6b-r5-20261020/runner.py) | 真正執行 6B R5 KD 的 trainer |
| [implementation/checkpointing.py](../../experiments/kd6b-r5-20261020/checkpointing.py) | 所有 optimizer owners 的完整 checkpoint |
| [implementation/contracts.py](../../experiments/kd6b-r5-20261020/contracts.py) | token-clock schedule、28,619 updates 與預算合約 |
| [runtime/](../../experiments/kd3b-hidden8-20261019/source/) | 訓練底層、NorMuon、資料、attention 與 scheduler |
| [fused_kd.py](fused_kd.py) | fused full-vocabulary CE/KL |
| [accelerations.py](accelerations.py) | local compilation、whole-matrix optimizer owners |
| [job.sbatch](job.sbatch) | 原 run 的 8 H200 / 8 CPU / 9h Slurm 腳本 |

這些入口使用 repository-relative symlinks，指向原 run 的封存檔案。
原本的 bootstrap、相對路徑、授權與雜湊保持一致；修改任一入口會修改封存來源。
要做新實驗請建立新的 recipe/run，不能覆寫既有 formal19。

## 配方

- 隨機 residual-half seed42 的 Llama 3.2 1B；context8192、intradoc causal mask。
- Frozen official Llama 3.2 3B teacher；0.75 CE + 0.25 KL，temperature1。
- NS5 NorMuon matrix LR `0.00425961126463665`；輔助 AdamW LR `0.000255576675878199`。
- Microbatch2；global batch16→64，切換於 `3,000,107,008` inputs。
- Token-clock 兩階段，phase split `5,000,003,584` inputs、second peak0.1×；
  各 phase warmup170 BS16-equivalent units，40% tail，4:1:1 ramp/hold/final、plateau0.2。
- 硬 endpoint `6,000,082,944` inputs / 28,619 optimizer updates；teacher 計數獨立。
- 實測 allocation `64.7111111111` H200-hours；適用該 run 已批准的72H例外。

固定10k PPL **12.3989218778**。完整50k尚未評估。
`config.json` 的 `max_steps=45777` 是 BS16-equivalent schedule units；實際更新數是28,619。
其中 `proposed-full` 等舊 metadata 保留原 hash，完成狀態以
[completion-status.json](../../experiments/kd6b-r5-20261020/completion-status.json) 為準。

## 查看與驗證

從 repository root 執行 `make sota`、`make check-sota`、`make test-sota`。
CLI 只讀 metadata；不提交 allocation、不載入 checkpoint。
歷史 `launch.py submit` 的一次性授權已使用，不能用它自動重跑。
模型評估／推論也須在 compute node 進行；使用
[manifest.json](manifest.json) 的固定 `model_revision`，避免下載到後來不同的模型。
