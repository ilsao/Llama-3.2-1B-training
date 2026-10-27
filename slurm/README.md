# Slurm 入口

| 任務 | 腳本／控制器 |
| --- | --- |
| 選定6B R5 KD | [kd6b-r5-20261020.sbatch](kd6b-r5-20261020.sbatch)、[sealed launcher](../experiments/kd6b-r5-20261020/launch.py) |
| 6B non-KD | [raw6b-normuon-adam2x-recovery2.sbatch](raw6b-normuon-adam2x-recovery2.sbatch) |
| 6B C2 early decay | [kd6b-c2-earlydecay-20261020.sbatch](kd6b-c2-earlydecay-20261020.sbatch) |
| 基礎資料與 baseline | [歷史操作手冊](../docs/legacy/baseline.md) |

腳本保留歷史名稱與路徑，因為 source seals 與 submission records 直接引用。
上述正式 run 的一次性 allocation 均已使用；本次整理沒有提交新 jobs。
新訓練須有新的 identity、配置與授權，且遵守8H200/8CPU的每-job限制。
Login node 僅執行編輯、metadata 查詢與輕量檢查；模型載入、評估、訓練與安裝須經 sbatch。
