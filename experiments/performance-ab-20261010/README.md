# 原版與整合加速版

已接入 torch.compile、native Muon Newton–Schulz 核心編譯、FA3 dense path、可重用 pinned buffers 與 copy stream。訓練的參數、資料順序、mask、8192 context、batch 與 BE60P20 schedule 保持。A 使用父 run 的封存原版，B 使用 source-b 封存來源；共用只讀量測觀察器記錄 compile metrics、每 rank 顯存峰值與 graph 數。

## 驗證狀態

38 項純函數／配置／budget／submission／report 契約檢查通過，Python AST、sbatch syntax 與 git diff --check 通過。首次 GPU correctness 提交遇到 Slurm outage，回報 socket timeout，未取得 job ID。2026-10-10 21:52 Asia/Taipei，Slurm 恢復，查核佇列、accounting 與啟動紀錄沒有原 job 後，依使用者「現在應該好了，重新試試」明確指示，以新 run ID 提交一次驗證重試：job 524270，runtime-recovery-1。原 submission 與 4 H200-hours reservation 保留；新驗證已 COMPLETED 並通過，耗時474秒／1.053333 H200-hours。模式 max-autotune；Muon core 相對誤差1.782%，8192 context 梯度誤差0.221%；checkpoint/RNG replay、eager evaluation、HF reload 通過。A/B 已提交：A job 524312，B job 524313，B afterok:A，兩組已 COMPLETED/reconciled，HF exports reload verified。量測21–120 updates：A 197,109 tokens/s，B 262,408 tokens/s，B 加速1.331×；PPL A 912.852739，B 912.201729，相對差0.0713%。這是各120 updates／15,728,640 tokens的短跑，仍在170-update warmup內，不是既有2B模型的PPL。測量區間8 ranks graph count固定，沒有新增recompile。完整結果見 results.md / results.json。

## 恢復與比較

先查核唯一的首次提交：

```bash
python3 slurm/performance_ab.py status
```

只有 active runtime 正常 COMPLETED、correctness passed 且 actual allocation 對帳完成後：

```bash
python3 slurm/performance_ab.py benchmarks
```

A 先執行，B 使用 afterok:A 依序執行；source seal 在提交前核對。完成後：

```bash
python3 slurm/performance_ab.py status
python3 slurm/performance_report.py
```

報告寫入 results.json 與 results.md。主要比較 21–120 updates 的完整 update 吞吐量；compiler 類別計時互有巢狀，不加總。cold-start excess 與 2B projection 明確標示估計。PPL relative 差 <=1% 且端到端加速 >=5% 才推薦 B。

## 預算與 artifacts

Runtime 30 分鐘／4 H200-hours，A/B 各45分鐘／6 H200-hours，初始 allocation cap16 H200-hours。明確授權的重試新增最多4 H200-hours，舊不確定 reservation 仍保留，實際耗用待 accounting 確認。各 job8 H200、8 CPUs，compiler 每 rank1 worker。所有 actual allocation 保留於共享 ledger，未確定 submission 持續保留 reservation。

配置 max_steps/scheduler_steps 保留15259，warmup170，用15,728,640 non-padding input token target停止於120 updates。每12 updates與結尾評估同一1000 reserved documents。

大型 caches、checkpoints、exports、W&B 與 observer logs 在 /work/asciibase64/Llama-3.2-1B-training/。此實作未啟動新2B正式run、checkpoint continuation 或 HF上傳。

完成時已知實際 validation 成本：runtime 1.053333 + A 0.475556 + B 1.137778 = 2.666667 H200-hours；首次 timeout 的4 H200-hours reservation仍保留，不能當作已確認零耗用。B估計冷啟動額外326秒；A整個allocation214秒，B512秒。以median update差估計約2114 updates（約2.77億tokens）攤平編譯成本。2B投影為估計，未啟動新的正式訓練。
