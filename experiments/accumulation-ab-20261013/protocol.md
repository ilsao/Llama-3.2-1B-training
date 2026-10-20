# 梯度累積開關小實驗

使用者 2026-10-13 指示：「用小實驗比對：開關梯度累積哪個比較好。」
本實驗是 batch geometry runtime validation，非 optimizer sweep 或新正式 run。

| 組別 | GPU | 每 rank microbatch | accumulation | global sequences/update | input tokens/update |
|---|---:|---:|---:|---:|---:|
| on | 8 H200 | 1 | 2 | 16 | 131072 |
| off | 8 H200 | 2 | 1 | 16 | 131072 |

兩組各 120 updates / 15,728,640 non-padding inputs，前20 updates暖機、21–120測速。
沿用已 GPU 驗證的 performance-ab-20261010/source-b 封存加速版、FA3、bf16、
activation checkpointing、Muon/AdamW、BE60P20 LR schedule、seed42 random initialization。
保留15259 scheduler horizon及170 warmup，不將效能試驗變成短期品質ranking。
兩組configuration只允許microbatch/accumulation不同；每次update的global sequence
membership完全相同，但sequence分配到rank的方式改變。沿用原canonical split、
dolma-raw-baseline-20261008、8192 packing/masks與固定first1000 reserved evaluation documents。
每12 updates評估、結尾評估，新的W&B identity，project lab5-training-llama。

on完成120 updates並對帳後才提交off，off指定on實際使用的同一節點。
各job單一launcher task、8 CPU total、8 H200、200G RAM、15分鐘，保留180秒stopping
margin及30秒final evaluation reserve。每組預留2 H200-hours，合計上限4 H200-hours；
暖機、compiler、初始化、評估、checkpoint與idle均計入shared ledger。
來源與配置在提交和compute node開始時驗證SHA256 seal；compiler每rank1 worker，
兩組使用各自全新的/work compiler caches，所有大型artifacts保留於/work。
保留final checkpoint，不做額外HF export、upload或cleanup。禁止checkpoint初始化、
自動resume/retry；OOM/數值錯誤/未完成120updates/不確定提交均停止，保留成本及reservation。

量測主要是sum input tokens / sum max-rank完整update wall time，包含data/counting、
token journal、forward/backward、optimizer及logging，排除evaluation/checkpoint。
另報CUDA-synchronized compute throughput、median/p95 update duration與每rank顯存峰值。
顯存峰值包含初始化及評估；observer在forward後記錄graph count，須在20–120update
間穩定，避免把持續recompile當作穩態。cold-start編譯成本在allocation記帳並另報估計。

比較前核對random weight signature、code/environment identity、split/eval/packing
provenance、optimizer parameter coverage，以及每update input/prediction tokens與LR。
數值接受條件：final held-out PPL相對差<=1%、first-update loss相對差<=0.5%、
first-update pre-clipping grad-norm相對差<=3%。這些是scalar一致性，非full-gradient等價。
只有數值檢查通過且穩態吞吐差至少5%才推薦较快組；差<5%標記inconclusive。
單次短跑不證明最終訓練品質或batch saturation，不能推論GPU utilization已100%。

```bash
python3 slurm/accumulation_ab.py check
python3 slurm/accumulation_ab.py on
python3 slurm/accumulation_ab.py status
# 僅在on完整完成且對帳後：
python3 slurm/accumulation_ab.py off
python3 slurm/accumulation_ab.py status
python3 slurm/accumulation_ab.py report
```

## 使用者要求停止監控（2026-10-13）

依「你可以不用監控他訓練，等我叫你看再看」，停止assistant polling。
off改為afterok:on，指定on已確認的25a-hgpn117節點。
compute node開始off訓練前核對on完整120updates/token endpoint並對帳，
不完整或失敗則停止。off使用slurm/accumulation-off-queued.sbatch，
另記錄其SHA256，原execution seal及已執行的on配置不修改。
報告和off最終對帳等使用者下次要求查看時執行。

## 使用者授權修正與並行重跑

新實驗：`accumulation-ab-20261013-rerun1`；使用者要求修好後並行重跑。
原536902失敗、536909取消已對帳，保留所有原artifacts及成本；詳見新實驗protocol.md。
