# 實驗索引

目前最佳配方：[6B R5 KD](../recipes/sota6b/README.md)。比較依據：[6B results](../results/README.md)。

這裡保存原始 identities、授權、成本及封存程式。`source*`、`launch-source`、
`worktrees` 可能是 sealed copies 或實際 Git worktrees，保留原址與雜湊。
各實驗是否完成以 final summary / completion / accounting 為準，早期 startup-status 可能停留在啟動時。

## 6B 最終比較

- [kd6b-c2-earlydecay-20261020](kd6b-c2-earlydecay-20261020/)
- [kd6b-r5-20261020](kd6b-r5-20261020/)
- [raw6b-normuon-adam2x-20261018](raw6b-normuon-adam2x-20261018/)

## KD / acceleration / methods

- [kd3b-hidden8-20261019](kd3b-hidden8-20261019/)
- [kd3b-io8-validation-20261019](kd3b-io8-validation-20261019/)
- [kd3b-online-pipeline-20261019](kd3b-online-pipeline-20261019/)
- [kd3b-tokens-20261019](kd3b-tokens-20261019/)
- [kd6b-lossless-20261020](kd6b-lossless-20261020/)
- [kd6b-lossless2-20261020](kd6b-lossless2-20261020/)
- [methods-parallel-20261019](methods-parallel-20261019/)
- [methods-research-20261019](methods-research-20261019/)
- [methods-validation-20261019](methods-validation-20261019/)

## 初始化、LR、batch、scheduler

- [batch-normuon-1b-20261016](batch-normuon-1b-20261016/)
- [be60p20-lrscaled-2b-20261010](be60p20-lrscaled-2b-20261010/)
- [cooldown-grid-20261010](cooldown-grid-20261010/)
- [critical-batch-20261020](critical-batch-20261020/)
- [critical-batch-20261020-recovery1](critical-batch-20261020-recovery1/)
- [critical-batch-20261020-recovery2](critical-batch-20261020-recovery2/)
- [horizon-wsd-20261014](horizon-wsd-20261014/)
- [init-recipe-20261014](init-recipe-20261014/)
- [init-recipe-20261014-followup](init-recipe-20261014-followup/)
- [lr-transfer-20261015](lr-transfer-20261015/)
- [lr-transfer-horizons-20261015](lr-transfer-horizons-20261015/)
- [lr-transfer-wsd-bracket-20261015](lr-transfer-wsd-bracket-20261015/)
- [optimizer-sweep](optimizer-sweep/)
- [ppl-next-20261016](ppl-next-20261016/)
- [ppl-next-20261017](ppl-next-20261017/)
- [raw3b-bs64-stepwsd40-20261016](raw3b-bs64-stepwsd40-20261016/)
- [raw3b-bs64-wsd40-20261016](raw3b-bs64-wsd40-20261016/)
- [raw3b-sota-20261015](raw3b-sota-20261015/)
- [raw3b-sota-wsd20-20261015](raw3b-sota-wsd20-20261015/)
- [raw4b-intradoc-normuon-20261017](raw4b-intradoc-normuon-20261017/)
- [raw4b-normuon-adam2x-20261017](raw4b-normuon-adam2x-20261017/)
- [raw4b-normuon-adam2x-step30-20261018](raw4b-normuon-adam2x-step30-20261018/)

## 資料、baseline 與其他比較

- [accumulation-ab-20261013](accumulation-ab-20261013/)
- [accumulation-ab-20261013-rerun1](accumulation-ab-20261013-rerun1/)
- [adamw-baseline](adamw-baseline/)
- [baselines](baselines/)
- [compressed-corpus-audit](compressed-corpus-audit/)
- [data-equivalence](data-equivalence/)
- [filtering-recovery](filtering-recovery/)
- [gpu-scaling](gpu-scaling/)
- [lossless-schema-loader](lossless-schema-loader/)
- [native-loader-recovery](native-loader-recovery/)
- [performance-ab-20261010](performance-ab-20261010/)
- [raw-buffer-6b-20261015](raw-buffer-6b-20261015/)
- [raw-filtered-3b-20261015](raw-filtered-3b-20261015/)
- [training-diagnosis-20261010](training-diagnosis-20261010/)

## 模型發布紀錄

- [hf-be60p20-2b-20261010](hf-be60p20-2b-20261010/)
- [hf-kd6b-r5-20261020](hf-kd6b-r5-20261020/)
- [hf-raw3b-sota-20261015](hf-raw3b-sota-20261015/)
- [hf-raw4b-20261017](hf-raw4b-20261017/)
- [hf-raw6b-20261019](hf-raw6b-20261019/)
