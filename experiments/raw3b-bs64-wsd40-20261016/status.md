# Fresh3B BS64 /40% phase tails

Slurm556710 COMPLETED on25a-hgpn058 with8H200/8CPUs/200GB and359minute limit. Formal12,one allocation only,48H200-hour hard logical cap. User approved the plan with Implement the plan;configuration digest a1f147bde17a4d3ff050873568d40b2482263675e91338a68f8a550979e07da7. Code commit e7bc8ca.

12 pure tests passed;all42 Python files parse and sbatch shell syntax passes. Native update math,parameter routing,random initializer and data reader match the validated1B source. Actual allocated-node startup passed:random residual-half initialization matches prior signature32b43fc6e5c34a787fc02db6674dde1afc341123e39d48abb60b5d84e932a05d,no pretrained weights loaded,finite optimizer updates,W&B online. Timestamped progress:startup-evidence.json.

Table peaks:Muon.005593716406714465,AdamW.00016781149220143396. At1499987968 inputs,BS16 to64 with beta2=.95 preserved and first decay begins. Each5/6+1/6 phase has40% tail and170Q warmup;second peak.1 times first. No optimizer reset or LR batch scaling. Exact endpoint3000107008 inputs/14312 actual updates. Full final50000-document evaluation and local export reload verification stay inside the same allocation.

[W&B run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/6ae9e9ddd2ad4dffb3c8fc3a5688143d). Live controller state and terminal accounting:report.md/controller-state.json. Final full50000-document PPL15.21170256746107;subset PPL15.213240094314846. Actual inputs3000107008,optimizer updates14312,allocation14376s/31.9466667H200-hours. Local final export reload verified. No automatic retry/resume/requeue,cleanup or HF upload. The minute controller removes only its own cron after reconciliation;all historical budgets remain preserved.

Decay audit:all1432 logged updates match expected actual LR,tokens,batch and fixed beta2 without mismatch. First tail starts at49.9978% inputs but79.9609% optimizer updates because BS quadruples at that point. Each40% phase tail includes a20%-LR plateau;the tail is not one continuous ramp. Actual curves on both axes:audit/compare-axes.svg. Machine-readable audit:audit/schedule-audit.json. No training rerun or scheduler change was made.
