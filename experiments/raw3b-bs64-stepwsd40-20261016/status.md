# Actual-step40% phase tails:3B training

Slurm558510 RUNNING on25a-hgpn069,8H200/8CPUs,359minute allocation,formal13. Config digest a6084bbce35936772098e364b2c1bab5affcfe5ae3b979a77af2be32c81daf90;source commit5f7e44d.14 pure tests passed,43Python files parsed and sbatch shell syntax passed. Random initialization matches prior signature,no pretrained/checkpoint load,finite updates,W&B online.

Both warmup and decay use ACTUAL optimizer updates. Warmup170updates per phase,as explicitly selected by the user. Phase1 has13355updates,decay after8014;phase2 has957updates,decay after574 within that phase(global13929). Keep40% tail4:1:1 ramps/20%-LR plateau. First decay corresponds to1050411008 inputs. This preserves the5/6 TOKEN phase split,not a5/6 split of actual updates.

Batch/data windows and table LRs match formal12:BS16->64 at1499987968 inputs,beta2=.95,Muon peak.005593716406714465,AdamW peak.00016781149220143396. Stop at3000107008 inputs/14312actual updates. Second-phase warmup is now170actual updates versus token-clock warmup in formal12;record this difference in result comparisons.

Expected complete LR table:/work/asciibase64/Llama-3.2-1B-training/runs/raw3b-bs64-stepwsd40-20261016/expected-step-lr.csv. Startup actual LR checks:startup-evidence.json. Completion requires a passing actual-log schedule-audit.json,final export reload and full50000-document evaluation;no final PPL yet.

[W&B](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/cba7b737a27141ba84ab7ec861b6fe73). Live controller status:report.md/controller-state.json. No automatic retry/resume/requeue,cleanup or HF upload;all prior costs remain recorded.
