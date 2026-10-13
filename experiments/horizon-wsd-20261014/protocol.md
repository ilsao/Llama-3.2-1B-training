# Token horizon scaling, clipping and WSD checkpoint branches

Status: CONFIRMATORY protocol, locked before compute. Campaign `horizon-wsd-20261014`.
User requested all four stages through autoresearch, clarified sweep **0.7B** and final **2B** on 2026-10-14. Each candidate gets its own independent sbatch; sibling jobs have no Slurm dependencies or node affinity. Stages advance only after completed, reconciled and validated endpoints. No automatic retry, unexpected continuation, cleanup, or HF upload.

## Fixed comparison

Reference: `cg-20261010-be60p20-s42`, 399,900,672 non-padding inputs, Muon peak 0.02 / AdamW peak 0.0003; endpoint subset PPL 27.4331322294. Architecture Llama 3.2 1B, random initialization, seed42, 8192 context, microbatch1, accumulation2, eight H200/eight CPU total, global batch16, 131,072 inputs/update. Native Muon settings, AdamW betas, both WD=0.1 and 170 warmup updates remain fixed. Keep BE60P20's `wsd_hold_zero` family, plateau 0.2 and cooldown first-ramp:hold:last-ramp ratio4:1:1. Changing cooldown length scales all three tail lengths proportionally. No cosine schedule is introduced.

Use existing raw preparation `dolma-raw-baseline-20261008`, dataset revision `afa92bfb22366821c5e6cd427cdd036b34b713ef`, complete canonical 104,020,922 documents shuffled by native HF seed42; last50,000 evaluation originals are excluded. All candidates evaluate the identical first1,000 of the fixed10,000 reserved subset, with exact token-weighted NLL. No data cleaning/filtering/preparation changes. The canonical packed prefix already covers the full final horizon; substantial checksum and mask checks run on allocated nodes.

Tuning horizon: 5,341 updates /700,055,552 inputs (first complete batch crossing700M). Final horizon:15,259 updates /2,000,027,648 inputs (first complete batch crossing2B). Token counters include every forward pass, including skipped updates; skipped/nonfinite updates disqualify endpoints. Report LR actually used, synchronized pre-clipping norm, tokens/s excluding evaluation, cumulative actual non-padding inputs, and optimizer-group LRs to `lab5-training-llama`.

## Hypotheses and sequence

H1: Longer horizons favor a lower peak LR. Use arXiv:2409.19913v3, Eq1/4: `eta_pred(D)=eta_ref*(D/399900672)^(-0.32)` for both optimizer peaks. This is a transferred heuristic: the cited fit used AdamW/cosine and horizons much larger than0.7–2B; its exponent is not established for Muon or this WSD shape. Test multipliers0.5,1,2 rather than claim a theoretical WSD optimum. Same seed, token horizon, batch and complete cooldown for all three fresh random-init candidates.

H2: Clipping may suppress rare unstable updates but excessive clipping may impede learning. With H1's endpoint winner, train four independent fresh random-init candidates at global clipping0.5,1,2,off. `max_grad_norm=0` means off; report a finite pre-clipping global norm without modifying gradients. The replicated DDP norm counts tied/unique parameters once, not once per GPU.

H3: Cooldown timing trades stable-phase progress against noise reduction. Train one fresh stable-prefix parent with H1/H2 winners. Pause after2,136 updates /279,969,792 inputs, approximately40% of the0.7B horizon, before every candidate's first LR decrease. Each of five independent sbatches loads this same complete model/optimizer/scheduler/RNG checkpoint and data cursor. Tail ratios `[0.6,0.5,0.4,0.2,0.1]` mean decay starts with that fraction **remaining**, at approximately completed40/50/60/80/90%. Only schedule shape and branch identity/budget metadata change. Saved schedule state is explicitly rebased to each future curve; all pre-fork LR histories must be identical. New W&B identity per branch, optimizer step begins at2,136, inherited tokens are included in each branch's logical budget. Physical shared prefix is charged once in the aggregate ledger; its allocation hours count toward each branch's logical64-hour cap.

Stage winners use strict argmin of **final endpoint NLL**, ties resolved in listed order. Require all listed candidates to finish successfully before promotion; a failed stage stops new submissions for diagnosis. No early-time training loss or historical/best checkpoint selection. Single seed establishes ranking here, not statistical significance or proven scaling exponents.

H4: Reheating and shortening the final training horizon may improve adaptation on unseen tail data. After all sweeps, train two fresh2B logical runs using H1/H2/H3 winners, two sbatches each:

1. `final-standard`: one2B-scaled peak pair and one continuous WSD schedule across both jobs. Pause after12,715 updates; second job resumes the exact same schedule.
2. `final-resplit`: reserve the final2,544 complete batches of the **2B selected training budget**, without using them in the first phase. First phase12,715 updates /1,666,580,480 inputs; second phase2,544 updates /333,447,168 inputs. This is a disjoint contiguous partition of packed sequences in the chosen budget, approximately5/6 and1/6; it is not a partition of all150B corpus documents. A long document may span this packed-sequence boundary; no tokens or sequences repeat and reserved evaluation membership never changes. Both jobs retain the same logical run, optimizer momentum, RNG, cumulative cursor/tokens/compute and W&B identity. Phase1 peak uses `D1`; phase2 peak uses `D2`, with the winning multiplier. At the boundary re-warm for170 local updates and replay the winning WSD tail shape. No optimizer reset or data shuffle. The full phase schedule is immutable from initialization; standard resumes require identical configuration.

Both arms export the complete15,259-update endpoint, verify HF reload on compute, and evaluate all50,000 reserved originals during the second sbatch. Compare full-set NLL/PPL; tuning subset NLL remains separate. These are formal runs five and six, requested by the current user message; preserve all four previous formal runs and costs. Scope the formal-run/180-minute allocation exception to these two named campaign runs only; no unrelated exception.

## Allocations and stopping

| Work | sbatches | GPUs/CPU each | minutes each | aggregate H200-hour cap |
|---|---:|---|---|---:|
| Distributed clipping/checkpoint +20-update fresh-model validation |1|8/8|10|1.333333|
| LR candidates |3|8/8|90|36|
| Clipping candidates |4|8/8|90|48|
| Stable parent |1|8/8|45|6|
| Decay branches |5|8/8|75|50|
| Final standard phase1/phase2 |2|8/8|180/60|32|
| Final resplit phase1/phase2 |2|8/8|180/60|32|
| Total |18|per job8/8| |205.333333|

All overhead, evaluation, checkpointing and idle allocation time count. Sweep branches cap16 logical H200-hours including their parent. Each final logical run caps32 H200-hours including both jobs, below64; max2,000,027,648 inputs below6B. Stop margin180sec, tuning final evaluation reserve120sec, final full evaluation/export reserve600sec. Actual sacct reconciliation is mandatory before dependent submission. Runtime failures/incomplete endpoints do not authorize extra jobs. At most five sibling jobs may run simultaneously (40H200/40CPU); per-job limit remains8/8. No total historic-budget reset; outstanding old reservations remain charged. The planned total is a conservative allocation maximum, not actual cost or a remaining-budget claim.

## Continuity and controls

`slurm/horizon_campaign.py` is the sole submission owner. `run-once` performs one bounded stage/accounting cycle. A dedicated20-minute cron would call that controller, preserving all other schedules. Legacy heartbeat does not submit this campaign. Installation was rejected by automatic review for persistent-schedule authorization; keep the schedule uninstalled pending explicit user approval. The skill's mandatory continuity instruction is not treated as permission to bypass the rejection.

```bash
python3 slurm/horizon_campaign.py check
python3 slurm/horizon_campaign.py test-only
python3 slurm/horizon_campaign.py run-once
python3 slurm/horizon_campaign.py status
python3 slurm/horizon_campaign.py install  # persistent schedule; awaiting review approval
python3 slurm/horizon_campaign.py disable  # stop new submissions; retain accounting
```

Source and immutable metadata/configuration SHA256 seals are in plan.json. Protocol commit precedes compute. No heavy task runs on login. Model/tensor/runtime verification occurs in the10-minute validation allocation before tuning. Pure contract tests cover horizons, disabling clipping, shared-prefix LR equivalence, phase-boundary schedule restart, scoped formal limits, and inherited token journals. Final checkpoint loads are restricted to authorized branches and the two planned final continuations.
