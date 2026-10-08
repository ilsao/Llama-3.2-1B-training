# Native Muon optimizer sweep

The sweep variable is `optimizer: adamw | muon`. Muon uses **PyTorch's
`torch.optim.Muon`**, with `torch.optim.AdamW` for embeddings, the output head,
normalization parameters and biases. There is no local optimizer algorithm.
Parameter identity deduplication assigns tied embeddings/output weights once.
Unknown non-Linear matrices remain on AdamW. Complete parameter names, aliases,
element counts and native optimizer options are saved in `optimizer-groups.json`
and W&B. `train/learning_rate` is the auxiliary `adamw_no_decay` group's rate
for Muon; all group rates are logged separately.

Both optimizers receive the same warmup/cosine multiplier and one step per
successful global update. Accelerate prepares both optimizers and saves/loads
both native state dictionaries and both registered schedulers. RNG, canonical
data cursor, consumed shard prefix, W&B identity and cumulative budgets continue
through the existing checkpoint protocol. DDP replicates native optimizer state;
measure overhead before drawing compute-efficiency conclusions.

The implementation is isolated in `.worktrees/muon-sweep`, branch `sweep/muon`,
so the running baseline's source and PyTorch 2.6.0 environment stay frozen.
Native Muon is documented in PyTorch 2.9; the new environment pins **2.9.1**
with the official CUDA 12.6 wheel and rebuilds the same pinned FA3 source against
that version. It uses a separate environment/source/wheel cache under `/work/`.
An AdamW control must also use this new environment to compare optimizers fairly.
The new dependency setup is separately charged to the shared ledger, including
its one H200 allocation. Setup performs no model or CUDA-kernel runtime checks.

`configs/optimizer-sweep.json` records three prepared candidates: AdamW LR 3e-4,
Muon LR 0.02 and 0.04 with auxiliary AdamW LR 3e-4. These are candidates, not
claimed optima. The purpose is optimizer LR/numerical-stability screening, so
allocations are 15 minutes each (ultrashort), at 8 H200/8 CPU, global batch 16,
8192 context and the same raw canonical split/fixed 1,000-document subset.
Promote using held-out NLL at a common measured token checkpoint, finite global
gradients and measured costs. A subsequent optimizer × scheduler × data sweep
would start at medium (2 hours/candidate), only with recorded remaining budget;
it is not automatically scheduled.

The baseline-fa3-20261009 combined **16 H200-hour AND 7,200 allocation-second**
cap remains unchanged. At preparation time, measured validation plus committed
baseline allocation was 15.35 H200-hours / 7,027 seconds. This cannot fund a
full 15-minute sweep candidate. **No runtime validation or sweep has been
submitted.** A new run ID does not create new baseline budget. The existing
baseline controller remains its sole submission owner. Dependency installation
is permitted separately, under the shared ledger rules.

From this worktree, setup and future checks use the existing ledger-aware wrapper:

```bash
python3 slurm/submit_training.py setup muon-torch291-fa3-setup --minutes 60
TRAIN_ENV_PREFIX=/work/asciibase64/Llama-3.2-1B-training/environments/training-muon-torch291-fa3-94e22c9 \
  python3 slurm/submit_training.py runtime muon-runtime-8 --optimizer muon --gpus 8 --minutes 5
```

The second command is prepared, **unrun**, and must fit the recorded budget before
submission. It tests tied parameter coverage, real token-weighted DDP gradients,
uneven held-out evaluation, and an exact second-update replay after restoration
of both optimizers/schedulers/RNG. A later full-size 8192-token Muon validation
must also succeed before significant sweep compute is committed. All allocations,
including failed checks and continuations, require Slurm reconciliation.

References: [PyTorch 2.9 Muon API](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html),
[official 2.9.1 CUDA install command](https://pytorch.org/get-started/previous-versions/#v291).

Dependency job **516510** was submitted and was PENDING at recording time.
The existing baseline controller adopted it for terminal Slurm reconciliation
only. Its separately charged cost does not change the baseline campaign
reservation (15.35 H200-hours / 7,027 allocation-seconds). Installation completion
and FA3 runtime compatibility are not yet verified. Lightweight configuration,
budget and submission checks passed (30 tests); model checks remain unrun.
