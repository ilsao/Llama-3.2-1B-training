# Latest correction: filtered cancelled

User explicitly cancelled the filtered sbatch before any filtered or mixprep
job was submitted. Only raw formal9 remains active. Filtered formal10 is revoked
in the exact authorization contract, and both filtered submission paths return
without Slurm RPCs. Raw append-only preprocessing552535 continues, then the
controller starts the unchanged random-init3B raw training. The user also
forbade repeats in filtered data; the earlier cycling proposal below is rejected,
historical design only, and must not execute.

# Matched3B raw versus old filtered with restored token proportions

The user explicitly requested two3B training runs using the extrapolated LR: original raw; old filtered restored to raw token mixture. These are fresh formal runs9/10, preserving the previous eight formal runs, all failed allocations and the restored historical ledger. Exact authorization is restricted by configuration digest in pair3b_contracts.py. Each uses one6-hour allocation with8H200/8CPUs,48H200-hours including initialization, training, final checkpoint, verified HF-format local export and full reserved-set evaluation. No automatic retry/resume, checkpoint initialization, cleanup or HF upload. The earlier aggregate256H gate remains lifted. Each run stays below6B tokens/64H200-hours.

## Fixed training recipe

Muon peakLR0.005593716406714465; AdamW peakLR0.00016781149220143396. These are the frozen3B extrapolation from true-WSD0.2/0.4/0.6B calibration, not yet a validated3B optimum. Residual-half initialization,seed42,native Muon and companion AdamW,original parameter groups/WD/clipping,8192 context,bf16 FA3,8 ranks,microbatch1,accumulation2,global16 sequences/131072 nonpadding inputs per update.170 warmup updates. Single original WSD:40%first decay,10%hold at20%peak,10%final decay to zero; no resplit,EMA OFF.22889updates/3,000,107,008 actual nonpadding inputs, the first complete batch crossing3B. Both numerical configs are identical.

Evaluate the same first1000 canonical reserved documents at least every2288steps and at the end; full final evaluation uses all50000 original documents/76,930,208 prediction tokens. Export the FINAL22889-update weights, not a midtraining checkpoint. Required W&B metrics, exact processed token journal, global gradient norm, optimizer-group LR and update timing retain their already-validated semantics.

## Data prerequisites and parallel scheduling

The current raw buffer contains about2.821B tokenized inputs, below the requested3B. First extend its append-only stream faithfully from its preparation cursor, with the existing bounded reader/tokenization overlay. Do not rewrite preparation identity, existing shards, split, permutation or evaluation assets. New source hashes and authorization seal the external execution overlay. One1H200/8CPU preprocessing allocation, at most4hours, stops after enough complete packed rows for the3B request. All allocations are charged as preprocessing.

When the raw extension finishes and published rows cover the target, launch raw training and reweighted-filtered preprocessing independently. Reweighting uses a separate1H200/8CPU,maximum16hour allocation, seven bounded workers and coordinator, no GPU algorithm. The filtered formal run starts automatically only after its verified preparation succeeds. Neither training waits for the other's checkpoint. Controller is sole submission owner; uncertain outcomes retain reservations and are recovered, never resubmitted.

## Restore actual source TOKEN shares

Reference is the source-token histogram of the FIRST3,000,107,008 packed raw nonpadding inputs which the raw arm actually processes, including BOS/EOS. Recover each token's source from packed document_spans plus the immutable canonical source ranges. This is not a document-count ratio or the mixture of the filtered candidate population.

Use only the old dolma-dclm10-005 retained tokenized training pool (6B tokens/3,631,845 documents). Do not rescore, retokenize or reintroduce rejected raw documents. Verify every old shard checksum and every document's source, canonical shuffle position and exclusion from all50000 reserved documents. Preserve original canonical order WITHIN each source. Choose next source by greatest deficit relative to its target token share, then take the next complete retained document. Cycle a source pool only if it is exhausted; record cycles, all selected repeated tokens and actually consumed repeated tokens explicitly. Selection order across sources changes as requested.

Pack with the original TrainingPacker, unchanged document BOS/EOS, spans, attention and loss masks. Full retained documents enter packing; the training stream naturally ends at the common complete-batch token boundary. New preparation is dolma-dclm10-005-rawmix3b-20261015; old preparation and all evaluation artifacts remain untouched. First parallel output is compared to serial reconstruction including labels/masks/spans. Independently count source tokens from the WRITTEN packed spans for the exact consumed prefix. Each source's absolute token share error must be <=0.0001 (0.01 percentage points). Fail validation rather than quietly start a mixture outside tolerance. Report raw/old-pool/reweighted counts and any repetition.

## Validation and delivery

Pure tests verify exact run9/10 authorization and6h caps, token/batch geometry, LR equality, no resume authorization and token-deficit mixing. Shell syntax checks do not load models. The new heavy data path is checked inside its Slurm job; final filtered training is gated on its successful seal and mix tolerance. Training tensor payload is inherited from validated source; the only training changes are exact formal accounting authorization and ensuring final-weight export. No redundant model-loading check is run on the login node.

Deliver persistent W&B links, full reserved-set PPL, exact tokens, source shares/repetition, actual Slurm H200-hours, local checkpoint/export paths and matched raw-versus-filtered delta. Calibration and past full2B results remain separate evidence.
