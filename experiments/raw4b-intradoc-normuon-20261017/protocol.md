# Approved fresh4B intradoc NorMuon run

The user reviewed the concrete4B configuration and explicitly approved:「允許，提交」.
Run `raw4b-intradoc-normuon-20261017`, category full, formal14; preserve all13
historical formal identities and all failed/complete allocation costs. The exact
configuration digest is `ed7d3b6317bd308ce76d1bcecc5db4c3332d8db2568ad56cdee1118bb0ab19bf`.
This named approval supersedes the three-formal-run and120-minute gates only for
this run. One8H200/8CPU allocation,7h/56H maximum, including preflight, startup,
evaluation, checkpoint/export and idle time. No training checkpoint load,
automatic retry/resume/requeue, artifact cleanup, HF upload or other promotion.

## Recipe

- Architecture/tokenizer: unchanged Llama-3.2-1B, revision
  `4e20de362430cd3b72f300e6b0f18e50e7166e08`; random residual-half weights,seed42.
- Parent preparation: immutable `dolma-raw-baseline-20261008`, canonical complete
  seed42 split; all50000 reserved documents excluded from training. Use the
  existing canonical raw packed prefix; no filtering, reshuffling, download or
  change of document selection. Preserve tokenizer/special-token labels,
  original positions,8192 context and masks. Runtime attention uses archived
  `document_spans` for document-internal causal attention. Packing provenance
  remains unchanged; runtime mask is recorded explicitly in configuration.
-1B winner's native-Muon-compatible NorMuon: NS5 first,then FP32 second-moment
  normalization per original output-neuron row,beta2=.95,eps=1e-8,no bias
  correction;restore original update Frobenius norm. Muon momentum=.95,
  Nesterov,original shape LR scaling;no PE/cautious decay/row equilibration.
- Matrix peak LR=.004995613826805825;auxiliary AdamW=.00014986841480417473,
  betas=(.9,.95),eps=1e-8. These are exactly the nominal4B lookup row from
  `experiments/lr-transfer-horizons-20261015/calibration-result.json`,not a new
  fit or an adjustment for the55,296-token endpoint overshoot.
- Matrix WD=.1;embeddings/tied head/norms no decay;global preclip norm=.5.
  Parameters and optimizer states FP32,BF16 autocast,FA3,activation checkpointing.
- Global batch16 sequences:8ranks×micro1×accum2. At2,000,027,648 inputs,
  next batch uses global64(accum8). Truncate accumulation to whole BS16 units
  at predeclared events;preserve optimizer states and both beta2=.95 values.
- Token clock reference unit131072inputs;phase1 [0,25431),phase2 [25431,30518).
  Phase split3,333,292,032inputs;second peak=.1×;each warmup170reference
  units=22,282,240inputs. Each phase tail40%,ramp/hold/final=4:1:1,
  plateau=.2×phase peak,terminal LR zero. Preserve3B recipe proportions.
- Stop at4,000,055,296 confirmed nonpadding inputs /19,080 successful and
  attempted updates. Charge every processed/skipped batch;6B hard cap remains
  above the tighter configured4B target. Any skipped/nonfinite update stops
  this experiment and prevents a successful completion receipt.
- Raw endpoint only;no EMA/weight-averaging observers in actual training.

## Validation and completion

Use the known ACD115198/8gpus/H200 allocation,one launcher,8 total CPUs,
200G memory,one library thread per rank. Source complete dependencies including
third_party;preflight imports the real data entrypoint before any model work.
Run the proven synthetic native-NS parity,NorMuon normalization/state,
FA3-vs-FP32-math-SDPA forward/backward,GQA/padding/document isolation,tiny Llama
activation-checkpoint transport and8rank identity checks inside the same
allocation,at most600seconds. Synthetic preflight has no real training inputs.
Its toy observer checks reuse historical numerical tests;actual training EMA
is disabled and allocates no averaging shadows.

W&B project lab5-training-llama,new persistent ID;log every10actual updates.
Training evaluation uses the same first1000 reserved documents/1,479,302 valid
prediction tokens,every1908actual updates and common10%token/phase/batch knots,
plus the final endpoint. Record exact prediction-weighted NLL,PPL,preclip norm,
each group's LR,current-step throughput and globally counted inputs.

Reserve900seconds for end-evaluation/export and a minimum300second stopping
margin before the7h cap. Source uses faithful complete checkpoint state for
audit;checkpoint existence does not authorize continuation. After the exact
target and final raw HF reload equality pass,load the export for evaluation
only and evaluate all50000 original reserved documents/76,930,208prediction
tokens. No extra evaluation allocation is automatically submitted. Compare full
PPL with3B SOTA15.21170256746107,disclosing the different token budgets and
single-seed nature. Do not select a checkpoint based on full-set PPL.

Large artifacts,caches,logs and weights go under/work/asciibase64. All source,
config,lookup provenance,raw prefix metadata and historical ledger snapshot are
sealed before submission. The controller is the only submission owner;reserve
before sbatch,recover ambiguous submissions by exact tag,never resubmit. Its
dedicated cron only queries/reconciles and stops at the terminal allocation;
it cannot create a second allocation. Preserve other cron entries.
