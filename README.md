# Llama-3.2-1B-training

`dataset.py` prepares `allenai/dolma3_mix-150B-1025` for randomly initialized
Llama 3.2 1B training. It does not initialize or train a language model.

## Recipe

1. Download the complete pinned native document `train` split. Persist the native
   Hugging Face builder's file order; do not use the incomplete dataset viewer or
   auto-converted Parquet as a substitute.
2. Verify every raw JSONL document ID/text against the native dataset row order.
   Call map-style `Dataset.shuffle(seed=42)` and reserve its last **50,000**
   document indices for evaluation **before any cleaning/tokenization**. Persist
   the permutation, inverse, held-out indices, and first **10,000** held-out
   indices for the fixed in-training evaluation subset.
3. On training only, remove blank documents and exact duplicate unchanged UTF-8
   texts (SHA256). Keep the smallest shuffled position, even when that document
   appears later in physical shard order. Evaluation texts remain untouched;
   evaluation documents never enter deduplication or quality inference.
4. Score eligible Common Crawl documents with the official DCLM OH2.5 + ELI5
   **CPU fastText** classifier. Keep exactly `floor(N / 10)` by the original
   floating point high-quality probability, breaking ties by shuffled position.
   This is global document-count ranking, not per-shard selection, not the fixed
   DCLM threshold `0.018112`, and not a 10% token fraction. All eligible other
   sources remain candidates.
5. Visit candidates in the original shuffled order. Tokenize with the pinned
   Llama tokenizer, explicitly prepend BOS and append EOS once, and select only
   complete documents that fit the remaining **6,000,000,000 non-padding input
   token** budget. Count special tokens. Oversized documents are skipped, not
   truncated. No documents are repeated, and the result may be below 6B.
6. Pack training into **8192** tokens with EOS-separated documents and ordinary
   causal attention. Structural BOS and padding labels are `-100`. The first
   position of every sequence is also ignored because causal LM loss shifts
   predictions by one. Padding uses EOS with `attention_mask=0`, so real EOS
   tokens still count. The tail is padded; no training tokens are dropped.
7. Tokenize all original evaluation documents independently. For long documents,
   windows have length 8192 and stride 8191, so each within-document prediction
   target is scored once. Evaluation never crosses document boundaries. Its
   overlapping input tokens are reported separately from prediction tokens.

The final training artifact cap is **not** the trainer's processed-token counter.
A trainer must additionally count all non-padding inputs across ranks,
microbatches, skipped updates, repetitions and resumed jobs, and stop before 6B.
Evaluation perplexity is `exp(sum NLL / sum valid shifted labels)`, not an average
of per-batch perplexities. These preprocessing scripts do not claim measured
perplexity or training-budget compliance.

## Official implementation and provenance

The recipe follows the source/filtering comparisons in
[DataDecide](https://proceedings.mlr.press/v267/magnusson25a.html), rather than using
its experiment-results cleaning notebook to clean text. It does not replace the
assignment corpus with a released DataDecide corpus or claim an optimal filter.

The vendored scoring function comes from
[DCLM at the pinned commit](https://github.com/mlfoundations/dclm/blob/361714bdd60bb9b7f4b2d8354cebbf0dec0c329e/baselines/mappers/enrichers/quality_prediction_enrichers_calc_fasttext.py).
The MIT license is retained in `third_party/DCLM-LICENSE.txt`. Only the upstream
fastText-dependent type annotation is removed, allowing lightweight tests without
importing fastText. Line joining applies only to classifier input, not stored
training text. The original probability, including fastText epsilon, is retained.

All repository/model revisions and recipe settings are pinned in
`configs/dataset.json`. Direct dependencies are pinned in `requirements-data.txt`;
setup also writes the complete `requirements-resolved.txt` into the environment
and reuses it on subsequent setup jobs. All resolved distribution versions enter
the preparation fingerprint; code hashes, Git commit/status and configuration
are recorded per preparation. Code, config or environment changes require a new
preparation ID. Source filenames distinguish the six verified source families;
unknown source paths stop the pipeline rather than being silently treated as web.

## Slurm execution

**Do not run preparation or model-loading smoke checks on the login node.**
The user explicitly authorized allocating **one H200 even for CPU preprocessing**
because the current account lacks verified CPU-partition access. fastText does
not use this GPU. Jobs use the verified `ACD115198` / `8gpus` settings, 12 CPUs,
200 GiB memory, one H200, and a 24-hour allocation; this does not start training.
All these allocation hours, including setup/idle time, are recorded as data
preprocessing costs, separately from the training limits under the user's rule.

The scripts activate a dedicated Python 3.11 conda environment at
`/work/asciibase64/Llama-3.2-1B-training/environments/data-py311` using the verified
cluster Miniconda installation. Large data, native Arrow caches, fastText weights,
tokenizer assets, SQLite indexes, packed Arrow shards and temporary files stay
under the project's `/work` directory. No Llama pretrained weights are downloaded.
The native Dataset loader can maintain its own download cache in addition to hub
snapshots; provision storage for both raw caches and the complete native Arrow
corpus, plus indexes and the final token artifacts. Full native materialization
and document-level scoring can take substantial time; no throughput is assumed.

The submission wrapper creates log directories **before** `sbatch`. Submit setup,
wait for completion, then submit each stage after the previous one completes:

```bash
bash slurm/submit.sh setup
# Wait for setup and check its exit status.
bash slurm/submit.sh smoke dolma-smoke-001
# Wait for smoke to pass; it loads the real classifier/tokenizer in Slurm.
bash slurm/submit.sh download dolma-dclm10-001
bash slurm/submit.sh split    dolma-dclm10-001
bash slurm/submit.sh clean    dolma-dclm10-001
bash slurm/submit.sh score    dolma-dclm10-001
bash slurm/submit.sh select   dolma-dclm10-001
bash slurm/submit.sh pack     dolma-dclm10-001
bash slurm/submit.sh verify   dolma-dclm10-001
```

**These commands are separate submissions; wait between them.** No full jobs are
submitted by implementation or by the lightweight tests. Existing user-provided
`slurm/template.sbatch` is preserved.

A 15-minute stopping margin derives from the actual Slurm job end time. Normal
stage work checkpoints at committed batches/shards; exit **75** means resubmit
that same stage and ID before continuing. An interrupted source-lineage shard or
split-index generation is rechecked on restart. Global SQL ranking/index builds
and checksum/verification passes may need to restart an interrupted operation;
there is no partial verification seal. Slow filesystem operations or a single
native loader operation can still run past the margin, so Slurm's hard limit
remains the final bound. A per-preparation file lock prevents concurrent writers. Slurm test-only checks
confirmed the account requires at most 12 CPUs and 200 GiB per allocated GPU.
Do not modify committed artifacts or delete native caches referenced by manifests.

Jobs record IDs, timestamps, allocation/node details, exit status and an accounting
snapshot. After each job terminates, reconcile **actual Slurm allocation duration**:

```bash
python3 slurm/accounting.py JOB_ID
```

This small metadata command is suitable for the login node. The shared
`/work/asciibase64/Llama-3.2-1B-training/preprocessing-ledger.json` is idempotent by
job ID and records actual allocated GPU-hours/H200-hours. If typed GPU accounting
is unavailable it stops instead of inventing H200 usage. It is not a training
experiment ledger and does not silently relabel training allocations.

Authentication uses existing `HF_TOKEN`/`HF_TOKEN_PATH` or the existing Hugging
Face login token path; cache relocation preserves that path. Obtain access to the
gated Llama tokenizer before running select/smoke. Credentials are never printed
or persisted in configuration. `DATA_ROOT` and `DATA_CONFIG` may be exported
before submission; custom storage must remain under `/work/asciibase64/`.

## Output and training integration

Each stage writes a manifest with checksums, lineage and statistics under
`preprocessed/PREPARATION_ID/STAGE/`. Cleaning's immutable SQLite index records
winning original row IDs, hashes, source and shuffle position; an `empty` table
records blank-document rejects. All other original training rows are duplicate
rejects. Scoring stores every eligible web score. Selection stores the top-10%
candidate index plus chosen document IDs, text hashes, token IDs and evaluation
slots. Budget skips are reproducible from candidate order and recorded totals.
Evaluation IDs/texts remain available unchanged in the pinned native Arrow cache.

Packed Arrow contains `input_ids`, `attention_mask`, `labels`, `document_spans`
(original row and token offsets) and `evaluation_slot` (-1 for training).
Full verification independently reconstructs the canonical native shuffle and
checks checksums, split bijection, all selected provenance,
quality rank membership, token limits, and reconstructs every sequence/mask.
It seals the immutable pack manifest. A trainer can then load on compute nodes:

```python
from dataset import load_prepared
path = '/work/asciibase64/Llama-3.2-1B-training/preprocessed/dolma-dclm10-001'
train = load_prepared(path, 'train')
evaluation = load_prepared(path, 'evaluation_subset')
# Remove provenance columns when constructing model batches.
train = train.select_columns(['input_ids', 'attention_mask', 'labels'])
```

The loader rejects unverified or changed pack manifests and checks packed shard
checksums by default. It returns a map-style Hugging Face Dataset. Keep the
`evaluation_slot` column until subset selection is complete; the same first
10,000 reserved documents are used across comparable preparations.

## Lightweight checks

```bash
python3 -m unittest discover -s tests -v
python3 dataset.py --help
bash -n slurm/common.sh slurm/setup.sbatch slurm/prepare.sbatch slurm/score.sbatch slurm/submit.sh
```

These tests use small synthetic inputs and SQLite fixtures, without loading
models, corpus data or workers. The Slurm `smoke` stage additionally checks native
HF shuffle semantics, official classifier loading/inference, gated tokenizer
access and fixed-length Arrow/Hugging Face round trips. Full real-data validation
is the separate `verify` stage, not implied by passing lightweight checks.
