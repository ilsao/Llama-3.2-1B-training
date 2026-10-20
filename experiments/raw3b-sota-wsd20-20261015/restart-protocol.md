# One-time fresh restart explicitly approved

The user replied "是" to the concrete one-time fresh restart proposal after
job553561 failed during a dataset module import, before model initialization
and before training inputs. Keep the same formal11 identity, unchanged config,
all historical budgets and the failed20seconds/0.044444444444444446H cost.
Authorize exactly ONE new8H200/8CPU allocation,21580seconds maximum, so combined
allocation remains <=21600seconds/48H. This supersedes the original one-allocation
restriction only for this explicitly approved recovery. No additional automatic
retry, resume, checkpoint training load, cleanup or HF upload.

The original failed run directory contains only .run.lock,resolved-config.json,
resource-accounting.json. Archive that directory intact under /work/asciibase64/
Llama-3.2-1B-training/runs/failed-startups/raw3b-sota-wsd20-20261015/job-553561,
then initialize a new random model at the original logical-run path. Archive is
preservation, not deletion. New W&B identity is created because the failed
startup never initialized one. Every cumulative resource cost stays in the
shared ledger. Original configuration, initial/source seals, failure logs and
controller-state records remain available. source-restart1 adds only exact
fresh-restart budget/metadata handling, while tensor/data/eval code stays equal
to the repaired original run. All previous numerical recipe details remain.
