# Eight teacher GPUs plus eight student GPUs: completed measurements

The user's proposed geometry works for BF16 final hidden-state transport.
The earlier recommendation favoring online8GPU assumed a fixed total of8 GPUs
split4+4. It did not evaluate the requested8 teacher plus8 student configuration.
The new validation retains8 student replicas and adds8 teacher replicas on a
different node, at matched global BS16/64 and8192-token context.

## Results

Two warmup windows and six timed optimizer windows per format/batch, followed by
six GPU-resident student reference windows. Student LR/decay LR0, fresh seed42
weights, real complete-vocabulary KD backward and optimizer computation.

| Payload | Global batch | Teacher forward/copy/write | Resident student demand | Student including disk |
|---|---:|---:|---:|---:|
| BF16 final hidden |16|264,047 tok/s|181,620 tok/s|179,693 tok/s|
| BF16 final hidden |64|288,081 tok/s|217,409 tok/s|209,756 tok/s|
| BF16 full logits |16|20,886 tok/s|180,506 tok/s|21,127 tok/s|
| BF16 full logits |64|20,456 tok/s|219,036 tok/s|20,324 tok/s|

Hidden-state production has45.4% headroom at BS16 and32.5% at BS64 relative to
resident student demand. End-to-end disk overhead is1.07% and3.65%, respectively.
Compared with the previous sustained online8GPU run's median131,270/151,596 tok/s,
the new student rate is1.369x/1.384x. These runs used different nodes/windows;
the comparison is planning evidence, not a controlled long-duration scaling claim.

Full-logits writes improve to5.56–5.69GB/s but remain far below student demand.
The student with disk logits is8.54–10.78x slower than its resident reference.
More teacher GPUs help the compute portion; they do not remove the much larger
full-vocabulary storage requirement. Use post-normalization hidden states and
the frozen official teacher output head on the student to preserve the complete
teacher distribution. No top-k approximation was used.

Synthetic hidden/logits loss difference0, gradient relative L2~1.02e-10; teacher
attention GQA/document isolation/padding checks passed. Real transported hidden
states reconstructed the teacher-head projection checksum exactly. Repeated
resident/disk objectives matched. Every benchmark payload was acknowledged after
backward and removed from its own spool; original failure artifacts remain.

## Three-billion-token projection

The unchanged student token plan ends at3,000,107,008 nonpadding inputs and14,312
updates. Measured hidden-state rates imply4.306h of training-loop time, or4.456h
after adding the historical data/journal/logging overhead estimate. Initialization,
evaluation, export and a stopping margin are additional. This is about27% less
time than the corresponding online8GPU estimate6.073h.

Using the slowest of the six measured windows for each batch, adding historical
host overhead, allowing20% further slowdown and23min for startup/evaluation/exit
gives5.856h. The proposed conservative allocation is therefore6h per role:
8H200/8CPU teacher plus8H200/8CPU student, maximum96 H200-hours combined.
This is a candidate budget, not an approved allocation or an observed training
duration. The teacher's waiting/backpressure allocation must be charged too;
exiting after the last required output can reduce actual cost. Even the simple
both-roles-active training-loop estimate is68.89H before host/evaluation overhead,
so the previous64H proposal does not provide sufficient margin for this geometry.

Use decimal1TB as the hidden-cache capacity ceiling; start the student after a
small ready buffer rather than waiting to fill1TB. Keep the student architecture,
seed, full-vocabulary objective, optimizer groups/LRs, canonical data/masks and
3B token schedule unchanged. The persistent full-training cache integration is
not yet implemented or submitted. Resource gates have not been expanded.

## Startup repair and accounting

Initial producer575177 (67s) and consumer575178 (53s) failed before timed data
processing. The consumer's nonzero ranks could read their own READY path before
rank0 atomically published it. The original files and failures are preserved.
The user explicitly authorized exactly one replacement per role. Source-v2 waits
for both roles' READY files before reading either; the new regression reproduces
a delayed local rank0 with an already-ready external peer. Four stdlib tests passed.

Replacement producer575196 completed in390s; consumer575197 in382s, each8H200.
Their combined cost is1.7155555556H. Including the two initial failures gives
1.9822222222H for this8+8 validation, below the approved5.0666666667H replacement
plan ceiling. The same logical validation IDs retain all four allocations.
No model checkpoint was loaded; inherited conservative input reservations remain.

The study's reconciled cumulative cost is95.4186111111H. A new96H full-run maximum
would require a study cap of at least191.4186111111H (191.419 rounded upward),
and a scoped exception to the prior64H logical-run ceiling. The user's replacement
approval applies to the validation jobs only. No full-training authorization,
reservation, submission, automatic additional retry or old-artifact deletion has
been inferred from it.

Detailed evidence: results.json, training-resource-proposal.json,
validation-accounting.json, repair-accounting.json, repair-approval.json and
the immutable source-v2/ snapshot and repair-seal.json inventory at the study root.
