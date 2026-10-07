# Data preparation equivalence protocol

CONFIRMATORY runtime validation, registered before Slurm execution.
Prediction: pinned native ordering, seed-42 split, selection, token budget,
packing/masks and Arrow bytes are identical for 1 versus 7 workers and committed
resume. Independent eval original archive preserves all fields and reserved order,
including interrupted archive resume. Invalid worker input advances no cursor.
Run tests/preparation_runtime.py through Slurm with eight CPUs total and one H200.
Use the synthetic 128-document fixture, reserved five-document eval and subset two;
full real-corpus verification remains a separate required check. Pass only if all
assertions pass and result.json says passed. Charge actual allocation, even on failure.
Earlier lightweight tests do not constitute this runtime result.

Follow-up validation (registered before the next allocation): create an isolated
synthetic sealed fixture, record training/full-eval/subset loader output, retire only
manifest-owned redundant inputs, and require identical offline loader outputs with
all eval originals/tokens preserved. This is a loader/retention contract fixture,
not evidence of complete real-corpus verification.
