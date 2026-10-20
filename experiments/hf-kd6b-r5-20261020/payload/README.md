---
library_name: transformers
pipeline_tag: text-generation
license: llama3.2
tags:
- llama
- text-generation
- safetensors
- pretraining
- from-scratch
- knowledge-distillation
datasets:
- allenai/dolma3_mix-150B-1025
---
# Llama 3.2 1B trained from scratch with online knowledge distillation

Final checkpoint of `kd6b-r5-20261020-seed42`: **6,000,082,944 non-padding student input tokens**, **28,619 optimizer updates**.
The student uses the Llama 3.2 1B architecture, randomly initialized with seed42 and residual-half scaling. No pretrained student weights or previous training checkpoint were loaded.
The frozen teacher is the official pretrained `meta-llama/Llama-3.2-3B`, revision `13afe5124825b4f3751f836b40dafda64c1ed062`.

| Evaluation | PPL | Documents | Valid prediction tokens |
|---|---:|---:|---:|
| Fixed training-time subset | 12.2678137445 | 1,000 | 1,479,302 |
| Additional disjoint subset | 12.4131968566 | 9,000 | 13,666,812 |
| Combined fixed subset | **12.3989218778** | **10,000** | **15,146,114** |

PPL is exp(total held-out NLL / total valid prediction tokens). The full reserved 50,000-document set has not been evaluated for this checkpoint.
The canonical complete dataset document split was shuffled with seed42 before tokenization; the last50,000 documents were reserved and excluded from training. Documents are evaluated independently, using stride8191 for long documents.
Dataset: `allenai/dolma3_mix-150B-1025`, revision `afa92bfb22366821c5e6cd427cdd036b34b713ef`.

## Training

- Student architecture/tokenizer: `meta-llama/Llama-3.2-1B`, revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`; context8192; intradoc causal attention.
- Objective: 0.75 cross-entropy + 0.25 full-vocabulary KL(teacher || student), temperature1. Teacher and student each processed6,000,082,944 input tokens, accounted separately.
- Eight H200s serve both teacher inference and student training online, with no persistent logits cache.
- NS5 NorMuon for hidden matrices: momentum0.95, Nesterov, neuron variance beta2=0.95; peak matrix LR0.00425961126463665; ordinary matrix weight decay0.1.
- Auxiliary fused AdamW for tied embedding/output head and norms: peak LR0.000255576675878199 (table LR times2), betas(0.9,0.95), epsilon1e-8, no weight decay.
- Global gradient clipping0.5; FP32 student weights/optimizer states, BF16 arithmetic; microbatch2; global batch16 to64 at3,000,107,008 input tokens.
- Phase split at5,000,003,584 input tokens; second peak0.1x;170 BS16-equivalent token warmup units per phase;40% tail with4:1:1 ramp/hold/final and0.2 plateau ratio.
- R5 acceleration: fused CE/KL, local compilation, activation checkpointing disabled, and whole-matrix NorMuon ownership across ranks.
- One8H200/8CPU training allocation: **8h5m20s /64.7111111111 H200-hours**, including initialization, evaluation and export. This is within the explicitly approved72H allocation but exceeds the original64H target.
- All optimizer owner states and the model export passed reload verification inside the training allocation.

The10k PPL is3.94% lower than the previous6B non-distilled checkpoint (12.9078477387).
There is no matched full6B unaccelerated KD reference, so this comparison does not prove that acceleration preserves final PPL exactly.
[Training curves](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/58970e70c88c).
Resolved configuration and training/evaluation metadata are included as JSON files.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

This is a base language model. Llama3.2 license and attribution are retained in LICENSE.txt and NOTICE.
Previous checkpoints remain available in repository history.
