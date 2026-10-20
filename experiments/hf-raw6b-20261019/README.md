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
datasets:
- allenai/dolma3_mix-150B-1025
---
# Llama 3.2 1B architecture trained from scratch on Dolma

This revision publishes the final raw checkpoint of `raw6b-normuon-adam2x-20261018`:
**6,000,082,944 non-padding input tokens**, **28,619 optimizer updates**.
The model was randomly initialized (residual-half, seed42); no pretrained weights
or previous training checkpoint initialized this run. Model architecture and
tokenizer revision: `meta-llama/Llama-3.2-1B` at
`4e20de362430cd3b72f300e6b0f18e50e7166e08`.

| Evaluation | PPL | Documents | Valid prediction tokens |
|---|---:|---:|---:|
| Fixed training-time subset | 12.7739359504 | 1,000 | 1,479,302 |
| Additional disjoint confirmation subset | 12.9224263500 | 9,000 | 13,666,812 |
| Combined fixed subset | **12.9078477387** | **10,000** | **15,146,114** |

The full reserved 50,000-document set has not been evaluated for this checkpoint.
PPL is exp(sum held-out NLL / sum valid prediction tokens), not a mean of subset PPLs.
Documents are evaluated independently, with stride8191 for long documents.
The entire original document split was shuffled with seed42 before tokenization;
the last50,000 documents were held out and excluded from all training inputs.
Dataset revision: `afa92bfb22366821c5e6cd427cdd036b34b713ef`.

## Training configuration

- Context8192; intradoc document-span causal mask, prohibiting cross-document attention.
- NS5 NorMuon for hidden matrices, momentum0.95, Nesterov, neuron variance beta2=0.95,
  original shape LR scaling. Matrix peak LR: **0.00425961126463665**.
- Auxiliary AdamW: betas(0.9,0.95), eps1e-8; peak LR **0.000255576675878199**,
  equal to the nominal6B lookup LR multiplied by2.
- Ordinary weight decay0.1 for matrices; embeddings/tied head/norms have no decay.
- Global gradient clipping0.5; FP32 weights/states, BF16 autocast, FlashAttention3,
  activation checkpointing. No EMA, parameter averaging or Polar Express.
- Global batch16 to64 at3,000,107,008 inputs (8ranks, microbatch1, accumulation2 to8).
- Phase split5,000,003,584 inputs, second peak0.1x. Token-clock WSD with170
  BS16-equivalent warmup units per phase. Each phase ends with40% cooldown:
  26 2/3% ramp to0.2x,6 2/3% hold,6 2/3% ramp tozero.
- One completed8H200/8CPU allocation:8h3m36s /64.48H200-hours.
  Including two startup failures, logical-run cost is64.7066666667H200-hours.

[Training run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/3dccd801892342459af12b46727fa908).
Detailed resolved configuration, optimizer groups and provenance are included as JSON files.
The canonical data tail was extended in an isolated overlay, preserving all
original files and existing non-padding tokens/document spans; only terminal
padding was replaced with following canonical training-document tokens.

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

This is a base pretrained model, not an instruction-tuned or safety-aligned model.
The Llama3.2 Community License and attribution are retained in LICENSE.txt and NOTICE.
Previous model revisions remain available in repository history.
