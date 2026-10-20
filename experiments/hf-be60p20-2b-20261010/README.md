---
library_name: transformers
pipeline_tag: text-generation
license: llama3.2
datasets:
- allenai/dolma3_mix-150B-1025
tags:
- llama
- pretraining
- from-scratch
- muon
- wsd
---
# Llama-3.2-1B-dolma3_mix — BE60P20, 2B tokens

Built with Llama.

Llama 3.2 1B architecture and tokenizer, trained from **randomly initialized
weights** on the unfiltered canonical training split of
`allenai/dolma3_mix-150B-1025`. No pretrained model weights were loaded.
This revision contains the final step 15259 export of `be60p20-lrscaled-2b-20261010`.
Earlier model revisions remain in the repository history.

| Item | Value |
| --- | --- |
| Non-padding training input tokens | 2,000,027,648 |
| Optimizer updates | 15,259 |
| Final held-out PPL | 17.74755488 |
| Evaluation scope | Fixed first 1,000 reserved documents; 1,479,302 valid prediction tokens |
| Full reserved 50,000 / OJ PPL | Not measured for this run |
| Context | 8192 tokens |
| Parameters | 1,235,814,400 |
| Training geometry | 8 NVIDIA H200 / 8 CPU threads total |
| Global input batch | 16 sequences / 131,072 non-padding tokens per update |
| Microbatch / accumulation | 1 sequence per GPU / 2 microbatches |
| Initialization seed | 42 |
| Training allocation duration | 10,288 seconds (2h51m28s) |
| Training allocation cost | 22.862222222 H200-hours |
| torch.compile | Disabled; FlashAttention3, BF16 and fused auxiliary AdamW enabled |

Perplexity is `exp(sum held-out NLL / sum valid prediction tokens)`, excluding
ignored labels. Evaluation ran every 1,525 updates and at the endpoint.
The reported value is a subset measurement, not the full reserved-set or OJ score.

Native PyTorch Muon handles hidden Linear matrices. Auxiliary AdamW handles tied
embeddings/output and normalization/bias parameters. Each trainable parameter is
assigned exactly once; exact groups are in `training-details.json`.

Peak Muon LR **0.0119487688224**, momentum 0.95,
Nesterov, five Newton-Schulz steps, weight decay 0.1, native `adjust_lr_fn=original`.
Auxiliary AdamW peak LR **0.000179231532336**, betas (0.9,0.95),
epsilon 1e-8, weight decay 0.1 on decay groups and zero on excluded groups.
Global gradient clipping 1.0; unchanged Llama 3.2 1B architecture.

The [requested LR horizon transfer](https://arxiv.org/html/2409.19913v3#S6) uses
`LR_long = LR_short * (D_long / D_short)^(-0.32)`. Both peak LRs were multiplied by
0.5974384411214128, transferring from 399,900,672 to 2,000,027,648 input tokens.
Applying this exponent to the mixed Muon/AdamW BE schedule is an experiment
assumption; this run does not independently establish an optimal LR exponent.

BE60P20 schedule: 170 warmup updates; stable peak LR until roughly 40% progress;
first linear cooldown to 20% of peak by 80%; hold at 20% through 90%; final linear
cooldown to zero. Zero-based phase boundaries: (6103,12207,13733,15259).
The final zero-LR batch is included in processed-token accounting.

Dataset revision: `afa92bfb22366821c5e6cd427cdd036b34b713ef`. Complete canonical global HF document
shuffle with seed 42; last 50,000 shuffled documents reserved before tokenization
and excluded from all training. Documents are EOS-separated; padding, structural
BOS and sequence-first targets are excluded from loss. Evaluation scores each
within-document target once using document-local packing with stride 8191.
Tokenizer/model configuration revision: `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
Data split, packing, masks, exact parameter groups, dependency/source identity,
resolved configuration and actual Slurm allocation are in the metadata files.

Standard sharded safetensors and Transformers `LlamaForCausalLM` format. HF reload
verification completed on the training compute node. No custom model code is
required to load the exported model.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

W&B:[be60p20-lrscaled-2b-20261010](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/130117159a054a449f102ddcd454c2d4).
Meta's Llama 3.2 Community License accompanies this architecture and tokenizer
in `LICENSE.txt`; see the original license terms and `NOTICE`.
