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

This revision contains the final checkpoint of `raw3b-sota-20261015`, trained from **random initialization** on **3,000,107,008 non-padding input tokens**. It uses the Llama 3.2 1B architecture and tokenizer. No pretrained model weights or earlier training checkpoint initialized this run.

| Evaluation | Perplexity | Documents | Valid prediction tokens |
|---|---:|---:|---:|
| Full canonical held-out set | **15.2777345960** | 50,000 | 76,930,208 |
| Fixed training-time subset | 15.2238606449 | 1,000 | 1,479,302 |

The complete document split was shuffled with seed 42 before tokenization; the last 50,000 documents were held out. Dataset revision: `afa92bfb22366821c5e6cd427cdd036b34b713ef`. Perplexity is exp(total held-out NLL / total valid prediction tokens), excluding ignored labels, evaluated at step **22,889**. Training used the unfiltered canonical training prefix, with no repeated training documents.

## Training recipe

- Architecture/tokenizer: `meta-llama/Llama-3.2-1B`, revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Residual-half random initialization, seed 42; training context 8,192.
- Native Muon on hidden linear matrices; AdamW on embeddings/output and remaining parameter groups. Full assignments are in `training-details.json`.
- Peak learning rates: Muon `0.005593716406714465`, AdamW `0.00016781149220143396`.
- Muon: momentum 0.95, Nesterov, 5 Newton–Schulz iterations, original dimension adjustment, weight decay 0.1.
- AdamW: betas (0.9, 0.95), epsilon 1e-8. Embeddings/norms/biases have no weight decay.
- Global gradient clipping 0.5; bf16 and Flash Attention 3.
- Two WSD phases split at update 19,074 (approximately 5/6 of training). The second phase uses 0.1 times peak LR. Each phase has 170 warmup updates, a 40% first decay, a 10% hold at 20% peak, and a 10% final decay to zero.
- EMA disabled. Global batch: 16 sequences / 131,072 non-padding input tokens per update, 8 GPUs, microbatch 1 per GPU, gradient accumulation 2.
- Allocation: 8 NVIDIA H200 GPUs, 8 CPUs, 4h 20m 21s / **34.7133333333 H200-hours**, including initialization, evaluation, checkpointing and export. Local export reload verification passed.

[Training curves on Weights & Biases](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/f905ff728cec44f4b2bce516804b3ab0).

The matched single-WSD control, trained on the same 3B-token prefix with the same initialization and peak LRs, reached full-set PPL 15.5799369980. This resplit recipe reduced PPL by 1.94% in this seed-42 comparison. The learning rates were extrapolated from shorter WSD experiments; this run does not establish a globally optimal LR.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

The repository retains previous revisions in Git history. Training context was 8,192; the architecture configuration's larger positional range was not validated by this training run. `LICENSE.txt` and `NOTICE` retain the Llama 3.2 Community License and attribution.
