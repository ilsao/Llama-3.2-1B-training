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

This revision publishes the raw final endpoint of
`raw4b-intradoc-normuon-20261017`, trained from **random initialization** on
**4,000,055,296 non-padding input tokens**. The architecture and tokenizer are
from Llama 3.2 1B; no pretrained weights or earlier training checkpoint
initialized this run.

| Evaluation | Perplexity | Documents | Valid prediction tokens |
|---|---:|---:|---:|
| Full canonical held-out set | **13.8380209610** | 50,000 | 76,930,208 |
| Fixed training-time subset | 13.8603574479 | 1,000 | 1,479,302 |

The complete original document split was shuffled with seed42 before
tokenization; the last50,000 documents were held out and excluded from training.
Dataset revision: `afa92bfb22366821c5e6cd427cdd036b34b713ef`.
Perplexity is exp(total NLL / total valid prediction tokens), excluding ignored
labels, evaluated at actual optimizer step **19,080**. Evaluation handles
documents independently, with stride8191 for long documents.

## Training recipe

- Unchanged Llama-3.2-1B architecture/tokenizer, revision
  `4e20de362430cd3b72f300e6b0f18e50e7166e08`.
- Residual-half random initialization,seed42;context8192. Unfiltered canonical
  Dolma training prefix,original packing and labels,document-span internal causal
  attention;original positional IDs preserved.
- Hidden linear matrices: native-Muon-compatible **NorMuon after NS5**.
  Momentum=.95,Nesterov,original shape LR scaling;five Newton-Schulz iterations
  with coefficients (3.4445,-4.775,2.0315). The orthogonalized update receives
  FP32 output-row second-moment normalization,beta2=.95,eps=1e-8,no bias
  correction,then is rescaled to preserve its Frobenius norm. No Polar Express.
- Matrix peak LR=.004995613826805825,WD=.1. Auxiliary AdamW peak
  LR=.00014986841480417473,betas=(.9,.95),eps=1e-8;embeddings/tied head/norms
  have no decay. Full parameter assignments are in training-details.json.
- Global preclip gradient norm limit=.5;FP32 parameters/optimizer state,BF16
  autocast,Flash Attention3 and activation checkpointing.
- Global batch16 sequences(8ranks×micro1×accum2),then64(accum8) after
  2,000,027,648 inputs. Event batches may shorten accumulation to whole BS16
  units. Both beta2 values and optimizer states remain intact across switches.
- Two token-clock WSD phases,split at3,333,292,032 inputs. Second peak=.1×.
  Each phase warmup is22,282,240 inputs(170 BS16-equivalent units),not170 actual
  large-batch updates. Each phase's final40% is ramp/hold/final in4:1:1 ratio,
  plateau=.2×phase peak,ending at zero LR. LR values come from the nominal4B
  horizon lookup;no extra LR search was performed for this run.
- Raw endpoint;EMA/weight averaging disabled.
- Allocation:8 NVIDIA H200 GPUs,8 CPUs,5h16m45s,**42.2333333333 H200-hours**,
  including preflight,initialization,training,evaluation,checkpoint and export.
  Local HF reload equality and the full held-out evaluation passed. The final
  bookkeeping guard exited unsuccessfully after these completed because the
  historical shared ledger was unavailable;the endpoint and Slurm cost were
  independently audited without retraining.

[Weights & Biases training curves](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/302d493de0854a3b816a2bee7b5d2a4f).

The previous3B full-set result was15.21170;this4B run improves PPL by9.03%.
Token budget and methods both changed,so this is not an isolated method ablation.
Results use one training seed. The full held-out set has been inspected across
multiple experiments and is not a pristine final test set.

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "asciibase64/Llama-3.2-1B-dolma3_mix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

Previous revisions remain in Git history. The trained context is8192;the larger
positional range in the architecture configuration was not validated by this
run. LICENSE.txt and NOTICE retain the Llama3.2 Community License and attribution.
