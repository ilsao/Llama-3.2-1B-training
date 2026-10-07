# AdamW baseline protocol

Prefer baseline before optimizer experiments. Run from random initialization with
pinned Llama 3.2 1B architecture, context 8192 and sealed preparation. Use the fixed
held-out subset, aggregate valid NLL/tokens before exponentiation, and required
W&B fields in lab5-training-llama. Start with configs/train-pilot.json following
runtime validation. Do not claim held-out perplexity <=20 until actually measured.
The 16-H200-hour baseline target is a planning estimate; initialization, evaluation,
checkpointing and idle allocation count. Every logical run retains cumulative
processed tokens <=6B and allocation <=64 H200-hours across resumes. Choose 4/8
GPUs from the locked scaling rule; geometry changes create a new run. Any changed
recipe/budget must be recorded in a new protocol before its experiment starts.
