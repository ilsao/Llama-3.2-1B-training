# Scaling Optimal LR Across Token Horizons

Johan Bjorck, Alon Benhaim, Vishrav Chaudhary, Furu Wei, Xia Song. ICLR2025; arXiv:2409.19913v3, revised2025-02-24.
Source: https://arxiv.org/html/2409.19913v3

Eq1 proposes LR*(D)=B D^(-beta). Eq4's joint model/token fit uses beta≈0.32 for models≥760M. Their optimizer is AdamW, the schedule cosine, and empirical horizons start at25B. Holding architecture and batch fixed lets us anchor a ratio at a measured reference. Neither Muon nor WSD, sub2B training, or resumed reheating is validated by that fit. Our exponent is therefore a hypothesis center, with independent multiplier0.5/1/2 validation at0.7B. Scale both Muon and AdamW peaks relative to the matched BE60P20 reference; do not copy the paper's absolute LR constant across tokenizer, batch or optimizer differences.
