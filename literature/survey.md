# Horizon WSD survey

- [LR scaling notes](scaling-lr-2409.19913.md): fixed architecture/batch token-horizon power-law transfer; cosine/AdamW evidence motivates a tested heuristic for WSD/Muon.
- [WSD notes](wsd-2410.05192.md): stable-trajectory branching and decayed-checkpoint continuation motivate timing and reheating comparisons.

Open question: can the existing hold/zero recipe retain its ranking under0.7B LR/clipping tuning and2B transfer? This campaign tests the requested recipe; it does not fit a new WSD scaling exponent from a single horizon.
