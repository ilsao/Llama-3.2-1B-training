# 初始化與配方 sweep

狀態：complete；已使用或保留 314.449 / 256 H200-hours。

Source commit：74c50782090e8930286444183b0d34e58360622f

| 候選 | variant | tokens | subset PPL | full50k PPL | H200-hours | W&B |
|---|---|---:|---:|---:|---:|---|
| adam05 | raw | 150077440 | 77.89800 | — | 1.951111111111111 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/47c2b025f07e46bdb95cd4ace5324992) |
| adam20 | raw | 150077440 | 54.56079 | — | 1.9288888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/0b5f98439c9a414cb11a82fede5edfeb) |
| batch32 | raw | 388759552 | 36.34659 | — | 4.155555555555556 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/270538cdd73044c38f041ed47720e067) |
| batch8 | raw | 388759552 | 28.30284 | — | 5.591111111111111 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8063b7efcc824ad0964ed3d98c561457) |
| clip-groups | raw | 388759552 | 30.22251 | — | 4.588888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/4cfb6eda81e3408eb6f34c8df87c2e22) |
| clip-late | raw | 388759552 | 30.18810 | — | 4.586666666666667 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/41272e0afb384d69b85c4580a08e8bed) |
| combo-baseline | raw | 388759552 | 30.19027 | — | 4.642222222222222 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/5b62246388b747ca869fcb33fc384f09) |
| combo-merged | raw | 388759552 | 29.00328 | — | 4.604444444444445 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/b640ae2751294bc9acf1a8cfa771c47a) |
| combo-merged-reranked | raw | 388759552 | 28.31854 | — | 4.624444444444444 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/89f78f9d3fe74868b20d6cea530ecd71) |
| embedding_half | raw | 150077440 | 63.81895 | — | 1.92 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/a24bf5ff68fe48da9e25c1d2fbfc92e6) |
| fan_in | raw | 150077440 | 60.30414 | — | 1.9 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/c5a9115ab3e44b9a9d34fe6d9a3e3b1d) |
| filtered | raw | 388759552 | 34.67434 | — | 4.6866666666666665 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/e3bab43c6432479b8d7b68d9c278121a) |
| final-2b | raw | 2000027648 | 16.88644 | 16.85416378575511 | 23.32 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/a52f3c49c8384ebd93efcdb23cbd0620) |
| gaussian | raw | 150077440 | 65.18871 | — | 1.9111111111111112 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/73756f2bae54428d9cbef7ac6a54828d) |
| muon07 | raw | 150077440 | 82.13748 | — | 1.9333333333333333 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/9d6da079d14d484cab2d95b6fa2705cb) |
| muon13 | raw | 150077440 | 58.12472 | — | 1.9133333333333333 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/63cf9b0cc8ad49debb937292e20c4ca2) |
| orthogonal | raw | 150077440 | 63.79565 | — | 1.9355555555555555 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/79c16e04ffca4f1bb91f7013a6ecb5a1) |
| promote-adam20 | raw | 388759552 | 28.10884 | — | 4.637777777777778 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/4b46ac3bb75e43b9aade45b0f09c2e01) |
| promote-muon13 | raw | 388759552 | 29.28552 | — | 4.608888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/5b4fb731337846ac9c08e92bbb79cecf) |
| promote-residual_depth | raw | 388759552 | 29.36125 | — | 4.604444444444445 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/9beec4aea7cb4ed68ece2e19c172ae72) |
| promote-residual_half | raw | 388759552 | 29.33736 | — | 4.648888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/e8657584b11a469ba4e00b984148f81b) |
| residual_depth | raw | 150077440 | 56.38224 | — | 1.9155555555555555 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/1d385f6a61dc4eed94862a04a164ad3f) |
| residual_half | raw | 150077440 | 59.84267 | — | 1.9133333333333333 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/3952b1e94fd949838adda05b5dcf0f2b) |
| seed43-baseline | raw | 388759552 | 29.91374 | — | 4.62 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/5c7e46171e604aa9a9afb28964b3eaed) |
| seed43-candidate | raw | 388759552 | 28.12310 | — | 4.644444444444445 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/d263a2fe4ce44c33be7ce9abdf54262e) |
| short-baseline | raw | 388759552 | 30.15295 | — | 4.655555555555556 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8af035717c3b4a6bb354a18afa7a24da) |
| short-baseline | 0.99 | 388759552 | 30.18635 | — | 4.655555555555556 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8af035717c3b4a6bb354a18afa7a24da) |
| short-baseline | 0.999 | 388759552 | 31.36532 | — | 4.655555555555556 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8af035717c3b4a6bb354a18afa7a24da) |
| short-baseline | 0.9995 | 388759552 | 32.03441 | — | 4.655555555555556 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/8af035717c3b4a6bb354a18afa7a24da) |
| tail-lr005 | raw | 2000027648 | 17.15894 | 17.123328455302058 | 4.437777777777778 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/c6d50ae8ec5a4ebea82b43dad71913c2) |
| tail-lr010 | raw | 2000027648 | 17.06407 | 17.041813353653176 | 4.471111111111111 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/13316ba29c9045bda5d9b268884b523f) |
| tail-lr020 | raw | 2000027648 | 17.30247 | 17.287776059421155 | 4.448888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/2c0f55af65c94c64b7f2a0ad9eaf45a7) |
| wd003 | raw | 388759552 | 32.25724 | — | 4.608888888888889 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/67010060618d496ea3f2c37fd213cc25) |
| wd005 | raw | 388759552 | 31.41100 | — | 4.6 | [run](https://wandb.ai/cerulean-labs/lab5-training-llama/runs/c6cc425ab7174853b29aaad0c609a138) |

## 分組與組合選擇

{
  "batch_final_feasible": "short-baseline",
  "candidate": "promote-adam20",
  "early-adam": [
    "adam20",
    "gaussian"
  ],
  "early-init": [
    "residual_depth",
    "residual_half"
  ],
  "early-muon": [
    "muon13",
    "gaussian"
  ],
  "ema": "raw",
  "final_changes": {
    "ema": [],
    "learning_rate": 0.00017923153233642382
  },
  "short-adam": "promote-adam20",
  "short-batch": "batch8",
  "short-clip": "short-baseline",
  "short-data": "short-baseline",
  "short-init": "promote-residual_half",
  "short-muon": "promote-muon13",
  "short-wd": "short-baseline",
  "tail": {
    "candidate": "tail-lr010",
    "full_perplexity": 17.041813353653176,
    "historical_standard": 17.2127001491
  }
}
