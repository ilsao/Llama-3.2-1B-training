# Horizon-scaled WSD campaign

Stage: complete; selections: {'clipping': 0.5, 'decay_tail': 0.6, 'lr_ratio': 0.5, 'resplit_lr_candidate': 'tail-lr030', 'resplit_lr_scale': 0.3}

| Candidate | PPL | Inputs | Physical H200-hours |
|---|---:|---:|---:|
| lr05 | 22.796939 | 700055552 | 8.131111 |
| lr10 | 22.896447 | 700055552 | 8.120000 |
| lr20 | 25.811330 | 700055552 | 8.137778 |
| clip05 | 22.610059 | 700055552 | 8.146667 |
| clip10 | 22.792119 | 700055552 | 8.126667 |
| clip20 | 22.865686 | 700055552 | 8.068889 |
| clipoff | 22.970393 | 700055552 | 8.208889 |
| decay60 | 22.638698 | 700055552 | 4.937778 |
| decay50 | 22.690755 | 700055552 | 4.893333 |
| decay40 | 22.863852 | 700055552 | 4.924444 |
| decay20 | 23.785174 | 700055552 | 4.957778 |
| decay10 | 25.008825 | 700055552 | 4.980000 |

Matched endpoint held-out NLL determines winners. Shared prefix is physically charged once; branches retain inherited tokens and compute.
No significance claim from a single seed. Final 50,000-document evaluation is separate from tuning.
