# Validation infrastructure result

Validation537355 timed out before1B model training. Native NCCL bootstrap stacks and8 SYN-SENT self-connect sockets identify the blocking stage; exact network/firewall cause is unconfirmed.625s=1.388888889H200-hours reconciled, including25s Slurm termination overhead beyond requested600s, logicalvalidation cap flagged exceeded. Training tokens0; no candidatePPL or recipe ranking.

Loopback recovery explicitly approved; no further retries. No failed allocation reset or cleanup.

## Approved recovery result

537466 on25a-hgpn176 completed158s=0.351111111H200-hours.8-rank clipping0.5/1/2/off and model/optimizer/scheduler/RNGrestore checks passed.1,235,814,400-parameter random-init Llama ran20updates/2,621,440 actualinputs without skips; initialization signaturematchesreference. Explicitloopbackbootstrap passed onthisnode; this does not identify the originalnode firewallcause. Originalfailedcost1.388888889 remainscharged; combinedvalidationactual1.74H200-hours.

Three0.7BLR candidates537499/537500/537501 nowstarted; no ranking/PPLqualityresult yet. Deterministiccontrollercontinuesapprovedstagegraph; source-v2andNCCLsettingspinned.
