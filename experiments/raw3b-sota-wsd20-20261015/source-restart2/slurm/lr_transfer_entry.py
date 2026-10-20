"""Allocated-only validation plus the existing training worker entrypoint."""
import copy
import os
import time
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from training_state import load_config
    path=sys.argv[sys.argv.index('--config')+1];cfg=load_config(path)
    if cfg['lr_transfer']['kind']=='validation':
        import torch
        from recipe_runtime import EMA
        torch.set_num_threads(1)
        device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
        torch.cuda.set_device(device)
        started=time.monotonic()
        tiny=torch.nn.Linear(2,2,bias=False).to(device)
        with torch.no_grad():tiny.weight.fill_(1.)
        shadow=EMA(tiny,cfg);batch=cfg['world_size']*cfg['microbatch_size']*cfg['gradient_accumulation_steps']*cfg['sequence_length']
        shadow.update(batch);shadow.update(2*batch)
        if shadow.updates!=1:raise ValueError('EMA cadence initialization failed')
        with torch.no_grad():tiny.weight.fill_(3.)
        shadow.update(3*batch);shadow.update(4*batch)
        saved=tiny.weight.detach().clone();before_rng=torch.cuda.get_rng_state(device)
        with shadow.weights(.6):
            if not torch.allclose(tiny.weight,torch.full_like(tiny.weight,1.8)):raise ValueError('Incorrect EMA recurrence')
        if not torch.equal(saved,tiny.weight) or not torch.equal(before_rng,torch.cuda.get_rng_state(device)):raise ValueError('EMA measurement changed training weights/RNG')
        state=copy.deepcopy(shadow.state_dict());restored=EMA(tiny,cfg);restored.load_state_dict(state)
        if restored.updates!=2 or restored.last_tokens!=4*batch:raise ValueError('EMA token-cursor roundtrip failed')
        print('EMA_PREFLIGHT_OK rank='+os.environ['RANK']+' seconds='+str(time.monotonic()-started),flush=True)
        del tiny,shadow,restored
    from slurm.training_entry import main as worker_main
    return worker_main()

if __name__=='__main__':sys.exit(main())
