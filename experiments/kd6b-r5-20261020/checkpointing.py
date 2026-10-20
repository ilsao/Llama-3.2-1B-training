"""Complete whole-matrix-owner checkpoints; no training resume entrypoint."""
import os
import random
import uuid
from pathlib import Path
from training_state import atomic_json,file_digest,digest,read_json


def collectively(acc,call):
    import torch.distributed as dist
    error=None;value=None
    try:value=call()
    except Exception as exc:error=repr(exc)
    errors=[None]*acc.num_processes
    dist.all_gather_object(errors,error)
    if any(errors):raise RuntimeError('Rank operation failed: '+str(errors))
    return value


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def assert_same(a,b):
    import numpy as np
    import torch
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a.detach().cpu(),b.cpu())
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b,(a,b)


def save_verified(path,state):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(temporary,map_location='cpu',weights_only=False)
    assert_same(state,loaded)
    del loaded
    temporary.rename(path)
    return dict(name=path.name,size=path.stat().st_size,sha256=file_digest(path),reload_equal=True)


def serializer_preflight(acc,out):
    import torch
    p=torch.nn.Parameter(torch.arange(16,device=acc.device,dtype=torch.float32).view(4,4)/16)
    opt=torch.optim.AdamW([p],lr=.001,betas=(.9,.95),fused=True)
    p.grad=torch.ones_like(p);opt.step()
    return collectively(acc,lambda:save_verified(out/f'serializer-rank{acc.process_index}.pt',
        dict(parameter=p.detach(),optimizer=opt.state_dict(),rng=rng_state(),rank=acc.process_index,
             parameter_names=['synthetic'],ownership={'synthetic':acc.process_index})))


def save_checkpoint(acc,raw,opts,schedulers,owners,out,state):
    import train
    name=train.on_main(acc,lambda:f'checkpoint-{state["global_step"]:06d}-{uuid.uuid4().hex[:10]}')
    pending=out/'checkpoints'/(name+'.partial')
    train.on_main(acc,lambda:pending.mkdir(parents=True,exist_ok=False))
    names={id(p):n for n,p in raw.named_parameters()}
    owned_names=[[names[id(p)] for p in g['params']] for g in opts[1].param_groups]
    expected={n for n,r in owners.metadata['assignments'].items() if r==acc.process_index}
    assert set(n for group in owned_names for n in group)==expected
    local=dict(rank=acc.process_index,matrix_optimizer=opts[1].state_dict(),
               parameter_names_by_group=owned_names,ownership=owners.metadata,rng=rng_state())
    matrix=collectively(acc,lambda:save_verified(pending/f'owner-rank{acc.process_index}.pt',local))
    import torch.distributed as dist
    records=[None]*acc.num_processes;dist.all_gather_object(records,matrix)
    def finish():
        adam=save_verified(pending/'adamw.pt',dict(optimizer=opts[0].state_dict(),
             parameter_names_by_group=[[names[id(p)] for p in g['params']] for g in opts[0].param_groups]))
        backend=raw.config._attn_implementation
        raw.config._attn_implementation='sdpa'
        try:raw.save_pretrained(pending/'student',safe_serialization=True,max_shard_size='2GB')
        finally:raw.config._attn_implementation=backend
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(out/'assets',local_files_only=True).save_pretrained(pending/'student')
        for role in ('teacher','student'):
            atomic_json(pending/f'{role}-input-journal.json',read_json(out/f'{role}-input-journal.json'))
        atomic_json(pending/'training-state.json',{**state,'schedulers':[s.state_dict() for s in schedulers],
                                                 'owner_parameter_map':owners.metadata})
        files=[dict(name=str(p.relative_to(pending)),size=p.stat().st_size,sha256=file_digest(p))
               for p in sorted(pending.rglob('*')) if p.is_file()]
        atomic_json(pending/'complete.json',dict(files=files,owner_receipts=records,adamw_receipt=adam,
                    state_digest=digest(read_json(pending/'training-state.json')),all_optimizer_states_reload_verified=True))
        final=pending.with_name(name);pending.rename(final)
        atomic_json(out/'latest.json',dict(checkpoint=name))
        return str(final)
    return train.on_main(acc,finish)

