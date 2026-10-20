"""Compute-node helpers shared by both roles; no model work at import time."""
import hashlib
from pathlib import Path
import torch
from training_state import read_json, file_digest, atomic_json


def byteview(tensor):
    return memoryview(tensor.view(torch.uint8).numpy()).cast('B')


def verify_teacher(cfg, accelerator, out):
    from train import on_main
    spec = cfg['parallel_method']['teacher']
    root = Path(spec['export'])
    def verify():
        path = root/'teacher-manifest.json'
        if file_digest(path) != spec['manifest_sha256']:
            raise ValueError('Teacher manifest changed')
        manifest = read_json(path)
        if (manifest['model_id'] != spec['model_id'] or manifest['revision'] != spec['revision']
                or not manifest['equivalent_tokenization_verified']):
            raise ValueError('Teacher/tokenizer identity mismatch')
        for name, item in manifest['files'].items():
            f = root/name
            if f.stat().st_size != item['size'] or file_digest(f) != item['sha256']:
                raise ValueError('Teacher artifact changed: '+name)
        atomic_json(out/'teacher-provenance.json', manifest)
        return manifest
    return on_main(accelerator, verify)


def loader_preflight(rows, cfg, accelerator):
    from train import collate_rows
    for index in (accelerator.process_index,
                  cfg['target_training_tokens']//8192-8+accelerator.process_index):
        batch = collate_rows([rows[index]], 8192)
        if (int(batch['attention_mask'].sum()) != 8192 or
                batch['document_ids'].shape != (1,8192) or
                int((batch['labels'][:,1:] != -100).sum()) <= 0):
            raise ValueError('Canonical loader first/last preflight failed')
    return True


def numerical_check(device):
    from fast_kd import PrecomputedKD
    from torch.nn import functional as F
    records=[]
    for amp in (False,True):
        for lam,ignored,scale in ((0.,False,1.),(.25,False,3e-5),(1.,False,1.),(.25,True,1.)):
            h=torch.randn(23,32,device=device,requires_grad=True)
            w=(torch.randn(71,32,device=device)*.1).requires_grad_()
            th=torch.randn(23,48,device=device);tw=torch.randn(71,48,device=device)*.1
            labels=torch.randint(0,71,(23,),device=device);labels[::4]=-100
            if ignored:labels.fill_(-100)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                ls=F.log_softmax((h@w.T).float(),-1);lt=F.log_softmax((th@tw.T).float(),-1)
                reference=(1-lam)*F.nll_loss(ls,labels,ignore_index=-100,reduction='sum')+lam*((lt.exp()*(lt-ls)).sum(-1)*(labels!=-100)).sum()
                actual=PrecomputedKD.apply(h,w,th,tw,labels,lam,7)[0]
            ref=torch.autograd.grad(reference*scale,(h,w));got=torch.autograd.grad(actual*scale,(h,w))
            errors=[float((a-b).norm()/a.norm().clamp_min(1e-12)) for a,b in zip(ref,got)]
            if max(errors) >= (.02 if amp else 2e-5):raise ValueError(str(errors))
            torch.testing.assert_close(actual,reference,rtol=2e-6,atol=3e-5)
            records.append(dict(bf16=amp,lam=lam,ignored=ignored,relative_l2=errors))
    return records


def projection_digest(hidden, head):
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
        sample=(hidden[:4]@head.T).cpu().contiguous()
    return hashlib.sha256(byteview(sample)).hexdigest()
