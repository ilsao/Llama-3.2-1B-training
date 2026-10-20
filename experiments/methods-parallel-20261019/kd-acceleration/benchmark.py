#!/usr/bin/env python3
"""Allocated-node-only paired KD validation. LR=0; never changes model weights.

Counts every full-model input (including repeated benchmark windows). This is
performance/numerical validation, not a PPL or convergence experiment.
"""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('Submit the benchmark with sbatch')
import sys, json, time, statistics, contextlib
from pathlib import Path
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'source'
sys.path.insert(0, str(SOURCE))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from training_state import atomic_json, read_json, DEFAULT_ROOT
import train, parallel_losses, parallel_runtime
from fast_kd import PrecomputedKD, kd_loss as fast_loss


def small_checks(device):
    """Dense reference, masked/all-ignored, identical teacher, tied embedding path."""
    records = []
    torch.backends.cuda.matmul.allow_tf32 = False
    for amp in (False, True):
        for lam, scale, ignored, identical in ((0.,1.,False,False),(.25,3e-5,False,False),
                                               (1.,1.,False,False),(.25,1.,True,False),
                                               (.25,1.,False,True)):
            h = torch.randn(19,32,device=device,requires_grad=True)
            w = (torch.randn(67,32,device=device)*.1).requires_grad_()
            th,tw = (h.detach(),w.detach()) if identical else (torch.randn_like(h),torch.randn_like(w)*.1)
            labels=torch.randint(0,67,(19,),device=device); labels[::3]=-100
            if ignored:labels.fill_(-100)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                ls=F.log_softmax((h@w.T).float(),-1); lt=F.log_softmax((th@tw.T).float(),-1)
                ref=(1-lam)*F.nll_loss(ls,labels,ignore_index=-100,reduction='sum')+lam*((lt.exp()*(lt-ls)).sum(-1)*(labels!=-100)).sum()
                actual=PrecomputedKD.apply(h,w,th,tw,labels,lam,7)[0]
            a=torch.autograd.grad(ref*scale,(h,w)); b=torch.autograd.grad(actual*scale,(h,w))
            errors=[float((x-y).norm()/x.norm().clamp_min(1e-12)) for x,y in zip(a,b)]
            assert max(errors)<(.02 if amp else 2e-5), (amp,lam,errors)
            torch.testing.assert_close(actual,ref,rtol=2e-6,atol=2e-5)
            records.append(dict(bf16=amp,lam=lam,scale=scale,all_ignored=ignored,identical_teacher=identical,relative_l2=errors))
        # Tied input/output parameter must receive both gradient paths.
        w=torch.randn(67,32,device=device,requires_grad=True)*.1
        ids=torch.arange(19,device=device); h=F.embedding(ids,w)
        labels=(ids+1)%67; th=torch.randn_like(h); tw=torch.randn_like(w)*.1
        ls=F.log_softmax(h@w.T,-1); lt=F.log_softmax(th@tw.T,-1)
        ref=.75*F.nll_loss(ls,labels,reduction='sum')+.25*(lt.exp()*(lt-ls)).sum()
        actual=PrecomputedKD.apply(h,w,th,tw,labels,.25,7)[0]
        a=torch.autograd.grad(ref,w,retain_graph=True)[0]; b=torch.autograd.grad(actual,w)[0]
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)
    return records


def main():
    torch.set_num_threads(1)
    accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1)
    assert accelerator.num_processes==8
    set_seed(42)
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=True)
    cfg=read_json(HERE.parent/'configs/K1-seed42.json')
    checks=small_checks(accelerator.device)
    if accelerator.is_main_process:atomic_json(out/'small-checks.json',checks)
    accelerator.wait_for_everyone()
    prep=DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008'
    # Only reduce evaluation metadata loading/reserve; benchmark never evaluates.
    data_cfg=dict(cfg,eval_documents=1,final_evaluation_reserve_seconds=0,stop_margin_seconds=90)
    dataset,_,provenance=train.prepare_data(prep,out,data_cfg,accelerator)
    set_seed(42)
    model,initialization=train.build_model(cfg,out,provenance,accelerator)
    parallel_losses.attach_loss(model,cfg,accelerator)
    # The unrelated CCE branch leaves an empty closure cell; inspect.getclosurevars
    # visits every cell and fails. Read only the known teacher cell.
    forward_function=model.forward.__func__
    teacher=forward_function.__closure__[forward_function.__code__.co_freevars.index('teacher')].cell_contents
    optimizers,groups=parallel_runtime.build_optimizers(model,cfg)
    for opt in optimizers:
        for group in opt.param_groups:
            group['lr']=0.; group['decay_lr']=0.
    model=accelerator.prepare(model)
    model.train(); train.MASK_MODE='intradoc'
    batches=[]
    for micro in range(2):
        row_index=3051*16+micro*8+accelerator.process_index
        batches.append({k:v.to(accelerator.device) for k,v in train.collate_rows([dataset[row_index]],8192).items()})
    local_valid=sum((b['labels'][:,1:]!=-100).sum() for b in batches)
    global_valid=accelerator.reduce(local_valid,reduction='sum')
    step_inputs=int(accelerator.reduce(sum(b['attention_mask'].sum() for b in batches),reduction='sum'))
    old_loss=parallel_losses.kd_loss
    variant={'fn':old_loss,'chunk':128}
    events=[]; forward_phase=False
    def start_hook(module,args):
        if forward_phase:
            event=torch.cuda.Event(enable_timing=True);event.record()
            module._benchmark_event=event
    def end_hook(name):
        def hook(module,args,result):
            if forward_phase:
                end=torch.cuda.Event(enable_timing=True);end.record()
                events.append((name,module._benchmark_event,end))
        return hook
    unwrapped=accelerator.unwrap_model(model)
    for name,module in [('student_forward',unwrapped.model),('teacher_forward',teacher.model)]:
        module.register_forward_pre_hook(start_hook); module.register_forward_hook(end_hook(name))
    def selected(*args,**kwargs):
        a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True);a.record()
        result=variant['fn'](*args,lam=.25,chunk=variant['chunk'])
        b.record();events.append(('head_forward_including_pregrad',a,b));return result
    parallel_losses.kd_loss=selected
    total_inputs=0; rows=[]
    def step(name, do_optimizer=True):
        nonlocal total_inputs,forward_phase
        if time.time()>int(os.environ['TRAIN_END_EPOCH'])-90:raise RuntimeError('Stopping margin reached')
        if total_inputs+step_inputs>20_000_000:raise RuntimeError('Benchmark token cap reached')
        total_inputs+=step_inputs
        if accelerator.is_main_process:atomic_json(out/'input-accounting.json',{'student_inputs_reserved':total_inputs,'teacher_inputs_reserved':total_inputs,'includes_repeated_validation_windows':True})
        for opt in optimizers:opt.zero_grad(set_to_none=True)
        events.clear(); parallel_losses._STATS=None
        accelerator.wait_for_everyone();torch.cuda.synchronize(); begin=time.perf_counter()
        losses=[]
        for i,batch in enumerate(batches):
            with accelerator.no_sync(model) if i==0 else contextlib.nullcontext():
                forward_phase=True
                with accelerator.autocast(): loss=train.causal_loss_sum(model,batch)
                forward_phase=False
                a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
                accelerator.backward(loss*(8/global_valid))
                b.record();events.append(('backward',a,b));losses.append(loss.detach())
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        if do_optimizer:
            accelerator.clip_grad_norm_(model.parameters(),.5)
            for opt in optimizers:opt.step()
        b.record();events.append(('clip_optimizer',a,b))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-begin
        names=['student_forward','teacher_forward','head_forward_including_pregrad','backward','clip_optimizer']
        times=torch.tensor([elapsed]+[sum(x.elapsed_time(y)/1000 for n,x,y in events if n==k) for k in names],device=accelerator.device,dtype=torch.float64)
        dist.all_reduce(times,op=dist.ReduceOp.MAX)
        objective=accelerator.reduce(torch.stack(losses).sum(),reduction='sum')/global_valid
        return dict(variant=name,seconds=dict(zip(['total']+names,times.tolist())),loss=float(objective),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
    # Entire DDP gradient, including the tied embedding's two paths, same fixed weights/data.
    variant.update(fn=old_loss,chunk=128)
    reference=step('gradient_reference',False)
    grads=[p.grad.detach().clone() for p in model.parameters()]
    variant.update(fn=fast_loss,chunk=1024)
    candidate=step('gradient_candidate',False)
    measures=torch.stack([torch.stack(((p.grad-g).square().sum(),g.square().sum(),p.grad.square().sum(),(p.grad*g).sum())) for p,g in zip(model.parameters(),grads)]).sum(0)
    err=float((measures[0]/measures[1]).sqrt()); cosine=float(measures[3]/(measures[1]*measures[2]).sqrt())
    assert err<.02 and cosine>.999, (err,cosine)
    assert abs(candidate['loss']-reference['loss'])<1e-4
    if accelerator.is_main_process:atomic_json(out/'full-gradient-check.json',dict(relative_l2=err,cosine=cosine,reference=reference,candidate=candidate,all_ranks_synchronized=True))
    del grads
    variants=[('recompute128',old_loss,128),('recompute1024',old_loss,1024),('pregrad128',fast_loss,128),('pregrad1024',fast_loss,1024)]
    # Warm up every shape and optimizer, then reverse order on the second block.
    for name,fn,chunk in variants:
        variant.update(fn=fn,chunk=chunk)
        for _ in range(2):step(name)
    for block in range(2):
        for name,fn,chunk in variants[::1 if block==0 else -1]:
            variant.update(fn=fn,chunk=chunk);torch.cuda.reset_peak_memory_stats()
            for repeat in range(5):
                record=step(name);record.update(block=block,repeat=repeat);rows.append(record)
                if accelerator.is_main_process:
                    with (out/'timings.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
    summary={name:{key:statistics.median(r['seconds'][key] for r in rows if r['variant']==name) for key in rows[0]['seconds']} for name,_,_ in variants}
    if accelerator.is_main_process:
        atomic_json(out/'results.json',dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),world_size=8,seed=42,config=cfg,
            token_inputs=total_inputs,optimizer_updates_to_weights=0,global_batch=16,gradient_check_relative_l2=err,gradient_check_cosine=cosine,
            median_seconds=summary,peak_allocated={name:max(r['peak_allocated'] for r in rows if r['variant']==name) for name,_,_ in variants},
            limitations='Fresh random student, fixed repeated canonical window, LR=0; no PPL/convergence evidence. Component times are per-component rank maxima, so may not sum to total.',
            versions=dict(torch=torch.__version__),complete=True))
        print(json.dumps(summary),flush=True)
    accelerator.wait_for_everyone()

if __name__=='__main__':main()
