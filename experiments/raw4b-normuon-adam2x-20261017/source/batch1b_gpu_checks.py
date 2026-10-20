"""Heavy validation: run only through the registered eight-H200 Slurm job."""
import argparse
import contextlib
import copy
import datetime
import math
import os
from pathlib import Path
import time


def compare_tree(a, b, exact=False):
    import torch
    if torch.is_tensor(a):
        torch.testing.assert_close(a, b, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-7)
    elif isinstance(a, dict):
        if a.keys() != b.keys():raise AssertionError('Checkpoint dictionary keys changed')
        for key in a:compare_tree(a[key], b[key],exact)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):raise AssertionError('Checkpoint list length changed')
        for x,y in zip(a,b):compare_tree(x,y,exact)
    else:
        if a != b:raise AssertionError((a,b))


def check_beta_math(device):
    import torch
    from batch1b_runtime import adamw_class
    cls=adamw_class()
    initial=torch.tensor([.2,-.7,1.3],dtype=torch.float64,device=device)
    p=torch.nn.Parameter(initial.clone());q=torch.nn.Parameter(initial.clone())
    opt=cls([p],lr=.03,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=False)
    ref=torch.optim.AdamW([q],lr=.03,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=False)
    m=torch.zeros_like(p);v=torch.zeros_like(p);expected=initial.clone();product=1.
    for i,beta in enumerate([.95,.95,.95**4,.95**2],1):
        grad=torch.tensor([i*.11,-.23,.31],device=device,dtype=torch.float64)
        p.grad=grad.clone();q.grad=grad.clone()
        opt.param_groups[0]['betas']=(.9,beta)
        opt.param_groups[0]['variable_beta2']=i>2
        opt.step()
        if i<=2:
            ref.step();torch.testing.assert_close(p,q,rtol=0,atol=1e-15)
        product*=beta;m=m*.9+grad*.1;v=v*beta+grad.square()*(1-beta)
        expected=expected*(1-.03*.1)-.03*(m/(1-.9**i))/(torch.sqrt(v/(1-product))+1e-8)
        torch.testing.assert_close(p,expected,rtol=0,atol=1e-14)
        if not math.isclose(opt.state[p]['beta2_product'],product,rel_tol=1e-14):
            raise AssertionError('Incorrect historical beta2 product')
    saved=copy.deepcopy(opt.state_dict());r=torch.nn.Parameter(p.detach().clone())
    other=cls([r],lr=.03,fused=False);other.load_state_dict(saved)
    compare_tree(saved,other.state_dict())


def check_gradient_normalization(accelerator):
    import torch
    from torch.nn.parallel import DistributedDataParallel
    # Deliberately unequal valid targets across ranks and accumulation windows.
    layer=torch.nn.Linear(3,2,bias=False,device=accelerator.device,dtype=torch.float64)
    with torch.no_grad():layer.weight.fill_(.3)
    model=DistributedDataParallel(layer,device_ids=[accelerator.local_process_index])
    values=torch.arange(64*3,device=accelerator.device,dtype=torch.float64).reshape(64,3)/100
    labels=torch.arange(64*2,device=accelerator.device,dtype=torch.float64).reshape(64,2)/300
    masks=(torch.arange(64,device=accelerator.device)%3!=0).double()
    total=masks.sum()*2
    grads=[]
    for microbatch,accumulation in [(1,8),(4,2)]:
        model.zero_grad(set_to_none=True)
        for micro in range(accumulation):
            first=micro*8*microbatch+accelerator.process_index*microbatch
            ix=slice(first,first+microbatch)
            context=contextlib.nullcontext() if micro==accumulation-1 else model.no_sync()
            with context:
                loss=((model(values[ix])-labels[ix]).square()*masks[ix,None]).sum()
                (loss*8/total).backward()
        grads.append(layer.weight.grad.detach().clone())
    expected=torch.nn.Linear(3,2,bias=False,device=accelerator.device,dtype=torch.float64)
    with torch.no_grad():expected.weight.fill_(.3)
    (((expected(values)-labels).square()*masks[:,None]).sum()/total).backward()
    for grad in grads:torch.testing.assert_close(grad,expected.weight.grad,rtol=1e-12,atol=1e-12)


def main():
    from train import require_compute
    require_compute()
    import torch
    import pyarrow as pa
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs,set_seed
    from training_state import DEFAULT_ROOT,TokenJournal,atomic_json,load_config,read_json,digest,window_indices
    from batch1b_contracts import STUDY,Q
    from batch1b_runtime import build_optimizers,TokenScheduler,configure_update,verify_checkpoint_files
    from recipe_runtime import EMA,clip
    from train import (on_main,prepare_data,build_model,collate_rows,causal_loss_sum,
                       save_checkpoint,restore_rng,evaluate,reduce_max)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    cfg=load_config(parser.parse_args().config)
    accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1,
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False),
                         InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=10))])
    if accelerator.num_processes!=8 or 'H200' not in torch.cuda.get_device_name(accelerator.device):
        raise ValueError('Validation requires eight H200s')
    started=time.monotonic()
    result=dict(job_id=os.environ['SLURM_JOB_ID'],native_pass=False,normuon_pass=False,
                checks=[],processed_training_inputs=0)
    print('VALIDATION beta2 and gradient contracts',flush=True)
    check_beta_math(accelerator.device)
    check_gradient_normalization(accelerator)
    result['checks']+=['fixed-beta native equivalence','historical-product variable-beta recurrence',
                       'variable-beta optimizer state roundtrip','token-weighted unequal-mask DDP accumulation equivalence']
    run=DEFAULT_ROOT/'runs/experiments'/(STUDY+'-validation-r1')
    on_main(accelerator,lambda:run.mkdir(parents=True,exist_ok=True))
    set_seed(42)
    train,evaluation,provenance=prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',run,cfg,accelerator)
    journal=TokenJournal(run/'token-journal.json',cfg['max_training_tokens']) if accelerator.is_main_process else None
    model,initialization=build_model(cfg,run,provenance,accelerator)
    raw,groups=build_optimizers(model,cfg)
    schedulers=[TokenScheduler(o,cfg) for o in raw]
    model,*optimizers=accelerator.prepare(model,*raw)
    ema=EMA(accelerator.unwrap_model(model),cfg)
    accelerator.register_for_checkpointing(*schedulers,ema)
    result['initialization_signature']=initialization['sample_signature']
    result['provenance_digest']=digest(provenance)
    result['checks']+=['random residual-half initialization','canonical split and immutable raw prefix','parameter-group unique coverage']

    def update(cursor,width,configuration,opts,capture_gradients=False):
        batches=window_indices(cursor,len(train),8,accelerator.process_index,1,width//8)
        local=[collate_rows([train[i] for i in indices],8192) for indices in batches]
        counts=torch.tensor([sum(int(b['attention_mask'].sum()) for b in local),
                             sum(int((b['labels'][:,1:]!=-100).sum()) for b in local)],device=accelerator.device)
        inputs,targets=accelerator.reduce(counts,reduction='sum').tolist()
        if inputs!=width*8192:raise ValueError('Dense prefix accounting mismatch')
        transaction=on_main(accelerator,lambda:journal.reserve(inputs))
        configure_update(opts,configuration,width,cursor*8192)
        for opt in opts:opt.zero_grad(set_to_none=True)
        for micro,batch in enumerate(local):
            batch={k:v.to(accelerator.device) for k,v in batch.items()}
            context=contextlib.nullcontext() if micro==len(local)-1 else accelerator.no_sync(model)
            with context,accelerator.autocast():
                loss=causal_loss_sum(model,batch)
                accelerator.backward(loss*(8/targets))
        norm,_=clip(model,opts,configuration,cursor*8192,accelerator)
        if not bool(torch.isfinite(norm)):raise ValueError('Nonfinite validation gradient')
        gradients={n:p.grad.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()
                   if p.grad is not None} if capture_gradients else None
        for opt in opts:opt.step()
        torch.cuda.synchronize()
        on_main(accelerator,lambda:journal.confirm(transaction))
        result['processed_training_inputs']+=inputs
        return (cursor+width,gradients) if capture_gradients else cursor+width

    print('VALIDATION native full-model checkpoint roundtrip',flush=True)
    cursor=update(0,16,cfg,optimizers)
    cursor=update(cursor,16,cfg,optimizers)
    for s in schedulers:s.set_tokens(2*Q)
    state=dict(global_step=2,cursor=cursor,confirmed_tokens_at_checkpoint=2*Q,charged_tokens_at_checkpoint=2*Q)
    checkpoint=save_checkpoint(accelerator,run,state,retain_all=True)
    on_main(accelerator,lambda:verify_checkpoint_files(checkpoint,read_json(Path(checkpoint)/'complete.json')))
    checkpoint_weights={n:p.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()}
    checkpoint_optimizers=[copy.deepcopy(o.state_dict()) for o in optimizers]
    expected_rng=torch.randn(8,device=accelerator.device)
    restore_rng(checkpoint,accelerator)
    torch.testing.assert_close(torch.randn(8,device=accelerator.device),expected_rng,rtol=0,atol=0)
    switched=copy.deepcopy(cfg);switched.update(batch_switch_tokens=0,adamw_beta2_policy='token_scaled')
    _,golden_gradients=update(cursor,64,switched,optimizers,capture_gradients=True)
    expected_weights={n:p.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()}
    expected_optimizers=[copy.deepcopy(o.state_dict()) for o in optimizers]
    accelerator.load_state(checkpoint)
    restore_rng(checkpoint,accelerator)
    if any(s.tokens!=2*Q for s in schedulers):raise ValueError('Scheduler state did not restore')
    for name,p in accelerator.unwrap_model(model).named_parameters():
        torch.testing.assert_close(p,checkpoint_weights[name],rtol=0,atol=0)
    for old,opt in zip(checkpoint_optimizers,optimizers):compare_tree(old,opt.state_dict(),exact=True)
    del checkpoint_weights,checkpoint_optimizers
    # Test the next update using the SAME clipped gradients. Re-running BF16
    # attention/backward would mix checkpoint correctness with kernel noise.
    configure_update(optimizers,switched,64,cursor*8192)
    for opt in optimizers:opt.zero_grad(set_to_none=True)
    for name,p in accelerator.unwrap_model(model).named_parameters():
        p.grad=golden_gradients.get(name)
    for opt in optimizers:opt.step()
    torch.cuda.synchronize()
    for name,p in accelerator.unwrap_model(model).named_parameters():
        torch.testing.assert_close(p,expected_weights[name],rtol=0,atol=0)
    for old,opt in zip(expected_optimizers,optimizers):compare_tree(old,opt.state_dict(),exact=True)
    del expected_weights,expected_optimizers,golden_gradients
    result['checks']+=['full model/two-optimizer/scheduler checkpoint roundtrip','all-rank RNG restoration',
                       'bit-exact full weights and optimizer state restoration',
                       'bit-exact BS64 update replay with identical gradients','optimizer moments/step/beta2 product retained']
    metrics=evaluate(model,evaluation,cfg,accelerator)
    if metrics['eval/documents']!=1000 or metrics['eval/prediction_tokens']!=1479302:
        raise ValueError('Fixed evaluation identity differs')
    result['checks']+=['fixed1000-document evaluation tokens and finite NLL']
    result['native_pass']=True
    on_main(accelerator,lambda:atomic_json(Path(__file__).resolve().parent.parent/'validation-progress-r1.json',result))
    print('VALIDATION NorMuon+Polar Express full model',flush=True)
    variant=copy.deepcopy(cfg)
    variant['matrix_optimizer']=dict(algorithm='normuon',muon_beta2=.95,orthogonalization_strategy='polar_express',
                                    use_cautious_wd=False,package='muonium',version='0.2.0')
    error=None
    try:
        normuon_raw,normuon_groups=build_optimizers(accelerator.unwrap_model(model),variant)
        for group in normuon_raw[1].param_groups:
            if group['name']!='muon' or group['weight_decay']!=.1 or not group['foreach']:
                raise ValueError('NorMuon lost required metadata or driver')
        package_schedulers=[TokenScheduler(o,variant) for o in normuon_raw]
        for scheduler in package_schedulers:scheduler.set_tokens(170*Q)
        import json
        json.dumps({g['name']:float(g['lr']) for o in normuon_raw for g in o.param_groups},allow_nan=False)
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    failed=int(reduce_max(torch.tensor(int(error is not None),device=accelerator.device),accelerator).item())
    if failed:
        result['normuon_error']=error or 'NorMuon construction failed on another rank'
    else:
        try:
            update(96,16,variant,normuon_raw)
            saved=[copy.deepcopy(o.state_dict()) for o in normuon_raw]
            package_checkpoint=run/'normuon-optimizers.pt'
            on_main(accelerator,lambda:torch.save(saved,package_checkpoint))
            loaded=torch.load(package_checkpoint,map_location=accelerator.device,weights_only=False)
            for opt,old in zip(normuon_raw,loaded):
                opt.load_state_dict(old);compare_tree(old,opt.state_dict())
            matrix_states=normuon_raw[1].state.values()
            if not all('momentum_buffer' in s and 'variance_neuron' in s for s in matrix_states):
                raise ValueError('NorMuon state incomplete')
            result['normuon_pass']=True
            result['normuon_groups']=normuon_groups
            result['checks']+=['NorMuon+Polar Express full-model update','NorMuon momentum/variance optimizer file roundtrip']
        except Exception as exc:
            result['normuon_error']=type(exc).__name__+': '+str(exc)
    passed=int(1-reduce_max(torch.tensor(int(not result['normuon_pass']),device=accelerator.device),accelerator).item())
    result['normuon_pass']=bool(passed)
    result['seconds']=time.monotonic()-started
    result['checkpoint_comparison_tolerance']=dict(rtol=0,atol=0)
    result['replayed_gradients_do_not_process_inputs']=True
    on_main(accelerator,lambda:atomic_json(Path(__file__).resolve().parent.parent/'validation-result-r1.json',result))
    print('VALIDATION RESULT',result['native_pass'],result['normuon_pass'],flush=True)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__=='__main__':main()
