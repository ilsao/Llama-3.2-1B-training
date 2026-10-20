"""One-GPU eager NorMuon full-model check with durable stage/error diagnostics."""
import argparse
import copy
import math
import os
from pathlib import Path
import signal
import time
import traceback


def indexed_device(device,current_index):
    return type(device)(device.type,current_index) if device.type=='cuda' and device.index is None else device


def compare(a,b):
    import torch
    if torch.is_tensor(a):
        # torch Optimizer.load_state_dict may relocate scalar lr_ratio from CPU
        # to the parameter device; preserve values/dtype exactly.
        if a.device!=b.device:
            if a.ndim!=0:raise AssertionError('Unexpected nonscalar state relocation')
            b=b.to(a.device)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('State keys differ')
        for key in a:compare(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('State lengths differ')
        for x,y in zip(a,b):compare(x,y)
    elif a!=b:raise AssertionError((a,b))


def main():
    from train import require_compute
    require_compute()
    import torch
    import pyarrow as pa
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from training_state import DEFAULT_ROOT,TokenJournal,load_config,read_json,atomic_json,file_digest,digest
    from batch1b_contracts import STUDY
    from batch1b_runtime import build_optimizers,TokenScheduler,configure_update
    from train import prepare_data,build_model,collate_rows,causal_loss_sum,save_checkpoint,restore_rng,evaluate
    from recipe_runtime import EMA,clip
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    cfg=load_config(parser.parse_args().config)
    accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1)
    accelerator.state.device=indexed_device(accelerator.device,torch.cuda.current_device())
    if accelerator.num_processes!=1 or 'H200' not in torch.cuda.get_device_name(accelerator.device):
        raise ValueError('This check requires exactly one H200')
    if os.environ.get('TORCH_COMPILE_DISABLE')!='1' or not torch._dynamo.config.disable:
        raise ValueError('Eager execution must be explicit')
    def alarm(*_):raise TimeoutError('Validation stopping margin reached')
    signal.signal(signal.SIGALRM,alarm)
    signal.alarm(max(1,int(os.environ['TRAIN_END_EPOCH'])-int(time.time())-120))
    live=Path(__file__).resolve().parent.parent
    run=DEFAULT_ROOT/'runs/experiments'/(STUDY+'-normuon-check-r4')
    run.mkdir(parents=True,exist_ok=True)
    result=dict(job_id=os.environ['SLURM_JOB_ID'],normuon_pass=False,checks=[],execution='eager',
                source_root=str(Path(__file__).resolve().parent),stage='start')
    def note(stage):
        result['stage']=stage;result['timestamp']=time.time()
        atomic_json(live/'normuon-progress-r4.json',result)
        print('NORMUON_CHECK',stage,time.time(),flush=True)
    try:
        note('data')
        set_seed(42)
        train,evaluation,provenance=prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',run,cfg,accelerator)
        note('random_model')
        model,initialization=build_model(cfg,run,provenance,accelerator)
        result['initialization_signature']=initialization['sample_signature']
        result['provenance_digest']=digest(provenance)
        # Construct on CPU, matching real training, then move via prepare().
        raw,groups=build_optimizers(model,cfg)
        schedulers=[TokenScheduler(o,cfg) for o in raw]
        model,*optimizers=accelerator.prepare(model,*raw)
        ema=EMA(accelerator.unwrap_model(model),cfg)
        accelerator.register_for_checkpointing(*schedulers,ema)
        journal=TokenJournal(run/'token-journal.json',cfg['max_training_tokens'])
        cursor=0
        updates=[]
        def update(capture=False):
            nonlocal cursor
            note('prepare_update_'+str(len(updates)+1))
            local=[collate_rows([train[cursor+i]],8192) for i in range(2)]
            inputs=sum(int(b['attention_mask'].sum()) for b in local)
            targets=sum(int((b['labels'][:,1:]!=-100).sum()) for b in local)
            if inputs!=16384:raise ValueError('Unexpected packed input mask')
            transaction=journal.reserve(inputs)
            configure_update(optimizers,cfg,2,journal.data['confirmed_tokens'])
            for opt in optimizers:opt.zero_grad(set_to_none=True)
            for batch in local:
                batch={k:v.to(accelerator.device) for k,v in batch.items()}
                with accelerator.autocast():
                    loss=causal_loss_sum(model,batch)
                    accelerator.backward(loss/targets)
            norm,_=clip(model,optimizers,cfg,journal.data['confirmed_tokens'],accelerator)
            if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite full-model gradient')
            gradients={n:p.grad.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()
                       if p.grad is not None} if capture else None
            note('optimizer_update_'+str(len(updates)+1))
            torch.cuda.synchronize();start=time.monotonic()
            for opt in optimizers:opt.step()
            torch.cuda.synchronize();duration=time.monotonic()-start
            journal.confirm(transaction);cursor+=2
            updates.append(dict(inputs=inputs,optimizer_seconds=duration,grad_norm=float(norm)))
            for scheduler in schedulers:scheduler.set_tokens(journal.data['confirmed_tokens'])
            result['updates']=updates
            return gradients
        for _ in range(3):update()
        note('checkpoint')
        train.commit_receipt(cursor)
        state=dict(global_step=3,cursor=cursor,confirmed_tokens_at_checkpoint=49152,charged_tokens_at_checkpoint=49152)
        checkpoint=save_checkpoint(accelerator,run,state,retain_all=True)
        weights_before={n:p.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()}
        states_before=[copy.deepcopy(o.state_dict()) for o in optimizers]
        rng_expected=torch.randn(8,device=accelerator.device)
        restore_rng(checkpoint,accelerator)
        compare(rng_expected,torch.randn(8,device=accelerator.device))
        gradients=update(capture=True)
        weights_after={n:p.detach().clone() for n,p in accelerator.unwrap_model(model).named_parameters()}
        states_after=[copy.deepcopy(o.state_dict()) for o in optimizers]
        note('restore')
        accelerator.load_state(checkpoint);restore_rng(checkpoint,accelerator)
        for name,p in accelerator.unwrap_model(model).named_parameters():compare(weights_before[name],p)
        for before,opt in zip(states_before,optimizers):compare(before,opt.state_dict())
        del weights_before,states_before
        note('identical_gradient_replay')
        configure_update(optimizers,cfg,2,49152)
        for opt in optimizers:opt.zero_grad(set_to_none=True)
        for name,p in accelerator.unwrap_model(model).named_parameters():p.grad=gradients.get(name)
        for opt in optimizers:opt.step()
        torch.cuda.synchronize()
        for name,p in accelerator.unwrap_model(model).named_parameters():compare(weights_after[name],p)
        for after,opt in zip(states_after,optimizers):compare(after,opt.state_dict())
        del weights_after,states_after,gradients
        if not all('momentum_buffer' in s and 'variance_neuron' in s for s in optimizers[1].state.values()):
            raise ValueError('NorMuon momentum/variance state missing')
        unique_graphs=int(torch._dynamo.utils.counters['stats']['unique_graphs'])
        if unique_graphs:raise ValueError('Unexpected torch.compile graph in eager validation')
        note('evaluation')
        metrics=evaluate(model,evaluation,cfg,accelerator)
        result.update(normuon_pass=True,checks=['CPU-first construction and GPU foreach dispatch',
            'four finite full-model NorMuon+AdamW+PE updates','all optimizer moments/variance retained',
            'bit-exact weight/state restore','bit-exact identical-gradient next update','RNG restore',
            'finite heldout NLL','zero Dynamo compiled graphs'],
            configuration_digest=digest(cfg),optimizer_groups=groups,evaluation=metrics,
            processed_training_inputs=journal.data['confirmed_tokens'],charged_training_inputs=journal.data['charged_tokens'],
            unique_compiled_graphs=unique_graphs,checkpoint=checkpoint)
        note('passed')
        atomic_json(live/'normuon-result-r4.json',result)
    except BaseException as error:
        result.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        atomic_json(live/'normuon-result-r4.json',result)
        atomic_json(Path(os.environ['JOB_RECORD'])/'worker-rank-0-error.json',result)
        print(result['traceback'],flush=True)
        raise
    finally:
        signal.alarm(0)
        accelerator.end_training()


if __name__=='__main__':main()
