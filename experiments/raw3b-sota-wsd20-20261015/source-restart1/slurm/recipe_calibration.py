"""H200-only full-model correctness and matched-batch timing validation."""
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    from accelerate import Accelerator
    from accelerate.utils import set_seed,broadcast_object_list,gather_object
    from train import require_compute,prepare_data,build_model,build_optimizers,collate_rows,causal_loss_sum,evaluate,on_main
    from training_state import DEFAULT_ROOT,read_json,atomic_json,TokenJournal,window_indices
    from recipe_runtime import INITIALIZERS,EMA,clip
    require_compute();a=Accelerator(mixed_precision='bf16')
    if a.num_processes!=8 or 'H200' not in torch.cuda.get_device_name(a.device):raise ValueError('Require8 H200 ranks')
    cfg=read_json(sys.argv[1]);run=DEFAULT_ROOT/'runs/experiments'/sys.argv[2]
    on_main(a,lambda:run.mkdir(parents=True,exist_ok=False))
    models={};max_init=0
    raw,ev,prov=prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',run/'raw',cfg,a)
    filtered,fev,fprov=prepare_data(DEFAULT_ROOT/'preprocessed/dolma-dclm10-005',run/'filtered',cfg,a)
    if prov['dataset_revision']!=fprov['dataset_revision'] or prov['evaluation_indices']!=fprov['evaluation_indices']:raise ValueError('Canonical evaluation membership differs')
    def evaluation_signature(rows):
        h=hashlib.sha256();tokens=0
        for row in rows:
            h.update(json.dumps({k:row[k] for k in ('input_ids','attention_mask','labels','evaluation_slot')},sort_keys=True,separators=(',',':')).encode())
            tokens+=sum(x!=-100 for x in row['labels'][1:])
        return {'sha256':h.hexdigest(),'prediction_tokens':tokens,'rows':len(rows)}
    signatures=on_main(a,lambda:[evaluation_signature(ev),evaluation_signature(fev)])
    if signatures[0]!=signatures[1]:raise ValueError('Raw/filtered evaluation tokens or masks differ')
    for mode in INITIALIZERS:
        set_seed(42);started=time.monotonic()
        model,metadata=build_model({**cfg,'initialization':mode},run/('init-'+mode),prov,a)
        signatures_by_rank=gather_object([metadata['sample_signature']])
        # Independently digest each rank's actual model samples, not the broadcast record.
        from training_state import digest
        actual=digest({n:p.detach().flatten()[:8].tolist() for n,p in model.named_parameters()})
        actuals=gather_object([actual])
        if len(set(actuals))!=1:raise ValueError('Rank initialization differs')
        if mode=='gaussian':
            base_sample=actual;baseline_residual=metadata['recipe']['statistics']
            baseline=read_json(DEFAULT_ROOT/'runs/experiments/horizon-wsd-20261014-final-standard/initialization.json')
            if actual!=baseline['sample_signature']:raise ValueError('Default initialization changed winning baseline')
        if mode in ('residual_half','residual_depth'):
            ratio=.5 if mode=='residual_half' else 1/(32**.5)
            for n,x in metadata['recipe']['statistics'].items():
                expected=ratio if n.endswith(('o_proj.weight','down_proj.weight')) else 1
                if abs(x['std']/baseline_residual[n]['std']-expected)>1e-5:raise ValueError('Residual initializer variance mismatch')
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:raise ValueError('Lost tying')
        _,groups=build_optimizers(model,cfg)
        if sum(groups['parameter_counts'].values())!=1235814400:raise ValueError('Optimizer coverage mismatch')
        max_init=max(max_init,time.monotonic()-started)
        models[mode]=metadata
        del model;gc.collect();torch.cuda.empty_cache()
    journal=None
    on_main(a,lambda:atomic_json(run/'resolved-config.json',cfg))
    def journal_open():
        nonlocal journal
        journal=TokenJournal(run/'token-journal.json',cfg['max_training_tokens'])
    on_main(a,journal_open)
    tracker=None
    def tracking():
        nonlocal tracker
        import wandb
        tracker=wandb.init(project='lab5-training-llama',name=sys.argv[2],dir=str(run),config=cfg)
        wandb.define_metric('train/global_step');wandb.define_metric('train/*',step_metric='train/global_step')
        return tracker.url
    wandb_url=on_main(a,tracking)
    measurements={};global_step=0
    for label,accum,data,ema_betas in [('batch8',1,raw,[]),('batch16',2,raw,[]),('batch32',4,raw,[]),('ema',2,raw,[.99,.999,.9995]),('filtered',2,filtered,[])]:
        localcfg={**cfg,'gradient_accumulation_steps':accum,'ema':ema_betas}
        set_seed(42);started=time.monotonic()
        model,_=build_model(localcfg,run/('timing-'+label),prov if label!='filtered' else fprov,a)
        opts,_=build_optimizers(model,localcfg)
        model,*opts=a.prepare(model,*opts)
        ema=EMA(a.unwrap_model(model),{**localcfg,'target_training_tokens':1})
        initialization=time.monotonic()-started;max_init=max(max_init,initialization)
        times=[];step_tokens=8*accum*8192
        for step in range(25):
            torch.cuda.synchronize();start=time.monotonic()
            ids=window_indices(step*8*accum,len(data),8,a.process_index,1,accum)
            batches=[collate_rows([data[i] for i in indices],8192) for indices in ids]
            local_counts=[sum(int(b['attention_mask'].sum()) for b in batches),sum(int((b['labels'][:,1:]!=-100).sum()) for b in batches)]
            counts=a.reduce(torch.tensor(local_counts,device=a.device),reduction='sum').tolist()
            if counts[0]!=step_tokens:raise ValueError('Nonpadding input token count mismatch')
            tx=on_main(a,lambda:journal.reserve(counts[0]))
            for o in opts:o.zero_grad(set_to_none=True)
            local_nll=torch.zeros((),device=a.device,dtype=torch.float64)
            for i,b in enumerate(batches):
                b={k:v.to(a.device) for k,v in b.items()}
                sync=contextlib.nullcontext() if i==accum-1 else a.no_sync(model)
                with sync,a.autocast():loss=causal_loss_sum(model,b)
                local_nll+=loss.detach().double()
                a.backward(loss*(8/counts[1]))
            norm,_=clip(model,opts,localcfg,0,a)
            if not torch.isfinite(norm):raise ValueError('Nonfinite calibration gradient')
            for o in opts:o.step()
            ema.update(step_tokens*(step+1))
            on_main(a,lambda:journal.confirm(tx))
            torch.cuda.synchronize();elapsed=time.monotonic()-start
            t=torch.tensor(elapsed,device=a.device,dtype=torch.float64)
            torch.distributed.all_reduce(t,op=torch.distributed.ReduceOp.MAX)
            global_step+=1
            total_nll=float(a.reduce(local_nll,reduction='sum'))
            confirmed=on_main(a,lambda:journal.data['confirmed_tokens'])
            if a.is_main_process:
                tracker.log({'train/global_step':global_step,'train/loss':total_nll/counts[1],'train/grad_norm':float(norm),'train/learning_rate':localcfg['learning_rate'],'train/tokens_per_second':counts[0]/float(t),'train/total_tokens_seen':confirmed,'calibration/case':label})
            if step>=5:times.append(float(t))
            if (step+1)%10==0:
                interim=evaluate(model,ev,localcfg,a)
                if a.is_main_process:tracker.log({'train/global_step':global_step,**interim})
        for o in opts:o.zero_grad(set_to_none=True)
        evaluation=evaluate(model,ev,localcfg,a)
        if evaluation['eval/prediction_tokens']!=signatures[0]['prediction_tokens']:raise ValueError('Evaluation prediction count mismatch')
        if a.is_main_process:tracker.log({'train/global_step':global_step,**evaluation})
        # Exercise all clipping modes with known synchronized gradients on a small model.
        tiny=torch.nn.Linear(2,1,bias=False).to(a.device)
        for mode in ('global','late_one','per_optimizer'):
            tiny.weight.grad=torch.tensor([[3.,4.]],device=a.device)
            o=torch.optim.SGD([{'params':list(tiny.parameters()),'name':'test'}],lr=.01)
            n,_=clip(tiny,[o],{**cfg,'clipping':mode},cfg['target_training_tokens'],a)
            limit=1 if mode=='late_one' else .5
            if abs(float(n)-5)>1e-5 or abs(float(tiny.weight.grad.norm())-limit)>1e-5:raise ValueError('Clipping semantics mismatch')
        if ema_betas:
            before=next(a.unwrap_model(model).parameters()).detach().clone()
            for beta in ema_betas:
                with ema.weights(beta):pass
            if not torch.equal(before,next(a.unwrap_model(model).parameters())):raise ValueError('EMA swap failed to restore raw weights')
            # Tiny shadow checkpoint round trip without writing full models.
            test=EMA(tiny,{**cfg,'ema':[.99],'target_training_tokens':1});test.update(1)
            a.register_for_checkpointing(test)
            path=run/'validation-resume'
            first=next(a.unwrap_model(model).parameters())
            expected=first.detach().flatten()[:8].clone();rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(a.device)
            old_lr=opts[0].param_groups[0]['lr']
            a.save_state(str(path))
            a.wait_for_everyone()  # No rank loads until main-rank files are durable.
            with torch.no_grad():first.flatten()[:8].add_(9)
            opts[0].param_groups[0]['lr']=99;test.updates=99
            torch.rand(8);torch.rand(8,device=a.device)
            a.load_state(str(path))
            if not torch.equal(first.detach().flatten()[:8],expected) or not torch.equal(torch.get_rng_state(),rng) or not torch.equal(torch.cuda.get_rng_state(a.device),cuda_rng) or opts[0].param_groups[0]['lr']!=old_lr or test.updates!=1:raise ValueError('Model/optimizer/RNG/EMA state round trip failed')
            a._custom_objects.remove(test)
        measurements[label]={'tokens_per_second':step_tokens/(sum(times)/len(times)),'initialization_seconds':initialization,'mean_complete_update_seconds':sum(times)/len(times)}
        del model,opts,ema,tiny,o;gc.collect();torch.cuda.empty_cache()
        a._models.clear();a._optimizers.clear()
    slow=min(x['tokens_per_second'] for x in measurements.values())
    h=int(min(600_000_000,.85*slow*(3600-max_init-600-180))//262144)*262144
    if h<200_000_000:raise ValueError('Short horizon cannot fit at least200M tokens')
    initial=int(min(150077440,.85*measurements['batch16']['tokens_per_second']*(1200-max_init-120-180))//131072)*131072
    if initial<100_000_000:raise ValueError('Initial horizon cannot fit at least100M tokens')
    result={'passed':True,'short_tokens':h,'initial_tokens':initial,'measurements':measurements,'max_initialization_seconds':max_init,'initializers':models,
        'evaluation_signature':signatures[0],'batch8_final_feasible':2_000_027_648/measurements['batch8']['tokens_per_second']/.85+max_init+600+180<=14400,
        'wandb_url':wandb_url,'tokens_processed':on_main(a,lambda:journal.data['confirmed_tokens']),'checkpoint_rng_ema':True,'job_id':os.environ['SLURM_JOB_ID']}
    on_main(a,lambda:atomic_json(run/'calibration.json',result))
    on_main(a,lambda:tracker.finish());a.end_training()

if __name__=='__main__':main()
