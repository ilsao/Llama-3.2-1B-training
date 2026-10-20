#!/usr/bin/env python3
"""Paired quality + throughput validation; only executable on a Slurm node."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('Use sbatch')
import contextlib
import gc
import hashlib
import json
import math
import statistics
import sys
import time
import types
import uuid
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'kd3b-hidden8-20261019/source'
PARENT=HERE.parent/'kd6b-lossless-20261020'
sys.path[:0]=[str(HERE),str(PARENT),str(SOURCE)]
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import train
import parallel_runtime
import teacher_attention
from parallel_contracts import Scheduler
from fast_kd import PrecomputedKD
from fused_kd import FusedKD
from accelerations import WholeMatrixOwners,CompiledParts,owner_update_gate
from training_state import DEFAULT_ROOT,TokenJournal,atomic_json,read_json,file_digest,window_indices

STEPS=128
ARMS=[
 dict(name='R0_original',micro=1,checkpoint=True,fused=False,compile=False,owners=False),
 dict(name='R1_prior_best',micro=1,checkpoint=False,fused=True,compile=False,owners=False),
 dict(name='R2_micro2',micro=2,checkpoint=False,fused=True,compile=False,owners=False),
 dict(name='R3_compile',micro=1,checkpoint=False,fused=True,compile=True,owners=False),
 dict(name='R4_owners',micro=1,checkpoint=False,fused=True,compile=False,owners=True),
 dict(name='R5_combined',micro=2,checkpoint=False,fused=True,compile=True,owners=True),
]
MANIFEST_SHA='bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507'


def main():
    torch.set_num_threads(1)
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1)
    assert acc.num_processes==8
    rank,device=acc.process_index,acc.device
    out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    deadline=int(os.environ['TRAIN_END_EPOCH'])-120
    def save(name,value):return train.on_main(acc,lambda:atomic_json(out/name,value))
    def maximum(x):
        return float(train.reduce_max(torch.tensor(x,device=device,dtype=torch.float64),acc))
    def guard():
        if time.time()>=deadline:raise TimeoutError('Stopping margin; preserve partial evidence, no retry')
    save('starting.json',dict(job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],timestamp=time.time()))
    save('owner-update-gate.json',owner_update_gate(device))
    save('teacher-attention-gate.json',teacher_attention.check(device))
    cfg=read_json(HERE.parent/'kd3b-hidden8-20261019/config.json')
    cfg.update(eval_documents=1000,stop_margin_seconds=120,final_evaluation_reserve_seconds=0)
    root=Path(cfg['parallel_method']['teacher']['export'])
    def verify():
        m=read_json(root/'teacher-manifest.json')
        assert file_digest(root/'teacher-manifest.json')==MANIFEST_SHA
        assert m['equivalent_tokenization_verified']
        for n,item in m['files'].items():assert file_digest(root/n)==item['sha256'],n
        return m
    manifest=train.on_main(acc,verify)
    data,evaluation,provenance=train.prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008',out,cfg,acc)
    short_indices=[i for i,r in enumerate(evaluation.select_columns(['evaluation_slot'])) if r['evaluation_slot']<128]
    short_eval=evaluation.select(short_indices)
    train.MASK_MODE='intradoc'
    teacher=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    teacher.requires_grad_(False);teacher.eval();teacher.config.use_cache=False
    teacher_attention.configure(teacher)
    set_seed(42)
    student,initialization=train.build_model(cfg,out,provenance,acc)
    assert initialization['sample_signature']=='32b43fc6e5c34a787fc02db6674dde1afc341123e39d48abb60b5d84e932a05d'
    assert not initialization['pretrained_weights_loaded']
    original_forward=student.forward
    active={'fused':False,'stats':None}
    head=teacher.get_output_embeddings().weight
    def forward(self,input_ids=None,attention_mask=None,labels=None,teacher_hidden=None,**kwargs):
        if not self.training:
            return original_forward(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        if teacher_hidden is None:raise ValueError('Online teacher hidden required')
        h=self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True).last_hidden_state
        fn=FusedKD if active['fused'] else PrecomputedKD
        loss,ce,kl=fn.apply(h[:,:-1].reshape(-1,h.shape[-1]),self.get_output_embeddings().weight,
            teacher_hidden[:,:-1].reshape(-1,head.shape[1]),head,labels[:,1:].reshape(-1),.25,1024)
        values=torch.stack((ce.detach().double(),kl.detach().double()))
        active['stats']=values if active['stats'] is None else active['stats']+values
        return CausalLMOutputWithPast(loss=loss,logits=None)
    student.forward=types.MethodType(forward,student)
    model=acc.prepare(student);raw=acc.unwrap_model(model)
    parameters=list(raw.parameters())
    initial=[p.detach().clone() for p in parameters]
    parts=CompiledParts(raw,teacher)
    def initialize_arm(arm):
        parts.set_enabled(False)
        model.zero_grad(set_to_none=True)
        with torch.no_grad():torch._foreach_copy_(parameters,initial)
        set_seed(42)
        if arm['checkpoint']:raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:raw.gradient_checkpointing_disable()
        active['fused']=arm['fused'];active['stats']=None
        parts.set_enabled(arm['compile']);model.train()
    journals={}
    def create_journals():
        for role in ('teacher','student'):journals[role]=TokenJournal(out/(role+'-input-journal.json'),180_000_000)
        return True
    train.on_main(acc,create_journals)
    def reserve(inputs):
        return train.on_main(acc,lambda:{role:j.reserve(inputs) for role,j in journals.items()})
    def confirm(keys):
        def commit():
            for role,key in keys.items():journals[role].confirm(key)
            return {role:j.data['confirmed_tokens'] for role,j in journals.items()}
        return train.on_main(acc,commit)
    def batches(cursor,width,micro):
        indices=window_indices(cursor,len(data),8,rank,micro,width//(8*micro))
        assert len(indices)==width//(8*micro)
        local=[train.collate_rows([data[i] for i in ids],8192) for ids in indices]
        counts=acc.reduce(torch.tensor([sum(int(b['attention_mask'].sum()) for b in local),
            sum(int((b['labels'][:,1:]!=-100).sum()) for b in local)],device=device),reduction='sum').cpu().tolist()
        assert counts[0]==width*8192 and counts[1]>0
        return local,counts
    def backward_window(local,targets):
        active['stats']=None
        for i,cpu in enumerate(local):
            b={k:v.to(device) for k,v in cpu.items()}
            with torch.no_grad(),acc.autocast():
                h=teacher.model(input_ids=b['input_ids'],attention_mask=b['document_ids'],use_cache=False,return_dict=True).last_hidden_state
            sync=contextlib.nullcontext() if i==len(local)-1 else acc.no_sync(model)
            with sync,acc.autocast():
                loss=model(input_ids=b['input_ids'],attention_mask=b['document_ids'],labels=b['labels'],teacher_hidden=h).loss
                acc.backward(loss*(8/targets))
            del h,loss,b
        return active['stats']

    # Real full-model numerical screen before any quality/speed claims. All
    # variants see the same complete global batch; micro2 changes grouping only.
    gradient_ref=None; checks=[]
    for arm in ARMS:
        guard();initialize_arm(arm)
        local,(inputs,targets)=batches(0,16,arm['micro'])
        key=reserve(inputs);started=time.perf_counter()
        stats=backward_window(local,targets)
        torch.cuda.synchronize();seconds=maximum(time.perf_counter()-started)
        global_stats=acc.reduce(stats,reduction='sum')/targets
        if gradient_ref is None:
            gradient_ref=[p.grad.detach().clone() for p in parameters]
            loss_ref=float(.75*global_stats[0]+.25*global_stats[1])
        squared=torch.zeros(2,device=device,dtype=torch.float64)
        for p,g in zip(parameters,gradient_ref):
            squared[0]+=(p.grad-g).norm().double().square();squared[1]+=g.norm().double().square()
        error=maximum(float((squared[0]/squared[1]).sqrt()))
        loss=float(.75*global_stats[0]+.25*global_stats[1])
        check=dict(arm=arm['name'],gradient_relative_l2=error,loss=loss,loss_difference=loss-loss_ref,
                   seconds_including_compile=seconds,passed=error<.01 and abs(loss-loss_ref)<.001)
        checks.append(check);save('gradient-gates.json',checks);confirm(key)
    del gradient_ref,local,stats,global_stats,p,g
    model.zero_grad(set_to_none=True)
    save('setup.json',dict(config=cfg,arms=ARMS,quality_steps=STEPS,quality_inputs_per_arm=STEPS*16*8192,
        initialization=initialization,teacher_revision=manifest['revision'],data_provenance=provenance,
        world_size=8,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        teacher_parameters_frozen=True,head_outside_student_optimizer=True,
        snapshots='In-memory random initialization only; no trained checkpoint read',
        evaluation='Same eager evaluator for every arm; fixed first128 periodic and fixed1k endpoint',
        quality_gate='endpoint PPL <= original implementation; short prefix only, not full6B proof'))
    results=[];tracker=None
    eligible=[a for a,c in zip(ARMS,checks) if c['passed']]
    assert eligible and eligible[0]['name']=='R0_original'
    for arm in eligible:
        guard();initialize_arm(arm)
        name=arm['name'];directory=out/name
        train.on_main(acc,lambda:directory.mkdir(parents=True,exist_ok=True))
        opts,groups=parallel_runtime.build_optimizers(raw,cfg)
        owners=WholeMatrixOwners(opts[1],{id(p):n for n,p in raw.named_parameters()}) if arm['owners'] else None
        schedulers=[Scheduler(o,cfg) for o in opts]
        save(name+'/optimizer-groups.json',dict(original_groups=groups,owners=owners.metadata if owners else None))
        def tracking_start():
            nonlocal tracker
            import wandb
            tracker=wandb.init(project='lab5-training-llama',name='kd6b-lossless2-'+name,
                id=uuid.uuid4().hex[:12],group='kd6b-lossless2-20261020',job_type='quality-validation',
                mode='online',dir=str(directory),reinit=True,settings=wandb.Settings(init_timeout=30),
                config={**cfg,'validation_arm':arm,'validation_updates':STEPS,
                    'validation_tokens':STEPS*16*8192,'initialization':initialization,
                    'optimizer_groups':groups,'lr_schedule':'unchanged3B schedule prefix; no cooldown compression',
                    'training_metrics':'interval token-weighted hardCE; other metrics current step; norm before clip',
                    'source_seal':read_json(HERE/'execution-seal.json')})
            wandb.define_metric('train/global_step')
            wandb.define_metric('train/*',step_metric='train/global_step')
            wandb.define_metric('eval/*',step_metric='train/global_step')
            atomic_json(directory/'wandb.json',dict(id=tracker.id,url=tracker.url))
            return tracker.url
        url=train.on_main(acc,tracking_start)
        def log(values):
            def write():
                with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(values)+'\n')
                tracker.log(values)
                return True
            return train.on_main(acc,write)
        seen=0;interval_ce=0.;interval_targets=0;times=[];opt_times=[];evaluations=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(1,STEPS+1):
            guard()
            local,(inputs,targets)=batches((step-1)*16,16,arm['micro'])
            key=reserve(inputs);model.zero_grad(set_to_none=True)
            current_lrs=[float(g['lr']) for o in opts for g in o.param_groups]
            torch.cuda.synchronize();t0=time.perf_counter()
            stats=backward_window(local,targets)
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
            finite=torch.isfinite(norm)&torch.isfinite(stats).all()
            if maximum(int(not bool(finite))):raise FloatingPointError('Nonfinite gradient/loss')
            e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            e0.record()
            for opt in opts:opt.step()
            if owners is not None:owners.publish()
            e1.record();torch.cuda.synchronize()
            seconds=maximum(time.perf_counter()-t0);optimizer_seconds=e0.elapsed_time(e1)/1000
            sums=acc.reduce(stats,reduction='sum').cpu().tolist()
            confirm(key);seen+=inputs
            interval_ce+=sums[0];interval_targets+=targets
            for scheduler in schedulers:scheduler.set_tokens(seen)
            if step>3:times.append(seconds);opt_times.append(optimizer_seconds)
            if step==1 or step%8==0 or step==STEPS:
                log({'train/global_step':step,'train/loss':interval_ce/interval_targets,
                     'train/grad_norm':float(norm),'train/learning_rate':current_lrs[0],
                     'train/lr_muon':current_lrs[-1],'train/tokens_per_second':inputs/seconds,
                     'train/total_tokens_seen':seen,'train/global_batch_sequences':16,
                     'train/teacher_kl':sums[1]/targets,'train/step_seconds':seconds,
                     'train/optimizer_seconds':optimizer_seconds})
                interval_ce=0.;interval_targets=0
            if step%12==0 or step==STEPS:
                with parts.eager():
                    os.environ['EVAL_DOCUMENT_OUTPUT']=str(directory/f'eval-documents-{step:04d}.json')
                    result=train.evaluate(model,evaluation if step==STEPS else short_eval,
                                          {**cfg,'eval_documents':1000 if step==STEPS else 128},acc)
                    os.environ.pop('EVAL_DOCUMENT_OUTPUT')
                result['train/global_step']=step;result['eval/training_tokens']=seen
                evaluations.append(result);log(result)
                save(name+'/evaluations.json',evaluations)
        endpoint=evaluations[-1]
        peak=maximum(torch.cuda.max_memory_allocated())
        # Measure BS64 on trained weights with optimizer states already present.
        # Its LR/decay are zero and it happens AFTER endpoint quality evaluation.
        for opt in opts:
            for group in opt.param_groups:group['lr']=0.;group['decay_lr']=0.
        large_times=[];large_opt=[]
        for step in range(13):
            guard();local,(inputs,targets)=batches(2048+step*64,64,arm['micro'])
            key=reserve(inputs);model.zero_grad(set_to_none=True)
            torch.cuda.synchronize();t0=time.perf_counter()
            stats=backward_window(local,targets)
            norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
            if maximum(int(not bool(torch.isfinite(norm)&torch.isfinite(stats).all()))):
                raise FloatingPointError('Nonfinite BS64 benchmark')
            e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record()
            for opt in opts:opt.step()
            if owners is not None:owners.publish()
            e1.record();torch.cuda.synchronize()
            seconds=maximum(time.perf_counter()-t0);confirm(key)
            if step>=3:large_times.append(seconds);large_opt.append(e0.elapsed_time(e1)/1000)
        peak=max(peak,maximum(torch.cuda.max_memory_allocated()))
        rate16=131072/statistics.mean(times);rate64=524288/statistics.mean(large_times)
        training_seconds=3e9/rate16+3e9/rate64
        record=dict(arm=arm,quality_inputs=seen,endpoint_evaluation=endpoint,
            tokens_per_second_bs16=rate16,tokens_per_second_bs64=rate64,
            harmonic_tokens_per_second=6e9/training_seconds,
            optimizer_seconds_bs16_rank0=statistics.mean(opt_times),optimizer_seconds_bs64_rank0=statistics.mean(large_opt),
            peak_allocated_bytes=peak,projected_full6b_hours=8*(1.1*training_seconds+1800)/3600,
            short_prefix_ppl_not_worse_than_R0=endpoint['eval/perplexity']<=results[0]['endpoint_evaluation']['eval/perplexity'] if results else True,
            full6b_quality_proven=False,wandb_url=url)
        results.append(record);save('results.json',dict(arms=results,complete=len(results)==len(eligible),
            numerical_rejections=[c for c in checks if not c['passed']],
            warning='128-step paired prefix; no final6B PPL claim; all teacher/student allocation time charged'))
        save(name+'/timings.json',dict(bs16=times,bs64=large_times,bs16_optimizer_rank0=opt_times,bs64_optimizer_rank0=large_opt))
        def finish_tracking():
            import wandb
            from bounded_tracking import bounded
            return bounded(wandb.finish,directory/'wandb-finish.json',seconds=20)
        train.on_main(acc,finish_tracking)
        parts.set_enabled(False);model.zero_grad(set_to_none=True)
        parallel_runtime.ACTIVE=[]
        del opts,schedulers,owners,opt,group,scheduler,local,stats
        gc.collect();torch.cuda.empty_cache()
    from torch._dynamo.utils import counters,compilation_time_metrics
    save('compiler.json',dict(counters={k:dict(v) for k,v in counters.items() if v},
         compile_time_categories_seconds={k:sum(v) for k,v in compilation_time_metrics.items()},
         note='Nested timing categories; do not sum'))
    save('complete.json',dict(timestamp=time.time(),results=len(results),full6b_started=False))


if __name__=='__main__':main()
