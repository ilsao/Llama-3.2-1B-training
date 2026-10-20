#!/usr/bin/env python3
"""Fresh KD pilots: token-clock scheduling with shadow/adaptive batch control."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('Use sbatch')
import contextlib
import datetime
import json
import math
import signal
import sys
import time
import traceback
import types
import uuid
from pathlib import Path
from bootstrap import HERE
from contracts import RUN,TARGET,Q,config,event_units,eval_units,CONFIG_DIGEST
from critical_batch import Controller,window
from gradient_probe import synchronize_and_measure
from training_state import DEFAULT_ROOT,ExperimentLedger,TokenJournal,atomic_json,read_json,file_digest,digest,window_indices,locked
from checkpointing import collectively,serializer_preflight,save_checkpoint
from tail_data import TailDataset,verify_manifest
STOP=False


def stop(*_):
    global STOP
    STOP=True


def main():
    import torch
    import torch.distributed as dist
    import pyarrow as pa
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs,set_seed
    from transformers import AutoModelForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
    import train,parallel_runtime,teacher_attention
    from parallel_contracts import Scheduler
    from fused_kd import FusedKD
    from accelerations import WholeMatrixOwners,CompiledParts,owner_update_gate
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    cfg=config()
    from launch import verify_source_seal
    verify_source_seal()
    controller=Controller(mode=cfg['cbs']['mode'],fraction=cfg['cbs']['fraction'],
                          max_units=cfg['cbs']['max_units'],beta=cfg['cbs']['ema_beta'],
                          minimum_observations=cfg['cbs']['minimum_observations'],
                          hysteresis=cfg['cbs']['hysteresis'])
    token_events=[u*Q for u in event_units(cfg)]
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1,
                    kwargs_handlers=[InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=15))])
    assert acc.num_processes==8 and 'H200' in torch.cuda.get_device_name(acc.device)
    rank,device=acc.process_index,acc.device
    out=DEFAULT_ROOT/'runs/experiments'/RUN
    end_epoch=int(os.environ['TRAIN_END_EPOCH'])
    training_deadline=end_epoch-cfg['stop_margin_seconds']-cfg['final_evaluation_reserve_seconds']
    def save(name,value):return train.on_main(acc,lambda:atomic_json(out/name,value))
    def maximum(value):return float(train.reduce_max(torch.tensor(value,device=device,dtype=torch.float64),acc))
    lock_context=None
    def start():
        nonlocal lock_context
        out.mkdir(parents=True,exist_ok=True)
        lock_context=locked(out/'.run.lock',blocking=False);lock_context.__enter__()
        if any((out/n).exists() for n in ('initialization.json','student-input-journal.json','starting.json')):
            raise ValueError('Existing initialization/allocation evidence; no automatic retry/resume')
        allocation=ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'],RUN)
        assert allocation['config_digest']==CONFIG_DIGEST and allocation['requested_seconds']==cfg['allocation_seconds']
        atomic_json(out/'resolved-config.json',cfg)
        atomic_json(out/'source-seal.json',read_json(HERE/'source-seal.json'))
        atomic_json(out/'authorization.json',read_json(HERE/'authorization.json'))
        atomic_json(out/'starting.json',dict(job_id=os.environ['SLURM_JOB_ID'],node=os.environ['SLURMD_NODENAME'],
                    timestamp=time.time(),allocation=allocation,training_deadline=training_deadline))
        return allocation
    allocation=train.on_main(acc,start)
    save('owner-update-gate.json',owner_update_gate(device))
    serializer=serializer_preflight(acc,out/'preflight')
    receipts=[None]*8;dist.all_gather_object(receipts,serializer)
    save('serializer-preflight.json',receipts)
    save('teacher-attention-gate.json',teacher_attention.check(device))

    # Canonical data and real8rank reader gate BEFORE any LM weights are created.
    prep=DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008'
    unused,evaluation,provenance=train.prepare_data(prep,out,cfg,acc)
    del unused
    tail=train.on_main(acc,verify_manifest);save('training-tail-provenance.json',tail)
    data=TailDataset(tail)
    train.MASK_MODE='intradoc'
    probe=list(range(16))+list(range(len(data)-16,len(data)))
    train.on_main(acc,lambda:data.verify_indices(probe))
    def reader_probe():
        ids=[0+rank*2,1+rank*2,len(data)-16+rank*2,len(data)-15+rank*2]
        b=train.collate_rows([data[i] for i in ids],8192)
        assert int(b['attention_mask'].sum())==4*8192
        assert bool((b['document_ids']>0).all())
        assert bool((b['labels'][:,0]==-100).all())
        return dict(rank=rank,rows=ids,nonpadding_inputs=4*8192,passed=True)
    reader=collectively(acc,reader_probe);readers=[None]*8;dist.all_gather_object(readers,reader)
    save('reader-preflight.json',dict(ranks=readers,all_ranks_passed=True,before_model_initialization=True))

    teacher_cfg=cfg['parallel_method']['teacher'];root=Path(teacher_cfg['export'])
    def verify_teacher():
        assert file_digest(root/'teacher-manifest.json')==teacher_cfg['manifest_sha256']
        m=read_json(root/'teacher-manifest.json')
        assert m['equivalent_tokenization_verified'] and m['revision']==teacher_cfg['revision']
        for name,item in m['files'].items():assert file_digest(root/name)==item['sha256'],name
        return m
    manifest=train.on_main(acc,verify_teacher)
    teacher=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
    teacher.requires_grad_(False);teacher.eval();teacher.config.use_cache=False
    teacher_attention.configure(teacher)
    set_seed(42)
    student,initialization=train.build_model(cfg,out,provenance,acc)
    assert initialization['sample_signature']=='32b43fc6e5c34a787fc02db6674dde1afc341123e39d48abb60b5d84e932a05d'
    assert not initialization['pretrained_weights_loaded']
    original_forward=student.forward;head=teacher.get_output_embeddings().weight
    active={'stats':None}
    def forward(self,input_ids=None,attention_mask=None,labels=None,teacher_hidden=None,**kwargs):
        if not self.training:
            return original_forward(input_ids=input_ids,attention_mask=attention_mask,labels=labels,**kwargs)
        if teacher_hidden is None:raise ValueError('Missing online teacher hidden state')
        h=self.model(input_ids=input_ids,attention_mask=attention_mask,use_cache=False,return_dict=True).last_hidden_state
        loss,ce,kl=FusedKD.apply(h[:,:-1].reshape(-1,h.shape[-1]),self.get_output_embeddings().weight,
            teacher_hidden[:,:-1].reshape(-1,head.shape[1]),head,labels[:,1:].reshape(-1),.25,1024)
        values=torch.stack((ce.detach().double(),kl.detach().double()))
        active['stats']=values if active['stats'] is None else active['stats']+values
        return CausalLMOutputWithPast(loss=loss,logits=None)
    student.forward=types.MethodType(forward,student)
    model=acc.prepare(student);raw=acc.unwrap_model(model);parameters=list(raw.parameters())
    raw.gradient_checkpointing_disable()
    opts,groups=parallel_runtime.build_optimizers(raw,cfg)
    owners=WholeMatrixOwners(opts[1],{id(p):n for n,p in raw.named_parameters()})
    schedulers=[Scheduler(o,cfg) for o in opts]
    parts=CompiledParts(raw,teacher);parts.set_enabled(True);model.train()
    save('optimizer-groups.json',dict(original_groups=groups,owners=owners.metadata,
         distribution='whole-matrix NorMuon owner states; replicated AdamW; zero/clip all parameters'))
    save('execution.json',dict(initialization=initialization,teacher_revision=manifest['revision'],
        plan_updates='dynamic; compare at matched input tokens',world_size=8,teacher_and_student_share_gpus=True,
        peak_memory_from_validation_gb='unmeasured for this variant',torch=torch.__version__,source_seal=read_json(HERE/'source-seal.json'),
        logging='train/loss interval token-weighted KD objective; CE/KL separate; norm preclip; LR/tokens/s current step',
        timing='required tokens/s includes data, journals and GPU step; excludes evaluation and subsequent logging',
        no_training_checkpoint_load=True))
    journals={}
    def start_journals():
        for role in ('teacher','student'):journals[role]=TokenJournal(out/f'{role}-input-journal.json',TARGET)
        return True
    train.on_main(acc,start_journals)
    tracker=None
    def start_tracking():
        nonlocal tracker
        import wandb
        identity=uuid.uuid4().hex[:12]
        settings=wandb.Settings(init_timeout=30)
        options=dict(project=cfg['wandb_project'],name=RUN,id=identity,dir=str(out),
                     group=cfg['sweep_id'],job_type=cfg['category'],config={**cfg,
                     'optimizer_groups':groups,'execution':read_json(out/'execution.json')},settings=settings)
        failure=None
        try:tracker=wandb.init(mode='online',**options)
        except Exception as error:
            failure=repr(error);tracker=wandb.init(mode='offline',reinit=True,**options)
        wandb.define_metric('train/global_step')
        for prefix in ('train/*','eval/*','resource/*','cbs/*'):wandb.define_metric(prefix,step_metric='train/global_step')
        record=dict(id=identity,url=tracker.url if failure is None else None,
                    mode='online' if failure is None else 'offline',online_error=failure)
        atomic_json(out/'wandb.json',record);return record
    tracking=train.on_main(acc,start_tracking)
    def log(values):
        def write():
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(values,allow_nan=False)+'\n');f.flush()
            tracker.log(values)
            return True
        return train.on_main(acc,write)
    evaluations=[];last_eval_step=-1;eval_seconds=0.
    def evaluate_fixed(step,seen):
        nonlocal last_eval_step,eval_seconds
        t=time.perf_counter()
        with parts.eager():
            os.environ['EVAL_DOCUMENT_OUTPUT']=str(out/f'eval-documents-{step:06d}.json')
            try:result=train.evaluate(model,evaluation,cfg,acc)
            finally:os.environ.pop('EVAL_DOCUMENT_OUTPUT',None)
        assert result['eval/prediction_tokens']==1479302
        elapsed=maximum(time.perf_counter()-t);eval_seconds+=elapsed
        result.update({'train/global_step':step,'eval/training_tokens':seen,'eval/seconds':elapsed})
        evaluations.append(result);log(result);save('evaluations.json',evaluations);last_eval_step=step
        return result
    seen=0;cursor=0;step=0;interval_ce=interval_kl=0.;interval_targets=0
    train_seconds=compute_seconds=0.;reason='target_reached';timing_by_width={}
    eval_tokens={u*Q for u in eval_units(cfg)}
    torch.cuda.reset_peak_memory_stats()
    save('training-started.json',dict(timestamp=time.time(),tokens=0,updates=0,wandb=tracking))
    probe_interval=cfg['cbs']['probe_interval_tokens'];probe_seconds_total=0.;probe_count=0
    last_cbs={};branch_checkpoints=[]
    while seen<TARGET:
        desired=(1 if seen<cfg['batch_switch_tokens'] else 4) if controller.mode=='shadow' else controller.units
        row=window(seen,step,desired,token_events,TARGET)
        is_probe=row['end_tokens']%probe_interval==0
        if maximum(int(STOP or time.time()>=training_deadline)):
            reason='wall_time_or_signal';break
        assert row['start_tokens']==seen and cursor*8192==seen
        width=row['width'];accumulation=row['accumulation']
        torch.cuda.synchronize();start=time.perf_counter()
        train.on_main(acc,lambda:data.verify_indices(range(cursor,cursor+width)))
        def prepare_window():
            ids=window_indices(cursor,len(data),8,rank,2,accumulation)
            if len(ids)!=accumulation:raise ValueError('Incomplete canonical window')
            return [train.collate_rows([data[i] for i in pair],8192) for pair in ids]
        local=collectively(acc,prepare_window)
        counts=acc.reduce(torch.tensor([sum(int(b['attention_mask'].sum()) for b in local),
               sum(int((b['labels'][:,1:]!=-100).sum()) for b in local)],device=device),reduction='sum').cpu().tolist()
        inputs,targets=counts
        assert inputs==width*8192 and targets>0 and seen+inputs<=TARGET
        keys=train.on_main(acc,lambda:{role:j.reserve(inputs) for role,j in journals.items()})
        model.zero_grad(set_to_none=True);active['stats']=None
        lrs=[float(g['lr']) for o in opts for g in o.param_groups]
        torch.cuda.synchronize();compute_start=time.perf_counter()
        for i,cpu in enumerate(local):
            b={k:v.to(device) for k,v in cpu.items()}
            with torch.no_grad(),acc.autocast():
                hidden=teacher.model(input_ids=b['input_ids'],attention_mask=b['document_ids'],
                                     use_cache=False,return_dict=True).last_hidden_state
            sync=contextlib.nullcontext() if i==len(local)-1 and not is_probe else acc.no_sync(model)
            with sync,acc.autocast():
                loss=model(input_ids=b['input_ids'],attention_mask=b['document_ids'],labels=b['labels'],teacher_hidden=hidden).loss
                acc.backward(loss*(8/targets))
            del hidden,loss,b
        probe_seconds=0.
        if is_probe:
            probe_start=time.perf_counter()
            local_targets=sum(int((b['labels'][:,1:]!=-100).sum()) for b in local)
            moments=synchronize_and_measure(raw.named_parameters(),local_targets,targets)
            torch.cuda.synchronize()
            probe_seconds=maximum(time.perf_counter()-probe_start)
            probe_seconds_total+=probe_seconds;probe_count+=1
            phase=next(p for p in cfg['phase_schedule'] if p['start']*Q<=seen<p['end']*Q)
            allow_change=seen >= (phase['start']+phase['warmup_steps'])*Q
            last_cbs=controller.observe(moments['global']['signal'],moments['global']['noise'],
                                        targets/width,allow_change=allow_change)
            record=dict(global_step_before_update=step,window_start_tokens=seen,
                        window_end_tokens=row['end_tokens'],local_batch_sequences=width//8,
                        valid_targets=targets,moments=moments,controller=last_cbs,
                        probe_seconds=probe_seconds,state=controller.state_dict())
            def write_probe():
                with (out/'critical-batch.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                return True
            train.on_main(acc,write_probe)
        stats=active['stats'];norm=torch.nn.utils.clip_grad_norm_(parameters,cfg['max_grad_norm'])
        finite=torch.isfinite(norm)&torch.isfinite(stats).all()
        # Every processed window stays charged even if its update cannot execute.
        if maximum(int(not bool(finite))):
            train.on_main(acc,lambda:{r:j.confirm(keys[r]) for r,j in journals.items()})
            raise FloatingPointError('Nonfinite processed training window; tokens retained; no automatic retry')
        e0,e1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);e0.record()
        for optimizer in opts:optimizer.step()
        owners.publish();e1.record();torch.cuda.synchronize()
        compute=maximum(time.perf_counter()-compute_start);opt_seconds=e0.elapsed_time(e1)/1000
        sums=acc.reduce(stats,reduction='sum').cpu().tolist()
        train.on_main(acc,lambda:{r:j.confirm(keys[r]) for r,j in journals.items()})
        seen+=inputs;cursor+=width;step=row['update']
        for scheduler in schedulers:scheduler.set_tokens(seen)
        elapsed=maximum(time.perf_counter()-start)
        train_seconds+=elapsed;compute_seconds+=compute
        entry=timing_by_width.setdefault(str(width),dict(steps=0,inputs=0,seconds=0.,compute_seconds=0.))
        entry['steps']+=1;entry['inputs']+=inputs;entry['seconds']+=elapsed;entry['compute_seconds']+=compute
        interval_ce+=sums[0];interval_kl+=sums[1];interval_targets+=targets
        if step==1 or step%cfg['logging_steps']==0 or seen==TARGET:
            log({'train/global_step':step,'train/loss':(.75*interval_ce+.25*interval_kl)/interval_targets,
                 'train/ce_loss':interval_ce/interval_targets,'train/teacher_kl':interval_kl/interval_targets,
                 'train/grad_norm':float(norm),'train/learning_rate':lrs[0],'train/lr_muon':lrs[-1],
                 'train/tokens_per_second':inputs/elapsed,'train/compute_tokens_per_second':inputs/compute,
                 'train/total_tokens_seen':seen,'train/global_batch_sequences':width,
                 'train/step_seconds':elapsed,'train/compute_seconds':compute,'train/optimizer_seconds':opt_seconds,
                 'resource/h200_hours_elapsed':8*(time.time()-int(os.environ['TRAIN_START_EPOCH']))/3600,
                 'cbs/probe_seconds':probe_seconds,
                 'cbs/next_global_batch_sequences':(16 if seen<cfg['batch_switch_tokens'] else 64) if controller.mode=='shadow' else 16*controller.units,
                 **{'cbs/'+k:v for k,v in last_cbs.items() if isinstance(v,(int,float)) and v is not None}})
            interval_ce=interval_kl=0.;interval_targets=0
            save('progress.json',dict(timestamp=time.time(),global_step=step,confirmed_tokens=seen,
                 target_tokens=TARGET,data_cursor=cursor,train_seconds=train_seconds,compute_seconds=compute_seconds,
                 eval_seconds=eval_seconds,timing_by_width=timing_by_width,
                 peak_allocated_bytes=maximum(torch.cuda.max_memory_allocated()),wandb=tracking))
        if seen in eval_tokens or seen==TARGET:
            evaluate_fixed(step,seen)
        if seen in cfg['branch_checkpoint_tokens'] and seen != TARGET:
            state=dict(run_id=RUN,global_step=step,data_cursor=cursor,confirmed_tokens=seen,
                       target_tokens=TARGET,configuration_digest=CONFIG_DIGEST,
                       cbs_controller=controller.state_dict(),wandb=tracking,
                       probe_sync_seconds=probe_seconds_total,probe_count=probe_count,
                       train_seconds=train_seconds,compute_seconds=compute_seconds,eval_seconds=eval_seconds,
                       allocation=allocation,source_seal=read_json(HERE/'source-seal.json'))
            path=save_checkpoint(acc,raw,opts,schedulers,owners,out,state)
            branch_checkpoints.append(dict(tokens=seen,global_step=step,path=path))
            save('branch-checkpoints.json',branch_checkpoints)
        if cursor==len(data) or data.shard(cursor)!=data.shard(cursor-width):
            train.on_main(acc,lambda:data.commit_receipt(out/'raw-prefix-receipt.json',cursor))
    if last_eval_step!=step:evaluate_fixed(step,seen)
    parts.set_enabled(False);model.zero_grad(set_to_none=True)
    additional=None;combined=None
    if seen==TARGET:
        import datasets
        ready=read_json(prep/'raw/ready.json')
        pieces=[datasets.Dataset.from_file(f['path']) for f in ready['packed']['eval_files']]
        full=datasets.concatenate_datasets(pieces) if len(pieces)>1 else pieces[0]
        ids=[]
        for i,r in enumerate(full.select_columns(['evaluation_slot'])):
            if r['evaluation_slot']>=10000:break
            if r['evaluation_slot']>=1000:ids.append(i)
        os.environ['EVAL_DOCUMENT_OUTPUT']=str(out/'eval-additional9k-documents.json')
        t=time.perf_counter()
        try:additional=train.evaluate(model,full.select(ids),{**cfg,'eval_documents':10000},acc)
        finally:os.environ.pop('EVAL_DOCUMENT_OUTPUT',None)
        additional['eval/documents']=9000
        assert additional['eval/prediction_tokens']==13666812
        additional['eval/seconds']=maximum(time.perf_counter()-t)
        fixed=evaluations[-1];nll=fixed['eval/nll']+additional['eval/nll']
        tokens=fixed['eval/prediction_tokens']+additional['eval/prediction_tokens']
        combined=dict(documents=10000,nll=nll,prediction_tokens=tokens,perplexity=math.exp(nll/tokens),
                      global_step=step,training_tokens=seen)
        save('additional9k.json',additional);save('combined10k.json',combined)
        log({'train/global_step':step,'eval/additional9k_perplexity':additional['eval/perplexity'],
             'eval/combined10k_perplexity':combined['perplexity']})
    train.on_main(acc,lambda:data.commit_receipt(out/'raw-prefix-receipt.json',cursor))
    state=dict(run_id=RUN,global_step=step,data_cursor=cursor,confirmed_tokens=seen,
               target_tokens=TARGET,stop_reason=reason,configuration_digest=CONFIG_DIGEST,
               wandb=tracking,allocation=allocation,source_seal=read_json(HERE/'source-seal.json'),
               cbs_controller=controller.state_dict(),branch_checkpoints=branch_checkpoints,
               probe_sync_seconds=probe_seconds_total,probe_count=probe_count,
               fixed1k=evaluations[-1],additional9k=additional,combined10k=combined,
               train_seconds=train_seconds,compute_seconds=compute_seconds,eval_seconds=eval_seconds,
               wall_epoch=time.time(),h200_hours_including_allocation_so_far=8*(time.time()-int(os.environ['TRAIN_START_EPOCH']))/3600)
    checkpoint=save_checkpoint(acc,raw,opts,schedulers,owners,out,state)
    export=train.export_model(acc,model,out,state,validate=True)
    complete=seen==TARGET and combined is not None
    save('summary.json',{**state,'complete':complete,'checkpoint':checkpoint,'export':export,
                         'h200_hours_final':'pending Slurm reconciliation','full6b_quality_noninferiority_proven':False})
    if complete:save('complete.json',dict(tokens=seen,updates=step,timestamp=time.time(),checkpoint=checkpoint,export=export))
    from torch._dynamo.utils import counters
    save('compiler.json',{k:dict(v) for k,v in counters.items() if v})
    def finish():
        import wandb
        from bounded_tracking import bounded
        return bounded(wandb.finish,out/'wandb-finish.json',seconds=120)
    train.on_main(acc,finish);acc.wait_for_everyone()
    if lock_context is not None:lock_context.__exit__(None,None,None)
    return 0 if complete else 75


if __name__=='__main__':
    code=1
    try:code=main()
    except BaseException as exc:
        traceback.print_exc()
        try:atomic_json(DEFAULT_ROOT/'runs/experiments'/RUN/f'abort-rank{os.environ.get("RANK","unknown")}.json',
                        dict(error=repr(exc),timestamp=time.time(),job_id=os.environ.get('SLURM_JOB_ID')))
        except Exception:pass
    sys.stdout.flush();sys.stderr.flush();os._exit(code)
