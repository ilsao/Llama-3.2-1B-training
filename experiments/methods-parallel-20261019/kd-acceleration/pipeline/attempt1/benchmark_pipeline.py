#!/usr/bin/env python3
"""8-H200 paired validation, official 3B teacher + fresh 1B student, LR=0."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):raise SystemExit('Use sbatch')
import sys,time,json,contextlib,statistics,types,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(HERE.parent.parent/'source')]
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM
from training_state import DEFAULT_ROOT,read_json,atomic_json,file_digest
from online_pipeline import OnlineTeacherPipeline,attach_pipeline_loss
from pretrained_teacher import attach as attach_inline
from checks import run_checks
import train,parallel_runtime
from attention_backend import configure_attention


def main():
    torch.set_num_threads(1);accelerator=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=1)
    assert accelerator.num_processes==8
    set_seed(42);out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
    def save(name,value):
        if accelerator.is_main_process:atomic_json(out/name,value)
    checks=run_checks(accelerator.device);save('gpu-checks.json',checks)
    cfg=read_json(HERE.parent.parent/'configs/K1-seed42.json')
    proposal=read_json(HERE.parent/'pretrained-teacher-proposal.json')
    teacher_root=DEFAULT_ROOT/'caches/kd-teachers/llama32-3b-13afe512-pipeline-20261019'
    assert file_digest(teacher_root/'teacher-manifest.json')==read_json(HERE/'teacher-receipt.json')['manifest_sha256']
    manifest=read_json(teacher_root/'teacher-manifest.json')
    def verify():
        assert manifest['model_id']==proposal['teacher']['model_id'] and manifest['revision']==proposal['teacher']['revision']
        assert manifest['equivalent_tokenization_verified']
        for name,item in manifest['files'].items():
            f=teacher_root/name
            assert f.stat().st_size==item['size'] and file_digest(f)==item['sha256'],name
        return True
    train.on_main(accelerator,verify)
    prep=DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008'
    data_cfg=dict(cfg,eval_documents=1,final_evaluation_reserve_seconds=0,stop_margin_seconds=90)
    data,_,provenance=train.prepare_data(prep,out,data_cfg,accelerator)
    set_seed(42);student,initialization=train.build_model(cfg,out,provenance,accelerator)
    teacher=AutoModelForCausalLM.from_pretrained(teacher_root,local_files_only=True,torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa').to(accelerator.device)
    teacher.requires_grad_(False);teacher.eval();teacher.config.use_cache=False
    configure_attention(teacher,cfg['attention_backend'])
    method={'name':'kd','temperature':1.,'teacher_weight':.25,'logit_chunk':1024}
    original=student.forward
    inline=attach_inline(student,teacher,method,manifest);inline_forward=student.forward
    student.forward=original
    attach_pipeline_loss(student,method,manifest);buffered_forward=student.forward
    active={'mode':'inline'}
    def dispatch(self,*args,**kwargs):
        return (inline_forward if active['mode']=='inline' else buffered_forward)(*args,**kwargs)
    student.forward=types.MethodType(dispatch,student)
    optimizers,metadata=parallel_runtime.build_optimizers(student,cfg)
    for opt in optimizers:
        for g in opt.param_groups:g['lr']=0.;g['decay_lr']=0.
    model=accelerator.prepare(student);model.train();train.MASK_MODE='intradoc'
    batches=[];local_inputs=[]
    for micro in range(2):
        cpu=train.collate_rows([data[3051*16+micro*8+accelerator.process_index]],8192)
        local_inputs.append(int(cpu['attention_mask'].sum()))
        batches.append({k:v.to(accelerator.device) for k,v in cpu.items()})
    global_valid=accelerator.reduce(sum((b['labels'][:,1:]!=-100).sum() for b in batches),reduction='sum')
    global_inputs=int(accelerator.reduce(torch.tensor(sum(local_inputs),device=accelerator.device),reduction='sum'))
    student_reserved=0;teacher_reserved=0;records=[]
    save('setup.json',dict(teacher_manifest_sha256=file_digest(teacher_root/'teacher-manifest.json'),
        teacher_id=manifest['model_id'],teacher_revision=manifest['revision'],student_random_signature=initialization['sample_signature'],
        config=cfg,method=method,optimizer_groups=metadata,global_batch=16,global_inputs=global_inputs,
        note='LR and external decay LR are zero; frozen student weights throughout benchmark'))
    def run(mode,steps,*,optimizer=True):
        nonlocal student_reserved,teacher_reserved
        if time.time()>int(os.environ['TRAIN_END_EPOCH'])-150:raise RuntimeError('Benchmark stopping margin')
        if student_reserved+steps*global_inputs>20_000_000:raise RuntimeError('Benchmark token cap')
        student_reserved+=steps*global_inputs;teacher_reserved+=steps*global_inputs
        save('input-accounting.json',dict(student_inputs_reserved=student_reserved,teacher_inputs_reserved=teacher_reserved,includes_repeated_windows=True))
        active['mode']=mode;pipe=None
        if mode!='inline':pipe=OnlineTeacherPipeline(teacher,positions=8191,vocab_size=128256,chunk=1024,overlap=mode=='pipeline')
        accelerator.wait_for_everyone();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        epoch=torch.cuda.Event(enable_timing=True);epoch.record();start=time.perf_counter()
        losses=[];timelines=[];tickets=[];n=steps*2
        def enqueue(i):
            ticket=pipe.prefetch(i,batches[i%2],nonpadding_inputs=local_inputs[i%2]);tickets.append(ticket)
        if pipe:enqueue(0)
        for i in range(n):
            if i%2==0:
                for opt in optimizers:opt.zero_grad(set_to_none=True)
            ticket=pipe.acquire(i) if pipe else None
            if mode=='pipeline' and i+1<n:enqueue(i+1)
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            with accelerator.no_sync(model) if i%2==0 else contextlib.nullcontext():
                inputs=dict(batches[i%2]);inputs['attention_mask']=inputs.pop('document_ids')
                if ticket:inputs['teacher_ticket']=ticket
                with accelerator.autocast():loss=model(**inputs,use_cache=False,num_items_in_batch=1).loss
                accelerator.backward(loss*(8/global_valid))
            if i%2==1 and optimizer:
                accelerator.clip_grad_norm_(model.parameters(),.5)
                for opt in optimizers:opt.step()
            b.record();timelines.append((a,b));losses.append(loss.detach())
            if pipe:
                pipe.release(ticket)
                if mode=='serial_buffer' and i+1<n:enqueue(i+1)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        elapsed_tensor=torch.tensor(elapsed,device=accelerator.device,dtype=torch.float64);dist.all_reduce(elapsed_tensor,op=dist.ReduceOp.MAX)
        values=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device=accelerator.device,dtype=torch.int64)
        dist.all_reduce(values,op=dist.ReduceOp.MAX)
        loss_mean=accelerator.reduce(torch.stack(losses).sum(),reduction='sum')/(global_valid*steps)
        timeline={'student':[dict(start_ms=epoch.elapsed_time(a),end_ms=epoch.elapsed_time(b)) for a,b in timelines],
                  'teacher':[dict(key=t.key,start_ms=epoch.elapsed_time(t.teacher_start),end_ms=epoch.elapsed_time(t.ready)) for t in tickets]}
        counts=pipe.drain() if pipe else dict(teacher_inputs=sum(local_inputs)*steps,student_inputs=sum(local_inputs)*steps,discarded_teacher_inputs=0)
        assert counts['teacher_inputs']==counts['student_inputs']==sum(local_inputs)*steps and counts['discarded_teacher_inputs']==0
        result=dict(mode=mode,steps=steps,seconds=float(elapsed_tensor),seconds_per_step=float(elapsed_tensor)/steps,
                    loss=float(loss_mean),peak_allocated=int(values[0]),peak_reserved=int(values[1]),rank0_timeline=timeline,
                    fill_drain_included=True)
        return result

    reference=run('inline',1,optimizer=False)
    grads=[p.grad.detach().clone() for p in model.parameters()]
    candidate=run('pipeline',1,optimizer=False)
    sums=torch.stack([torch.stack(((p.grad-g).square().sum(),g.square().sum(),p.grad.square().sum(),(p.grad*g).sum())) for p,g in zip(model.parameters(),grads)]).sum(0)
    err=(sums[0]/sums[1]).sqrt();cosine=sums[3]/(sums[1]*sums[2]).sqrt()
    dist.all_reduce(err,op=dist.ReduceOp.MAX);dist.all_reduce(cosine,op=dist.ReduceOp.MIN)
    assert float(err)<1e-4 and float(cosine)>.999999,(float(err),float(cosine))
    assert abs(candidate['loss']-reference['loss'])<1e-5
    save('full-gradient-check.json',dict(relative_l2=float(err),cosine=float(cosine),reference=reference,candidate=candidate,threshold_relative_l2=1e-4))
    del grads
    modes=['inline','serial_buffer','pipeline']
    for mode in modes:run(mode,2)
    for block in range(2):
        for mode in modes[::1 if block==0 else -1]:
            result=run(mode,6);result['block']=block;records.append(result)
            save('timings.json',records)
    medians={mode:statistics.median(r['seconds_per_step'] for r in records if r['mode']==mode) for mode in modes}
    summary=dict(job_id=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),world_size=8,global_batch=16,seed=42,
        teacher_id=manifest['model_id'],teacher_revision=manifest['revision'],median_seconds_per_step=medians,
        pipeline_speedup_vs_inline=medians['inline']/medians['pipeline'],pipeline_speedup_vs_serial_buffer=medians['serial_buffer']/medians['pipeline'],
        peak_allocated={mode:max(r['peak_allocated'] for r in records if r['mode']==mode) for mode in modes},
        student_inputs=student_reserved,teacher_inputs=teacher_reserved,optimizer_updates_to_weights=0,
        full_gradient_relative_l2=float(err),full_gradient_cosine=float(cosine),complete=True,
        limits='Fixed repeated canonical windows and fresh random student; LR=0; no convergence/PPL claim; 2 blocks of 6 updates including pipeline fill/drain.')
    save('results.json',summary)
    if accelerator.is_main_process:print(json.dumps(summary),flush=True)
    accelerator.wait_for_everyone()

if __name__=='__main__':main()
