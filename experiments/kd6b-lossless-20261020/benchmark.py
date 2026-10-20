#!/usr/bin/env python3
"""Paired, single-allocation system validation; no trained weights are exported."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('GPU/model checks must run through sbatch')
import contextlib
import json
import statistics
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'kd3b-hidden8-20261019/source'
sys.path[:0] = [str(HERE), str(SOURCE)]
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import train
import parallel_runtime
import teacher_attention
from fast_kd import PrecomputedKD
from fused_kd import FusedKD
from training_state import DEFAULT_ROOT, atomic_json, read_json, file_digest

WARMUP, TIMED = 3, 15
ARMS = [('checkpoint_eager', True, False), ('no_checkpoint_eager', False, False),
        ('no_checkpoint_fused', False, True), ('checkpoint_fused', True, True)]
MANIFEST_SHA = 'bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507'


def main():
    torch.set_num_threads(1)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    assert accelerator.num_processes == 8
    rank, device = accelerator.process_index, accelerator.device
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    deadline = int(os.environ['TRAIN_END_EPOCH']) - 90
    def save(name, value):
        return train.on_main(accelerator, lambda: atomic_json(out / name, value))
    def guard():
        if time.time() >= deadline:
            raise TimeoutError('Validation stopping margin reached; no retry')
    def maximum(x):
        return float(train.reduce_max(torch.tensor(x, dtype=torch.float64, device=device), accelerator))
    counters = {'teacher_inputs': 0, 'student_inputs': 0}
    def charge(teacher, student):
        counters['teacher_inputs'] += teacher
        counters['student_inputs'] += student
        if max(counters.values()) > 80_000_000:
            raise ValueError('Validation token cap')
        save('input-accounting.json', dict(counters, semantics='reserved before processing, includes repeats and warmup'))

    # Small analytical/autograd reference, including ignored rows and lambda endpoints.
    set_seed(42)
    checks = []
    for bf16 in (False, True):
        for lam in (0., .25, 1.):
            for ignored in (False, True):
                h = torch.randn(17, 32, device=device, requires_grad=True)
                w = torch.randn(257, 32, device=device, requires_grad=True) * .1
                th = torch.randn(17, 48, device=device)
                tw = torch.randn(257, 48, device=device) * .1
                y = torch.arange(17, device=device) + 1
                y[::3] = -100
                if ignored: y.fill_(-100)
                amp = torch.autocast('cuda', dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
                with amp:
                    ref = PrecomputedKD.apply(h, w, th, tw, y, lam, 7)
                    got = FusedKD.apply(h, w, th, tw, y, lam, 7)
                    ls = (h @ w.T).float().log_softmax(-1)
                    lt = (th @ tw.T).float().log_softmax(-1)
                    ce = torch.nn.functional.nll_loss(ls, y, ignore_index=-100, reduction='sum')
                    kl = ((lt.exp() * (lt - ls)).sum(-1) * (y != -100)).sum()
                    exact = (1-lam)*ce + lam*kl
                rg = torch.autograd.grad(ref[0], (h,w))
                gg = torch.autograd.grad(got[0], (h,w))
                eg = torch.autograd.grad(exact, (h,w))
                error = max(float((a-b).norm()/a.norm().clamp_min(1e-12)) for a,b in zip(rg,gg))
                oracle = max(float((a-b).norm()/a.norm().clamp_min(1e-12)) for a,b in zip(eg,gg))
                tolerance = .01 if bf16 else 2e-5
                assert error < tolerance and oracle < tolerance, (bf16,lam,error,oracle)
                for a,b in zip(ref,got):
                    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-4)
                checks.append(dict(bf16=bf16,lam=lam,all_ignored=ignored,gradient_relative_l2=error,oracle_relative_l2=oracle))
    save('small-gradient-checks.json', checks)
    save('teacher-attention-check.json', teacher_attention.check(device))
    cfg = read_json(HERE.parent / 'kd3b-hidden8-20261019/config.json')
    cfg.update(eval_documents=1, final_evaluation_reserve_seconds=0, stop_margin_seconds=90)
    teacher_root = Path(cfg['parallel_method']['teacher']['export'])
    def verify_teacher():
        manifest = read_json(teacher_root / 'teacher-manifest.json')
        assert file_digest(teacher_root / 'teacher-manifest.json') == MANIFEST_SHA
        assert manifest['equivalent_tokenization_verified']
        for name,item in manifest['files'].items():
            assert file_digest(teacher_root / name) == item['sha256'], name
        return manifest
    manifest = train.on_main(accelerator, verify_teacher)
    data, _, provenance = train.prepare_data(DEFAULT_ROOT / 'preprocessed/dolma-raw-baseline-20261008', out, cfg, accelerator)
    train.MASK_MODE = 'intradoc'
    set_seed(42)
    student, initialization = train.build_model(cfg, out, provenance, accelerator)
    assert not initialization['pretrained_weights_loaded']
    teacher = AutoModelForCausalLM.from_pretrained(teacher_root, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device)
    teacher.requires_grad_(False); teacher.eval(); teacher.config.use_cache = False
    teacher_attention.configure(teacher)
    head = teacher.get_output_embeddings().weight
    active = {'fused': False, 'profile': False}
    def event():
        e = torch.cuda.Event(enable_timing=True); e.record(); return e
    def forward(self, input_ids=None, attention_mask=None, labels=None, teacher_hidden=None, **kwargs):
        if active['profile']: active['body_start'] = event()
        h = self.model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False, return_dict=True).last_hidden_state
        if active['profile']: active['body_end'] = event()
        fn = FusedKD if active['fused'] else PrecomputedKD
        loss,ce,kl = fn.apply(h[:,:-1].reshape(-1,h.shape[-1]), self.get_output_embeddings().weight,
            teacher_hidden[:,:-1].reshape(-1,head.shape[1]), head, labels[:,1:].reshape(-1), .25, 1024)
        if active['profile']: active['head_end'] = event()
        return CausalLMOutputWithPast(loss=loss, logits=None)
    student.forward = types.MethodType(forward, student)
    model = accelerator.prepare(student); model.train()
    raw = accelerator.unwrap_model(model)
    def teacher_hidden(batch):
        with torch.no_grad(), accelerator.autocast():
            return teacher.model(input_ids=batch['input_ids'], attention_mask=batch['document_ids'],
                                 use_cache=False, return_dict=True).last_hidden_state
    def set_checkpoint(enabled):
        if enabled: raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else: raw.gradient_checkpointing_disable()
    def cpu_batch(micro):
        return train.collate_rows([data[3051*16 + micro*8 + rank]], 8192)
    # Same real document spans and tensors for all variants. First full-model
    # comparison tests the combined optimization before collecting speed numbers.
    cpu = cpu_batch(0)
    count = int(accelerator.reduce(cpu['attention_mask'].sum().to(device), reduction='sum'))
    charge(count, 2*count)
    batch = {k:v.to(device) for k,v in cpu.items()}
    th = teacher_hidden(batch)
    gradients = None; ref_loss = None
    for fused,checkpoint in ((False,True),(True,False)):
        set_checkpoint(checkpoint); active['fused'] = fused
        model.zero_grad(set_to_none=True)
        with accelerator.autocast():
            loss = model(input_ids=batch['input_ids'], attention_mask=batch['document_ids'],
                         labels=batch['labels'], teacher_hidden=th).loss / 8191
        accelerator.backward(loss)
        if gradients is None:
            gradients = [p.grad.detach().clone() for p in raw.parameters()]
            ref_loss = float(loss)
        else:
            norm = torch.zeros((),device=device,dtype=torch.float64)
            diff = torch.zeros_like(norm)
            for ref,p in zip(gradients,raw.parameters()):
                norm += ref.norm().double().square()
                diff += (ref-p.grad).norm().double().square()
            err = maximum(float((diff/norm).sqrt()))
            assert err < .01, err
            assert abs(float(loss)-ref_loss) < .001
            save('full-model-gradient-check.json', dict(relative_l2_max_rank=err,
                 reference_loss=ref_loss,fused_loss=float(loss),checkpoint_recomputation_removed=True,
                 vocabulary=128256,context=8192,nonzero_optimizer_trajectory_tested=False))
    del gradients, th, loss, batch, ref
    model.zero_grad(set_to_none=True)
    save('setup.json',dict(config=cfg,initialization=initialization,teacher_revision=manifest['revision'],
        gpu=torch.cuda.get_device_name(),world_size=8,job_id=os.environ['SLURM_JOB_ID'],
        stage_timings='CUDA events, rank0 only; wall throughput uses slowest rank',
        objective='.75 CE + .25 forward KL, full vocabulary, T=1',
        timing_excludes='disk dataset loading and durable token journal; forecast adds 10% and 30min',
        optimizer_math_unchanged=True,learning_rate_zero_for_timing_only=True))
    records=[]
    for name,checkpoint,fused in ARMS:
        guard(); set_checkpoint(checkpoint); active['fused']=fused
        optimizers, groups = parallel_runtime.build_optimizers(raw,cfg)
        for opt in optimizers:
            for group in opt.param_groups: group['lr']=0.; group['decay_lr']=0.
        for width in (16,64):
            accumulation=width//8
            cpus=[cpu_batch(i) for i in range(accumulation)]
            inputs=int(accelerator.reduce(sum(b['attention_mask'].sum() for b in cpus).to(device),reduction='sum'))
            targets=int(accelerator.reduce(sum((b['labels'][:,1:]!=-100).sum() for b in cpus).to(device),reduction='sum'))
            windows=[]; stages=[]
            torch.cuda.reset_peak_memory_stats()
            for step in range(WARMUP+TIMED):
                guard(); charge(inputs,inputs)
                model.zero_grad(set_to_none=True)
                accelerator.wait_for_everyone(); torch.cuda.synchronize()
                started=time.perf_counter(); intervals=[]
                for micro,cpu in enumerate(cpus):
                    batch={k:v.to(device) for k,v in cpu.items()}
                    ts=event(); th=teacher_hidden(batch); te=event()
                    active['profile']=True
                    sync=contextlib.nullcontext() if micro==accumulation-1 else accelerator.no_sync(model)
                    with sync,accelerator.autocast():
                        loss=model(input_ids=batch['input_ids'],attention_mask=batch['document_ids'],
                                   labels=batch['labels'],teacher_hidden=th).loss
                        bs=event(); accelerator.backward(loss*(8/targets)); be=event()
                    intervals.append((ts,te,active['body_start'],active['body_end'],active['head_end'],bs,be))
                    active['profile']=False
                    del th,loss,batch
                cs=event(); accelerator.clip_grad_norm_(model.parameters(),cfg['max_grad_norm']); os_=event()
                for opt in optimizers: opt.step()
                oe=event(); torch.cuda.synchronize()
                seconds=maximum(time.perf_counter()-started)
                if step>=WARMUP:
                    windows.append(seconds)
                    row=dict(teacher_ms=0.,body_forward_ms=0.,head_kd_ms=0.,backward_ms=0.,
                             clipping_ms=cs.elapsed_time(os_),optimizer_ms=os_.elapsed_time(oe))
                    for ts,te,b0,b1,b2,bs,be in intervals:
                        row['teacher_ms']+=ts.elapsed_time(te);row['body_forward_ms']+=b0.elapsed_time(b1)
                        row['head_kd_ms']+=b1.elapsed_time(b2);row['backward_ms']+=bs.elapsed_time(be)
                    stages.append(row)
            record=dict(arm=name,global_batch=width,warmup=WARMUP,timed=TIMED,
                        seconds_per_update=statistics.mean(windows),p90_seconds=sorted(windows)[int(.9*(len(windows)-1))],
                        tokens_per_second=inputs/statistics.mean(windows),step_seconds=windows,
                        peak_allocated_bytes=maximum(torch.cuda.max_memory_allocated()),
                        stage_mean_ms_rank0={k:statistics.mean(x[k] for x in stages) for k in stages[0]})
            records.append(record);save('timings.json',records)
        del optimizers,groups,opt,group
        model.zero_grad(set_to_none=True)
    forecasts=[]
    for name,_,_ in ARMS:
        pairs=[r for r in records if r['arm']==name]
        if len(pairs)!=2:continue
        training=sum(3e9/r['tokens_per_second'] for r in pairs)
        seconds=1.1*training+1800
        forecasts.append(dict(arm=name,harmonic_tokens_per_second=6e9/training,
                              forecast_full_run_h200_hours=8*seconds/3600,
                              with_this_validation_max_h200_hours=8*(seconds+1200)/3600,
                              below_64_including_validation=8*(seconds+1200)/3600<=64,
                              projected_not_observed=True))
    save('results.json',dict(records=records,forecasts=forecasts,counters=counters,
         finished=time.time(),full_6b_training_started=False,
         limitation='Short repeated windows at random weights; nonzero-LR trajectory gate remains required.'))


if __name__=='__main__':
    main()
