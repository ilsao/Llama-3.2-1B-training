#!/usr/bin/env python3
"""Real cross-node KD producer/consumer validation; never train benchmark weights."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('Submit with sbatch')
import concurrent.futures
import contextlib
import hashlib
import json
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'kd3b-online-pipeline-20261019/source'
sys.path[:0] = [str(HERE), str(SOURCE)]
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from safetensors import safe_open
import train
import parallel_runtime
import teacher_attention
from online_pipeline import LogitsKD
from fast_kd import PrecomputedKD
from training_state import DEFAULT_ROOT, read_json, file_digest
from spool import Spool, atomic_json, wait_for

ROLE = sys.argv[1]
ROOT = Path(sys.argv[2])
DEADLINE = int(os.environ['TRAIN_END_EPOCH'])-150
MANIFEST_SHA = 'bf1dec0109c97ad0d00d5714127aca10a6046edbd0b35e0d4c1a55fb2f192507'
SEGMENTS = [(mode, batch) for mode in ('hidden', 'logits') for batch in (16, 64)]


def byteview(tensor):
    return memoryview(tensor.view(torch.uint8).numpy()).cast('B')


def batch_digest(batch):
    h = hashlib.sha256()
    for key in ('input_ids', 'labels', 'document_ids'):
        h.update(key.encode()); h.update(batch[key].numpy().tobytes())
    return h.hexdigest()


def small_gradient_check(device):
    set_seed(42)
    h = torch.randn(9, 16, device=device, requires_grad=True)
    w = torch.randn(67, 16, device=device, requires_grad=True)
    th = torch.randn(9, 24, device=device); tw = torch.randn(67, 24, device=device)
    labels = torch.tensor([1, 5, -100, 7, 9, 11, 15, 4, 2], device=device)
    a = PrecomputedKD.apply(h, w, th, tw, labels, .25, 4)[0]
    ga = torch.autograd.grad(a, (h, w))
    b = LogitsKD.apply(h, w, th@tw.T, labels, .25, 4)[0]
    gb = torch.autograd.grad(b, (h, w))
    err = max(float((x-y).norm()/x.norm()) for x, y in zip(ga, gb))
    assert err < 1e-6 and abs(float(a-b)) < 1e-5
    return dict(hidden_vs_logits_gradient_relative_l2=err, loss_difference=float(a-b))


def main():
    torch.set_num_threads(1)
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1)
    assert accelerator.num_processes == 4
    rank = accelerator.process_index
    out = ROOT/ROLE; out.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        if rank == 0:
            atomic_json(out/name, value)
    def gather(value):
        values = [None]*4
        dist.all_gather_object(values, value)
        return values
    def guard():
        if time.time() >= DEADLINE:
            raise TimeoutError('Stopping margin reached')
    save('starting.json', dict(job_id=os.environ['SLURM_JOB_ID'], node=os.environ['SLURMD_NODENAME'], role=ROLE, timestamp=time.time()))
    checks = small_gradient_check(accelerator.device)
    checks['attention'] = teacher_attention.check(accelerator.device)
    save('numerical-checks.json', checks)
    cfg = read_json(HERE/'training-config.proposed.json')
    cfg.update(world_size=4, gradient_accumulation_steps=4, eval_documents=1,
               final_evaluation_reserve_seconds=0, stop_margin_seconds=150)
    teacher_root = Path(cfg['parallel_method']['teacher']['export'])
    manifest = read_json(teacher_root/'teacher-manifest.json')
    def verify_teacher():
        assert file_digest(teacher_root/'teacher-manifest.json') == MANIFEST_SHA
        assert manifest['equivalent_tokenization_verified']
        for name, item in manifest['files'].items():
            assert (teacher_root/name).stat().st_size == item['size']
            assert file_digest(teacher_root/name) == item['sha256'], name
        return True
    train.on_main(accelerator, verify_teacher)
    data, _, provenance = train.prepare_data(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008', out, cfg, accelerator)
    train.MASK_MODE = 'intradoc'
    set_seed(42)
    if ROLE == 'producer':
        teacher = AutoModelForCausalLM.from_pretrained(teacher_root, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(accelerator.device)
        teacher.requires_grad_(False); teacher.eval(); teacher.config.use_cache = False
        teacher_attention.configure(teacher)
        head = teacher.get_output_embeddings().weight
        setup = dict(teacher_id=manifest['model_id'], teacher_revision=manifest['revision'])
    else:
        student, initialization = train.build_model(cfg, out, provenance, accelerator)
        assert initialization['pretrained_weights_loaded'] is False
        index = read_json(teacher_root/'model.safetensors.index.json')['weight_map']
        with safe_open(str(teacher_root/index['model.embed_tokens.weight']), framework='pt', device='cpu') as f:
            head = f.get_tensor('model.embed_tokens.weight').to(accelerator.device)
        assert head.shape == (128256, 3072) and head.dtype == torch.bfloat16
        head.requires_grad_(False)
        def forward(self, input_ids=None, attention_mask=None, labels=None, payload=None, mode=None, **kwargs):
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                use_cache=False, return_dict=True).last_hidden_state
            h = hidden[:, :-1].reshape(-1, hidden.shape[-1]); target = labels[:, 1:].reshape(-1)
            w = self.get_output_embeddings().weight
            if mode == 'hidden':
                loss, _, _ = PrecomputedKD.apply(h, w, payload, head, target, .25, 1024)
            else:
                loss, _, _ = LogitsKD.apply(h, w, payload, target, .25, 1024)
            return CausalLMOutputWithPast(loss=loss, logits=None)
        student.forward = types.MethodType(forward, student)
        optimizers, groups = parallel_runtime.build_optimizers(student, cfg)
        for opt in optimizers:
            for group in opt.param_groups:
                group['lr'] = 0.; group['decay_lr'] = 0.
        model = accelerator.prepare(student); model.train()
        setup = dict(student_initialization=initialization, optimizer_groups=groups,
                     weights_updated=False, resident_reference_repeats_are_counted=True)
    save('setup.json', dict(setup, world_size=4, cpu_count=4, config=cfg,
        manifest_sha256=MANIFEST_SHA, node=os.environ['SLURMD_NODENAME'],
        gpu=torch.cuda.get_device_name(), dtype='BF16', cache_limit_bytes=10**12,
        slots_per_rank=2, cross_node_required=True, fsync=True,
        page_cache_hint='POSIX_FADV_DONTNEED; different nodes; no direct-IO durability claim'))
    accelerator.wait_for_everyone()
    save('ready.json', dict(node=os.environ['SLURMD_NODENAME']))
    wait_for(ROOT/('consumer' if ROLE == 'producer' else 'producer')/'ready.json', DEADLINE, ROOT)
    assert read_json(ROOT/'producer/ready.json')['node'] != read_json(ROOT/'consumer/ready.json')['node']
    records = []; input_count = 0
    for segment_index, (mode, batch_size) in enumerate(SEGMENTS):
        guard()
        segment = f'{segment_index}-{mode}-{batch_size}'
        accumulation = batch_size//4
        micros = accumulation*3  # first optimizer window is warmup; two timed windows
        width = 3072 if mode == 'hidden' else 128256
        shape = (8191, width); payload_bytes = 8191*width*2
        queue = Spool(ROOT, rank, segment, payload_bytes, DEADLINE)
        batches = []
        identities = []
        for micro in range(micros):
            row = 3051*16 + segment_index*256 + micro*4 + rank
            cpu = train.collate_rows([data[row]], 8192)
            batches.append(cpu)
            identities.append(dict(segment=segment, row=row, rank=rank, micro=micro,
                batch_sha256=batch_digest(cpu), teacher_manifest_sha256=MANIFEST_SHA, mode=mode,
                shape=list(shape), dtype='bfloat16'))
        increment = sum(int(b['attention_mask'].sum()) for b in batches)
        # Reserve actual input processing, including resident benchmark repeats,
        # before the work. Producer and student accounting remain separate.
        if ROLE == 'consumer':
            increment += 2*sum(int(b['attention_mask'].sum()) for b in batches[-accumulation:])
        input_count += int(accelerator.reduce(torch.tensor(increment, device=accelerator.device), reduction='sum'))
        assert input_count <= 20_000_000
        save('input-accounting.json', dict(role=ROLE, reserved_actual_nonpadding_inputs=input_count,
            includes_warmup=True, includes_resident_repeats=ROLE == 'consumer', last_segment=segment))
        if ROLE == 'producer':
            wait_for(ROOT/'consumer'/f'{segment}.go', DEADLINE, ROOT)
            cpu_buffer = torch.empty(shape, dtype=torch.bfloat16, pin_memory=True)
            local = []
            for micro, cpu in enumerate(batches):
                guard(); t0 = time.perf_counter()
                batch = {k: v.to(accelerator.device) for k, v in cpu.items()}
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                    h = teacher.model(input_ids=batch['input_ids'], attention_mask=batch['document_ids'],
                                      use_cache=False, return_dict=True).last_hidden_state[:, :-1].reshape(8191, 3072)
                    if mode == 'hidden':
                        payload = h
                    else:
                        payload = torch.empty(shape, device=accelerator.device, dtype=torch.bfloat16)
                        for start in range(0, 8191, 1024):
                            payload[start:start+1024] = h[start:start+1024]@head.T
                    if micro == 0:
                        projection = (h[:4]@head.T).cpu().contiguous()
                        atomic_json(queue.directory/'projection-check.json', dict(sha256=hashlib.sha256(byteview(projection)).hexdigest()))
                    cpu_buffer.copy_(payload, non_blocking=True)
                torch.cuda.synchronize()
                compute = time.perf_counter()-t0
                io = queue.publish(micro, byteview(cpu_buffer), identities[micro])
                local.append(dict(micro=micro, compute_and_d2h_seconds=compute, **io))
                del h, payload, batch
            for micro in range(max(0, micros-2), micros):
                wait_for(queue.path(micro, 'ack'), DEADLINE, ROOT)
            all_ranks = gather(local)
            # Per-rank service excludes backpressure. Aggregate sustainable rate
            # uses the slowest rank, not the average of rank throughputs.
            service = max(sum(r['compute_and_d2h_seconds']+r['write_seconds'] for r in rows[accumulation:]) for rows in all_ranks)
            record = dict(segment=segment, mode=mode, global_batch=batch_size, timed_updates=2,
                service_seconds=service, production_tokens_per_second=2*batch_size*8192/service,
                payload_bytes_per_update=batch_size*payload_bytes, rank_records=all_ranks)
            records.append(record); save('timings.json', records)
            del cpu_buffer
            wait_for(ROOT/'consumer'/f'{segment}.done', DEADLINE, ROOT)
        else:
            pinned = [torch.empty(shape, dtype=torch.bfloat16, pin_memory=True) for _ in range(2)]
            def read(micro):
                return queue.read_into(micro, byteview(pinned[micro%2]), identities[micro])
            accelerator.wait_for_everyone()
            save(f'{segment}.go', dict(start=time.time()))
            cached = []; local = []; windows = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(read, 0)
                for micro, cpu in enumerate(batches):
                    guard()
                    if micro % accumulation == 0:
                        accelerator.wait_for_everyone(); torch.cuda.synchronize()
                        window_start = time.perf_counter()
                        for opt in optimizers:
                            opt.zero_grad(set_to_none=True)
                        valid_local = sum(int((b['labels'][:, 1:] != -100).sum()) for b in batches[micro:micro+accumulation])
                        valid = accelerator.reduce(torch.tensor(valid_local, device=accelerator.device), reduction='sum')
                        losses = []
                    t0 = time.perf_counter(); io = pending.result(); exposed = time.perf_counter()-t0
                    payload = pinned[micro%2].to(accelerator.device, non_blocking=True)
                    batch = {k: v.to(accelerator.device) for k, v in cpu.items()}
                    torch.cuda.synchronize()  # pinned slot is safe to reuse after this copy
                    if micro+1 < micros:
                        pending = pool.submit(read, micro+1)
                    if micro == 0 and mode == 'hidden':
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            projection = (payload[:4]@head.T).cpu().contiguous()
                        expected = read_json(queue.directory/'projection-check.json')['sha256']
                        assert hashlib.sha256(byteview(projection)).hexdigest() == expected
                    compute_start = time.perf_counter()
                    with accelerator.no_sync(model) if (micro+1)%accumulation else contextlib.nullcontext():
                        with accelerator.autocast():
                            loss = model(input_ids=batch['input_ids'], attention_mask=batch['document_ids'],
                                labels=batch['labels'], payload=payload, mode=mode).loss
                        accelerator.backward(loss*(4/valid))
                    losses.append(loss.detach())
                    if (micro+1)%accumulation == 0:
                        accelerator.clip_grad_norm_(model.parameters(), .5)
                        for opt in optimizers:
                            opt.step()
                    torch.cuda.synchronize()
                    compute = time.perf_counter()-compute_start
                    queue.acknowledge(micro)
                    local.append(dict(micro=micro, exposed_read_wait_seconds=exposed,
                        student_compute_seconds=compute, **io))
                    if micro >= micros-accumulation:
                        cached.append((batch, payload))
                    else:
                        del batch, payload
                    if (micro+1)%accumulation == 0:
                        windows.append(dict(seconds=time.perf_counter()-window_start,
                            loss=float(accelerator.reduce(torch.stack(losses).sum(), reduction='sum')/valid)))
            resident = []
            for repeat in range(2):
                guard(); accelerator.wait_for_everyone(); torch.cuda.synchronize()
                start = time.perf_counter()
                for opt in optimizers:
                    opt.zero_grad(set_to_none=True)
                losses = []
                for micro, (batch, payload) in enumerate(cached):
                    with accelerator.no_sync(model) if micro+1 < accumulation else contextlib.nullcontext():
                        with accelerator.autocast():
                            loss = model(input_ids=batch['input_ids'], attention_mask=batch['document_ids'],
                                labels=batch['labels'], payload=payload, mode=mode).loss
                        accelerator.backward(loss*(4/valid))
                    losses.append(loss.detach())
                accelerator.clip_grad_norm_(model.parameters(), .5)
                for opt in optimizers:
                    opt.step()
                torch.cuda.synchronize()
                duration = time.perf_counter()-start
                loss_value = float(accelerator.reduce(torch.stack(losses).sum(), reduction='sum')/valid)
                assert abs(loss_value-windows[-1]['loss']) < 1e-5, 'LR0 resident/disk objective mismatch'
                resident.append(dict(seconds=duration, loss=loss_value))
            all_ranks = gather(dict(micros=local, windows=windows, resident=resident))
            disk_seconds = max(sum(w['seconds'] for w in r['windows'][1:]) for r in all_ranks)
            resident_seconds = max(sum(w['seconds'] for w in r['resident']) for r in all_ranks)
            records.append(dict(segment=segment, mode=mode, global_batch=batch_size, timed_updates=2,
                disk_seconds=disk_seconds, resident_seconds=resident_seconds,
                disk_tokens_per_second=2*batch_size*8192/disk_seconds,
                resident_tokens_per_second=2*batch_size*8192/resident_seconds,
                disk_slowdown=disk_seconds/resident_seconds, rank_records=all_ranks))
            save('timings.json', records)
            cached.clear(); del payload, batch, pinned
            accelerator.wait_for_everyone()
            save(f'{segment}.done', dict(complete=True))
        accelerator.wait_for_everyone(); torch.cuda.empty_cache()
    save('results.json', dict(complete=True, role=ROLE, job_id=os.environ['SLURM_JOB_ID'],
        records=records, processed_input_reservation=input_count,
        limitation='Two timed optimizer windows per geometry; no long-duration filesystem or convergence claim'))
    print(f'{ROLE} rank {rank} validation complete', flush=True)
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        ROOT.mkdir(parents=True, exist_ok=True)
        atomic_json(ROOT/f'abort-{ROLE}-{os.environ.get("RANK", "unknown")}.json', dict(error=repr(exc), timestamp=time.time()))
        raise
