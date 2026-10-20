#!/usr/bin/env python3
"""Random-init Llama 3.2 1B pretraining. Heavy entrypoints require Slurm.

Accelerate owns BF16/DDP/checkpoint state. Accumulation is explicit with
Accelerator(gradient_accumulation_steps=1), so token-weighted loss receives no
hidden division. The sampler is not passed to prepare(): it is already sharded.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from training_state import (DEFAULT_ROOT, PROJECT, ExperimentLedger, TokenJournal,
                            atomic_json, digest, file_digest, identifier, load_config,
                            locked, schedule_multiplier, read_json, storage_root, window_indices,
                            training_target_reached)

from horizon_runtime import seed_journal, unclipped_gradient_norm
from batch3b_runtime import TokenScheduler, configure_update
from batch3b_contracts import STUDY, Q, window, count_updates, evaluation_units

STOP = False
MASK_MODE = 'dense'


def request_stop(*_):
    global STOP
    STOP = True


def require_compute():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise RuntimeError('Run training/model tests through sbatch on allocated compute nodes')


def broadcast(value, accelerator):
    from accelerate.utils import broadcast_object_list
    return broadcast_object_list([value], from_process=0)[0]


def reduce_max(tensor, accelerator):
    # Accelerate 1.6 reduce supports sum/mean, not a MAX reduction.
    import torch.distributed as distributed
    result = tensor.clone()
    if accelerator.num_processes > 1:
        distributed.all_reduce(result, op=distributed.ReduceOp.MAX)
    return result


def on_main(accelerator, function):
    """Propagate main-rank filesystem/metadata failures instead of barrier hangs."""
    packet = None
    if accelerator.is_main_process:
        try:
            packet = {'value': function()}
        except Exception as error:
            packet = {'error': f'{type(error).__name__}: {error}'}
    packet = broadcast(packet, accelerator)
    if 'error' in packet:
        raise RuntimeError(packet['error'])
    return packet['value']


def optimizer_groups(model, weight_decay):
    """Deduplicate tied parameters by identity; embeddings/norm/bias have no decay."""
    import torch
    embeddings = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                  for p in module.parameters(recurse=False)}
    groups = {'decay': [], 'no_decay': []}
    names = {'decay': [], 'no_decay': []}
    seen = set()
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not parameter.requires_grad:
            continue
        aliases.setdefault(id(parameter), []).append(name)
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        group = 'no_decay' if id(parameter) in embeddings or parameter.ndim < 2 or name.endswith('.bias') else 'decay'
        groups[group].append(parameter)
        names[group].append(name)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if seen != expected or sum(len(x) for x in groups.values()) != len(expected):
        raise ValueError('Each trainable parameter must be assigned exactly once')
    metadata = {'primary_learning_rate_group': 'decay', 'parameter_names': names,
                'tied_aliases': [v for v in aliases.values() if len(v) > 1],
                'parameter_counts': {key: sum(p.numel() for p in value) for key, value in groups.items()}}
    return [{'params': groups['decay'], 'weight_decay': weight_decay, 'name': 'decay'},
            {'params': groups['no_decay'], 'weight_decay': 0.0, 'name': 'no_decay'}], metadata


def build_optimizers(model, cfg):
    import torch
    if cfg.get('sweep_id') == STUDY:
        from batch3b_runtime import build_optimizers as build_study_optimizers
        return build_study_optimizers(model, cfg)
    if cfg.get('optimizer', 'adamw') == 'muon':
        from optimizers import build_muon_optimizers
        return build_muon_optimizers(model, cfg)
    groups, metadata = optimizer_groups(model, cfg['weight_decay'])
    return [torch.optim.AdamW(groups, lr=cfg['learning_rate'], betas=tuple(cfg['betas']),
                              eps=cfg['epsilon'], fused=True)], metadata


def collate_rows(rows, sequence_length):
    import torch
    batch = {name: torch.tensor([row[name] for row in rows], dtype=torch.int64)
             for name in ('input_ids', 'attention_mask', 'labels')}
    if batch['input_ids'].ndim != 2 or batch['input_ids'].shape[1] != sequence_length:
        raise ValueError('Packed sequence length differs from training configuration')
    mask, labels = batch['attention_mask'], batch['labels']
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError('attention_mask must describe actual non-padding input tokens')
    if bool((mask[:, 1:] > mask[:, :-1]).any()):
        raise ValueError('Prepared sequences must use right padding')
    if bool((labels[mask == 0] != -100).any()) or bool((labels[:, 0] != -100).any()):
        raise ValueError('Padding and first input positions must have ignored labels')
    if MASK_MODE=='intradoc':
        from attention_backend import span_ids
        batch['document_ids']=span_ids(rows,sequence_length)
    return batch


def causal_loss_sum(model, batch):
    # transformers 4.51.3 ForCausalLMLoss uses reduction=sum / num_items_in_batch.
    # Passing 1 gives sum NLL, including an all-ignored batch's differentiable 0.
    inputs=dict(batch)
    if 'document_ids' in inputs:inputs['attention_mask']=inputs.pop('document_ids')
    return model(**inputs, use_cache=False, num_items_in_batch=1).loss


def prepare_data(prep, run_path, cfg, accelerator):
    """Verify once on rank zero, then memory-map the same sealed immutable files."""
    if (prep / "raw/ready.json").exists():
        from raw_baseline import prepare_data as prepare_raw
        return prepare_raw(prep, run_path, cfg, accelerator)
    from dataset import check_artifact, load_prepared

    def verify():
        configuration = read_json(prep / 'configuration.json')
        if configuration['config']['sequence_length'] != cfg['sequence_length']:
            raise ValueError('Prepared context differs from training context')
        if (configuration['config']['tokenizer_id'] != cfg['model_id'] or
                configuration['config']['tokenizer_revision'] != cfg['model_revision']):
            raise ValueError('Tokenizer/config revisions must match this baseline')
        if cfg['eval_documents'] > configuration['config']['evaluation_subset_documents']:
            raise ValueError('Requested evaluation documents exceed the prepared fixed subset')
        packed = read_json(prep / 'pack/manifest.json')
        sealed = read_json(prep / 'verify/manifest.json')
        selected = read_json(prep / 'select/manifest.json')
        if selected['fingerprint'] != configuration['fingerprint']:
            raise ValueError('Tokenizer assets have different preparation provenance')
        items = packed['details']['train_files'] + packed['details']['eval_files']
        tokenizer_path = Path(selected['details']['tokenizer_directory']).resolve()
        tokenizer_items = [item for item in selected['artifacts']
                           if Path(item['path']).resolve().parent == tokenizer_path]
        if not tokenizer_items:
            raise ValueError('Missing sealed tokenizer assets')
        for item in tokenizer_items:
            check_artifact(item, True)
        # A checksum receipt is reusable only while every immutable file's stat,
        # manifest and seal remain identical. No preparation artifact is modified.
        stat_signature = [{**item, 'mtime_ns': Path(item['path']).stat().st_mtime_ns,
                           'ctime_ns': Path(item['path']).stat().st_ctime_ns} for item in items]
        signature = digest({'files': stat_signature, 'pack': file_digest(prep / 'pack/manifest.json'),
                            'seal': file_digest(prep / 'verify/manifest.json')})
        receipt = prep.parents[1] / 'caches/training-verification' / f'{configuration["fingerprint"]}.json'
        with locked(receipt.with_suffix('.lock')):
            cached = receipt.exists() and read_json(receipt).get('signature') == signature
            print(f'Checking verified preparation {prep.name}; packed checksum cache={cached}', flush=True)
            load_prepared(prep, 'train', verify_checksums=not cached)
            load_prepared(prep, 'evaluation', verify_checksums=not cached)
            atomic_json(receipt, {'signature': signature, 'verified_at': time.time()})
        import numpy as np
        evaluation = np.load(prep / 'split/evaluation_subset.npy', mmap_mode='r')[:cfg['eval_documents']].tolist()
        provenance = {'preparation_id': prep.name, 'fingerprint': configuration['fingerprint'],
                      'dataset_revision': configuration['config']['dataset_revision'],
                      'pack_manifest_sha256': file_digest(prep / 'pack/manifest.json'),
                      'verification_manifest_sha256': file_digest(prep / 'verify/manifest.json'),
                      'evaluation_indices': evaluation, 'evaluation_indices_sha256': digest(evaluation),
                      'evaluation_selection': f'first {cfg["eval_documents"]} ordered documents of fixed subset',
                      'pack_settings': {key: packed['details'][key] for key in
                                       ('training_attention', 'training_loss', 'evaluation_loss', 'pad_token_id')},
                      'tokenizer_directory': str(tokenizer_path),
                      'bos_token_id': selected['details']['bos_token_id'],
                      'eos_token_id': selected['details']['eos_token_id']}
        old = run_path / 'data-provenance.json'
        if old.exists() and read_json(old) != provenance:
            raise ValueError('Resume preparation/evaluation provenance changed')
        atomic_json(old, provenance)
        return provenance

    provenance = on_main(accelerator, verify)
    # Do not reread the large canonical shuffle seal on every rank. Rank zero
    # already verified it; peers check the small manifests and artifact sizes.
    import datasets
    packed = read_json(prep / 'pack/manifest.json')
    if (file_digest(prep / 'pack/manifest.json') != provenance['pack_manifest_sha256'] or
            file_digest(prep / 'verify/manifest.json') != provenance['verification_manifest_sha256']):
        raise ValueError('Preparation changed after main-rank verification')
    def memory_map(key):
        items = packed['details'][key]
        for item in items:
            check_artifact(item, False)
        pieces = [datasets.Dataset.from_file(item['path']) for item in items]
        if not pieces:
            raise ValueError('Empty packed dataset')
        return datasets.concatenate_datasets(pieces) if len(pieces) > 1 else pieces[0]
    train = memory_map('train_files')
    evaluation = memory_map('eval_files')
    # Packed eval rows are document ordered; scan only scalar slots until the
    # next document. Never filter/write dataset caches once per distributed rank.
    indices = []
    for index, row in enumerate(evaluation.select_columns(['evaluation_slot'])):
        if row['evaluation_slot'] >= cfg['eval_documents']:
            break
        indices.append(index)
    if not indices:
        raise ValueError('Empty held-out evaluation subset')
    return train, evaluation.select(indices), provenance


def build_model(cfg, run_path, provenance, accelerator):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    def assets():
        path = run_path / 'assets'
        config = AutoConfig.from_pretrained(cfg['model_id'], revision=cfg['model_revision'])
        if config.model_type != 'llama' or not config.tie_word_embeddings:
            raise ValueError('Expected original Llama 3.2 1B with tied embedding/output weights')
        config.save_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(provenance['tokenizer_directory'], use_fast=True, local_files_only=True)
        if not tokenizer.is_fast or (tokenizer.bos_token_id, tokenizer.eos_token_id) != (
                provenance['bos_token_id'], provenance['eos_token_id']):
            raise ValueError('Fast tokenizer special-token IDs differ from packing')
        if len(tokenizer) != config.vocab_size:
            raise ValueError('Tokenizer vocabulary differs from original architecture')
        tokenizer.save_pretrained(path)
        return {'configuration': config.to_dict(), 'fast_tokenizer': True,
                'bos_token_id': tokenizer.bos_token_id, 'eos_token_id': tokenizer.eos_token_id}

    asset_metadata = on_main(accelerator, assets)
    config = AutoConfig.from_pretrained(run_path / 'assets', local_files_only=True)
    config.use_cache = False
    # FP32 parameter/optimizer storage; Accelerator supplies BF16 autocast.
    # This is the only initial model construction: never from_pretrained(weights).
    model = AutoModelForCausalLM.from_config(config, attn_implementation='sdpa', torch_dtype=torch.float32)
    from recipe_runtime import initialize
    recipe_initialization = initialize(model, cfg, accelerator.device)
    from attention_backend import configure_attention
    configure_attention(model, cfg.get('attention_backend', 'sdpa'))
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError('Lost original tied embedding/output parameter')
    if cfg['activation_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    def initialization():
        sample = {name: parameter.detach().reshape(-1)[:8].float().cpu().tolist()
                  for name, parameter in model.named_parameters()}
        embedding = model.get_input_embeddings().weight.detach().reshape(-1)[:256]
        if not float(embedding.std()) > 0:
            raise ValueError('Randomly initialized embedding must have nonzero variance')
        signature = digest(sample)
        record = {'method': 'AutoModelForCausalLM.from_config', 'seed': cfg['seed'],
                  'sample_signature': signature, 'num_parameters': sum(p.numel() for p in model.parameters()),
                  'architecture': asset_metadata, 'pretrained_weights_loaded': False, 'recipe': recipe_initialization}
        # Persisted JSON converts integer mapping keys (HF id2label) to strings.
        # Compare the same representation; random-weight signature remains strict.
        record = json.loads(json.dumps(record, allow_nan=False))
        path = run_path / 'initialization.json'
        if path.exists() and read_json(path) != record:
            raise ValueError('Random initialization or model configuration changed on resume')
        atomic_json(path, record)
        return record
    initialization_metadata = on_main(accelerator, initialization)
    return model, initialization_metadata


def evaluate(model, evaluation, cfg, accelerator):
    import torch
    was_training = model.training
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    totals = torch.zeros(2, dtype=torch.float64, device=accelerator.device)
    document_totals = torch.zeros((cfg['eval_documents'],2),dtype=torch.float64,device=accelerator.device)
    try:
        with torch.no_grad():
            # Unwrapped replicated DDP weights permit uneven rank batch counts
            # without a forward-time collective or padded duplicate eval rows.
            for index in range(accelerator.process_index, len(evaluation), accelerator.num_processes):
                row = evaluation[index]
                batch = collate_rows([row], cfg['sequence_length'])
                # Right padding cannot affect preceding causal predictions.
                # Evaluate all real tokens/windows, without spending the pilot
                # allocation on thousands of trailing pad positions per document.
                length = int(batch['attention_mask'].sum())
                if length < 2:
                    raise ValueError('Evaluation window must contain at least two real input tokens')
                batch = {key: value[:, :length].to(accelerator.device) for key, value in batch.items()}
                with accelerator.autocast():
                    from performance import eager_loss_model
                    nll = causal_loss_sum(eager_loss_model(unwrapped), batch)
                totals[0] += nll.detach().double()
                count = (batch['labels'][:, 1:] != -100).sum().double()
                totals[1] += count
                document_totals[int(row['evaluation_slot']),0] += nll.detach().double()
                document_totals[int(row['evaluation_slot']),1] += count
        totals = accelerator.reduce(totals, reduction='sum')
        nll, predictions = totals.cpu().tolist()
        if predictions <= 0:
            raise ValueError('Invalid held-out prediction-token count')
        if not math.isfinite(nll):
            raise FloatingPointError('Nonfinite held-out NLL')
        perplexity = math.exp(nll / predictions) if nll / predictions < 700 else float('inf')
        if not math.isfinite(perplexity):
            raise FloatingPointError('Non-finite held-out perplexity')
        document_totals = accelerator.reduce(document_totals, reduction='sum')
        length_metrics = {}
        for lo,hi in ((0,256),(256,1024),(1024,4096),(4096,float('inf'))):
            select=(document_totals[:,1]>lo)&(document_totals[:,1]<=hi)
            sums=document_totals[select].sum(dim=0).cpu().tolist()
            if sums[1]:
                label=f'{lo}_{hi}'
                length_metrics[f'eval/length_{label}_nll']=sums[0]
                length_metrics[f'eval/length_{label}_prediction_tokens']=int(sums[1])
                length_metrics[f'eval/length_{label}_perplexity']=math.exp(min(699,sums[0]/sums[1]))
        return {**length_metrics, 'eval/perplexity': perplexity, 'eval/nll': nll, 'eval/prediction_tokens': int(predictions),
                'eval/documents': cfg['eval_documents'], 'eval/sequences': len(evaluation)}
    finally:
        model.train(was_training)


def save_checkpoint(accelerator, run_path, state, mark_best=False, retain_all=False):
    name = on_main(accelerator, lambda: f'checkpoint-{state["global_step"]:06d}-{uuid.uuid4().hex[:12]}')
    pending = run_path / 'checkpoints' / (name + '.partial')
    final = pending.with_name(name)
    accelerator.save_state(str(pending), safe_serialization=True)  # ALL ranks, including per-rank RNG
    accelerator.wait_for_everyone()
    def commit():
        atomic_json(pending / 'training-state.json', state)
        files = [{'name': str(path.relative_to(pending)), 'size': path.stat().st_size,
                  'sha256': file_digest(path)}
                 for path in sorted(pending.rglob('*')) if path.is_file()]
        atomic_json(pending / 'complete.json', {'state_digest': digest(state), 'files': files})
        pending.rename(final)
        atomic_json(run_path / 'latest.json', {'checkpoint': name})
        if mark_best:
            atomic_json(run_path / 'best.json', {'checkpoint': name, 'global_step': state['global_step'],
                        'confirmed_tokens': state.get('confirmed_tokens_at_checkpoint', 0)})
        best_path = run_path / 'best.json'
        protected = read_json(best_path)['checkpoint'] if best_path.exists() else None
        complete = sorted((path for path in final.parent.glob('checkpoint-*')
                           if (path / 'complete.json').exists()), key=lambda p: p.stat().st_mtime_ns)
        for old in ([] if retain_all else complete[:-2]):
            if old.name != protected:
                shutil.rmtree(old)
        return str(final)
    return on_main(accelerator, commit)


def checkpoint_state(run_path):
    name = read_json(run_path / 'latest.json')['checkpoint']
    identifier(name)
    path = run_path / 'checkpoints' / name
    complete = read_json(path / 'complete.json')
    state = read_json(path / 'training-state.json')
    if digest(state) != complete['state_digest']:
        raise ValueError('Checkpoint training metadata changed')
    for item in complete['files']:
        file = path / item['name']
        if not file.is_file() or file.stat().st_size != item['size']:
            raise ValueError('Incomplete checkpoint; refuse to resume')
    return str(path), state


def restore_rng(checkpoint, accelerator):
    """Fail closed: Accelerate 1.6 otherwise catches RNG restore errors silently."""
    import random
    import numpy as np
    import torch
    states = torch.load(Path(checkpoint) / f'random_states_{accelerator.process_index}.pkl',
                        map_location='cpu', weights_only=False)
    for key in ('random_state', 'numpy_random_seed', 'torch_manual_seed', 'torch_cuda_manual_seed'):
        if key not in states:
            raise ValueError(f'Checkpoint is missing required RNG state: {key}')
    random.setstate(states['random_state'])
    np.random.set_state(states['numpy_random_seed'])
    torch.set_rng_state(states['torch_manual_seed'])
    torch.cuda.set_rng_state_all(states['torch_cuda_manual_seed'])


def export_model(accelerator, model, run_path, metadata, validate=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    version = on_main(accelerator, lambda: f'export-step-{metadata["global_step"]}-{uuid.uuid4().hex[:8]}')
    pending = run_path / (version + '.partial')
    model = accelerator.unwrap_model(model)
    def export():
        # Export standard architecture metadata; custom adapter is runtime-only.
        backend = model.config._attn_implementation
        model.config._attn_implementation = 'sdpa'
        try:
            model.save_pretrained(pending, safe_serialization=True, max_shard_size='2GB')
        finally:
            model.config._attn_implementation = backend
        tokenizer = AutoTokenizer.from_pretrained(run_path / 'assets', use_fast=True, local_files_only=True)
        tokenizer.save_pretrained(pending)
        atomic_json(pending / 'training-metadata.json', metadata)
        if validate:
            loaded = AutoModelForCausalLM.from_pretrained(pending, local_files_only=True, torch_dtype=torch.float32)
            if loaded.get_input_embeddings().weight is not loaded.get_output_embeddings().weight:
                raise ValueError('HF export lost tied parameters')
            for name, parameter in loaded.named_parameters():
                if not torch.equal(parameter.detach(), dict(model.named_parameters())[name].detach().cpu()):
                    raise ValueError(f'HF export reload mismatch: {name}')
            del loaded
        final = pending.with_name(version)
        pending.rename(final)
        atomic_json(run_path / 'export.json', {'path': str(final), 'reload_verified': validate})
        return str(final)
    return on_main(accelerator, export)


def startup_note(label):
    if os.environ.get('RANK','0')!='0':return
    row={'stage':label,'timestamp':time.time()}
    print('STARTUP '+json.dumps(row),flush=True)
    with (Path(os.environ['JOB_RECORD'])/'startup-stages.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')


def run(args, cfg):
    global MASK_MODE
    MASK_MODE=cfg['ppl']['mask']
    require_compute()
    invocation_started = time.monotonic()
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('train_entry')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, set_seed

    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=1,
                              kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False),
                                              InitProcessGroupKwargs(timeout=datetime.timedelta(hours=1))])
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('distributed_ready')
    if accelerator.num_processes != cfg['world_size']:
        raise ValueError('Actual distributed world size differs from configuration')
    if accelerator.device.type != 'cuda' or 'H200' not in torch.cuda.get_device_name(accelerator.device):
        raise ValueError('Training allocation must contain H200 GPUs')
    root = storage_root(args.root)
    run_path = root / 'runs/experiments' / identifier(args.run_id)
    prep = root / 'preprocessed' / identifier(args.prep_id)
    job_id = os.environ['SLURM_JOB_ID']
    ledger = ExperimentLedger(root)
    allocation = on_main(accelerator, lambda: ledger.allocation(job_id, args.run_id))
    if allocation['config_digest'] != digest(cfg):
        raise ValueError('Training configuration differs from the ledger reservation')
    job_start = int(os.environ['TRAIN_START_EPOCH'])
    actual_job_end = int(os.environ['TRAIN_END_EPOCH'])
    job_end = min(actual_job_end, int(os.environ.get('CAMPAIGN_END_EPOCH', actual_job_end)))
    if actual_job_end - job_start > allocation['requested_seconds'] or allocation['gpu_count'] != cfg['world_size']:
        raise ValueError('Actual allocation exceeds reserved geometry/time')
    if allocation['previous_h200_hours'] + cfg['world_size'] * (actual_job_end - job_start) / 3600 > cfg['max_h200_hours'] + 1e-12:
        raise ValueError('Not enough logical-run compute budget for this allocation')
    job_end -= cfg.get('final_evaluation_reserve_seconds', 0)
    # Retain a nonblocking writer lock for the lifetime of this distributed job.
    lock_context = None
    def acquire_run():
        nonlocal lock_context
        run_path.mkdir(parents=True, exist_ok=True)
        lock_context = locked(run_path / '.run.lock', blocking=False)
        lock_context.__enter__()
        config_path = run_path / 'resolved-config.json'
        if config_path.exists():
            # Only failed pre-model import attempts from this study may retain
            # metadata. Any token journal, initialization, weights or unknown
            # artifact forbids fresh restart. No checkpoint is read or moved.
            metadata_only=(cfg.get('sweep_id')==STUDY and not args.resume
                and set(p.name for p in run_path.iterdir()) <= {'resolved-config.json','resource-accounting.json','.run.lock'}
                and allocation.get('ppl_metadata_only_repair') is True)
            if (not args.resume and not metadata_only) or read_json(config_path) != cfg:
                raise ValueError('Existing run requires --resume and identical configuration')
        elif args.resume:
            raise ValueError('Cannot resume a run without resolved configuration')
        atomic_json(config_path, cfg)
    on_main(accelerator, acquire_run)
    for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    set_seed(cfg['seed'])
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('data_start')
    train, evaluation, provenance = prepare_data(prep, run_path, cfg, accelerator)
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('data_ready')
    if time.time() >= job_end - cfg['stop_margin_seconds']:
        raise RuntimeError('Initialization/data verification exhausted this allocation; no training started')
    model, initialization = build_model(cfg, run_path, provenance, accelerator)
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('model_ready')
    raw_optimizers, group_metadata = build_optimizers(model, cfg)
    from performance import TrainingPerformance
    performance = TrainingPerformance(cfg)
    model = performance.configure(model)
    schedulers = [TokenScheduler(optimizer, cfg) for optimizer in raw_optimizers]
    prepared = accelerator.prepare(model, *raw_optimizers)
    model, optimizers = prepared[0], list(prepared[1:])
    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('optimizers_ddp_ready')
    from ppl_observers import Observers as EMA
    ema = EMA(accelerator.unwrap_model(model), cfg)
    accelerator.register_for_checkpointing(*schedulers, ema)
    all_groups = [group for optimizer in optimizers for group in optimizer.param_groups]
    primary_lr_index = next(i for i, group in enumerate(all_groups)
                            if group['name'] == group_metadata['primary_learning_rate_group'])
    set_seed(cfg['seed'] + accelerator.process_index)
    source = {str(path.relative_to(PROJECT)): file_digest(path) for path in (
        PROJECT / 'train.py', PROJECT / 'training_state.py', PROJECT / 'dataset.py',
        PROJECT / 'requirements-training.txt', PROJECT / 'preparation_parallel.py')}
    if cfg.get('optimizer', 'adamw') == 'muon':
        source['optimizers.py'] = file_digest(PROJECT / 'optimizers.py')
    source['recipe_runtime.py'] = file_digest(PROJECT / 'recipe_runtime.py')
    source['performance.py'] = file_digest(PROJECT / 'performance.py')
    if cfg.get('attention_backend') == 'flash_attention_3':
        source['attention_backend.py'] = file_digest(PROJECT / 'attention_backend.py')
        build = Path(os.environ['TRAIN_ENV_PREFIX']) / 'fa3-build.json'
        source['fa3-build.json'] = file_digest(build)
    if (prep / 'raw/ready.json').exists():
        source['raw_baseline.py'] = file_digest(PROJECT / 'raw_baseline.py')
    for name in ('horizon_runtime.py', 'horizon_contracts.py'):
        source[name] = file_digest(PROJECT / name)
    source['nccl-bootstrap-settings'] = digest({key: os.environ.get(key) for key in ('NCCL_SOCKET_IFNAME', 'NCCL_SOCKET_FAMILY')})
    for name in ('batch3b_contracts.py', 'batch3b_runtime.py'):
        source[name] = file_digest(PROJECT / name)
    for name in ('ppl_contracts.py','ppl_optimizer.py','ppl_observers.py','ppl_gate.py'):
        source[name]=file_digest(PROJECT/name)
    code_digest = digest(source)
    environment = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    if cfg.get('sweep_environment_digest', digest(environment)) != digest(environment):
        raise ValueError('Sweep environment differs from native calibration')
    def execution_identity():
        identity = {'source_hashes': source, 'environment': environment, 'config_digest': digest(cfg)}
        path = run_path / 'execution-identity.json'
        if path.exists() and read_json(path) != identity:
            raise ValueError('Logical run source/environment changed; restore original versions')
        atomic_json(path, identity)
        wandb_path = run_path / 'wandb-identity.json'
        if not wandb_path.exists():
            atomic_json(wandb_path, {'id': uuid.uuid4().hex})
        return read_json(wandb_path)['id']
    wandb_id = on_main(accelerator, execution_identity)
    state = {'global_step': 0, 'attempted_updates': 0, 'cursor': 0, 'interval_nll': 0.0,
             'interval_predictions': 0, 'code_digest': code_digest, 'environment_digest': digest(environment),
             'config_digest': digest(cfg), 'preparation_fingerprint': provenance['fingerprint'],
             'wandb_id': wandb_id, 'confirmed_tokens_at_checkpoint': 0,
             'planned_steps': count_updates(cfg), 'last_eval_step': -1,
             'checkpoint_seconds': 0.0, 'evaluation_seconds': 0.0, 'step_seconds': 0.0}
    checkpoint = None
    if args.resume and not (run_path / 'latest.json').exists():
        raise ValueError('Resume checkpoint missing')
    if args.resume and (run_path / 'latest.json').exists():
        checkpoint, saved = on_main(accelerator, lambda: checkpoint_state(run_path))
        for key in ('code_digest', 'environment_digest', 'config_digest', 'preparation_fingerprint', 'wandb_id'):
            if saved[key] != state[key]:
                raise ValueError(f'Resume changed {key}; use the original code/config/environment')
        accelerator.load_state(checkpoint)  # restores all rank RNG and registered scheduler
        state = saved
    timings = state.setdefault('timing_totals', {'initialization_seconds': 0.0,
        'initialization_max_seconds': 0.0, 'data_seconds': 0.0, 'journal_seconds': 0.0,
        'logging_seconds': 0.0, 'optimizer_seconds': 0.0, 'training_seconds': 0.0,
        'evaluation_seconds': 0.0, 'checkpoint_seconds': 0.0, 'checkpoint_count': 0})
    journal = None
    def open_journal():
        nonlocal journal
        if checkpoint is not None and not getattr(args, 'fork_from_run', None) and not (run_path / 'token-journal.json').exists():
            raise ValueError('Checkpoint exists but persistent token journal is missing; refuse budget reset')
        journal = TokenJournal(run_path / 'token-journal.json', cfg['max_training_tokens'])
        if journal.data['confirmed_tokens'] < state['confirmed_tokens_at_checkpoint']:
            raise ValueError('Persistent token journal is older than checkpoint; do not reset budgets')
        return journal.data
    journal_data = on_main(accelerator, open_journal)
    def start_tracking():
        import wandb
        tracker = wandb.init(project=cfg['wandb_project'], id=state['wandb_id'],
                             name=args.run_id, resume='allow', mode=cfg['wandb_mode'],
                             dir=str(run_path), config={**cfg, 'initialization': initialization,
                             'data_provenance': provenance, 'optimizer_groups': group_metadata,
                             'environment': environment, 'source_hashes': source,
                             'global_batch_sequences': cfg['world_size'] * cfg['microbatch_size'] * cfg['gradient_accumulation_steps'],
                             'gradient_norm': 'pre-clipping norm of synchronized DDP gradients; replicated weights counted once',
                             'throughput_timing': 'CUDA-synchronized max rank duration; forward/backward/optimizer only',
                             'loss_weighting': 'global sum NLL / global shifted prediction tokens across accumulation',
                             'sampler': 'canonical packed order; accumulation varies at token events; no repetition',
                             'batch_schedule': {'initial':16, 'after_switch':64, 'switch_tokens':cfg['batch_switch_tokens']},
                             'schedule_clock': 'nonpadding inputs;170 BS16-equivalent warmup updates per phase',
                             'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
                             'git_status': subprocess.check_output(['git', 'status', '--short'], cwd=PROJECT, text=True).strip()})
        wandb.define_metric('train/global_step')
        wandb.define_metric('train/*', step_metric='train/global_step')
        wandb.define_metric('eval/*', step_metric='train/global_step')
        atomic_json(run_path / 'wandb.json', {'id': tracker.id, 'url': tracker.url, 'mode': cfg['wandb_mode']})
        atomic_json(run_path / 'optimizer-groups.json', group_metadata)
        atomic_json(run_path / 'environment.json', environment)
        return tracker.url
    wandb_url = on_main(accelerator, start_tracking)
    # Tracker/metadata initialization must not perturb the resumed model RNG.
    if checkpoint is not None:
        restore_rng(checkpoint, accelerator)
    else:
        set_seed(cfg['seed'] + accelerator.process_index)

    def log(metrics):
        logging_started = time.monotonic()
        payload = {**metrics, 'train/global_step': state['global_step']}
        if accelerator.is_main_process:
            import wandb
            wandb.log(payload)  # W&B history index stays monotonic even if a checkpoint rewinds updates
            with (run_path / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps({**payload, 'job_id': job_id, 'timestamp': time.time()}, allow_nan=False) + '\n')
            print(json.dumps(payload, allow_nan=False), flush=True)
        timings['logging_seconds'] += time.monotonic() - logging_started

    if cfg.get('lr_transfer') or cfg.get('sweep_id') == STUDY:startup_note('tracking_ready')
    initialization_seconds = time.monotonic() - invocation_started
    timings['initialization_seconds'] += initialization_seconds
    timings['initialization_max_seconds'] = max(timings['initialization_max_seconds'], initialization_seconds)

    def compute_used():
        return allocation['previous_h200_hours'] + cfg['world_size'] * (time.time() - job_start) / 3600

    def checkpoint(mark_best=False):
        started = time.monotonic()
        state['confirmed_tokens_at_checkpoint'] = journal_data['confirmed_tokens']
        state['charged_tokens_at_checkpoint'] = journal_data['charged_tokens']
        state['resource_accounting'] = {'job_id': job_id, 'previous_h200_hours': allocation['previous_h200_hours'],
                                        'current_allocation_h200_hours_so_far': compute_used() - allocation['previous_h200_hours'],
                                        'accounting': 'running estimate; reconcile actual allocation with sacct after termination'}
        if hasattr(train, 'commit_receipt'):
            on_main(accelerator, lambda: train.commit_receipt(state['cursor']))
        path = save_checkpoint(accelerator, run_path, state, mark_best=mark_best,
                               retain_all=cfg.get('retain_all_checkpoints', False))
        elapsed = time.monotonic() - started
        state['checkpoint_seconds'] = max(state['checkpoint_seconds'], elapsed)
        timings['checkpoint_seconds'] += elapsed
        timings['checkpoint_count'] += 1
        return path

    def held_out():
        torch.cuda.synchronize()
        started = time.monotonic()
        try:
            metrics = evaluate(model, evaluation, cfg, accelerator)
        except FloatingPointError:
            state['evaluation_failed'] = True
            state['new_best'] = False
            log({'eval/nonfinite': 1, 'eval/total_training_tokens_seen': journal_data['confirmed_tokens']})
            return False
        if cfg.get('lr_transfer') and ema.betas:
            for beta in ema.betas:
                if str(beta) in ema.shadows:
                    with ema.weights(beta):
                        measured=evaluate(model,evaluation,cfg,accelerator)
                    for key in ('perplexity','nll','prediction_tokens'):
                        metrics[f'eval/ema_{beta}_{key}']=measured['eval/'+key]
                    metrics['eval/ema_updates']=ema.updates
                    metrics['eval/ema_last_snapshot_tokens']=(journal_data['confirmed_tokens']//(ema.batch*ema.transfer['ema_stride_steps']))*(ema.batch*ema.transfer['ema_stride_steps'])
        elapsed = time.monotonic() - started
        state['evaluation_seconds'] = max(state['evaluation_seconds'], elapsed)
        timings['evaluation_seconds'] += elapsed
        state['last_eval_step'] = state['global_step']
        state['last_evaluation'] = metrics
        average_nll = metrics['eval/nll'] / metrics['eval/prediction_tokens']
        state['new_best'] = average_nll < state.get('best_eval_nll', float('inf'))
        if state['new_best']:
            state['best_eval_nll'] = average_nll
        log({**metrics, 'eval/seconds': elapsed, 'eval/total_training_tokens_seen': journal_data['confirmed_tokens']})
        return True

    width = cfg['world_size'] * cfg['microbatch_size'] * cfg['gradient_accumulation_steps']
    stop_reason = 'max_steps'
    last_evaluation_time = time.monotonic()
    last_checkpoint_time = time.monotonic()
    transfer_update_times=[]
    planned_endpoint_updates = state['global_step'] + count_updates(cfg, journal_data['confirmed_tokens'])
    while state['global_step'] < cfg['max_steps']:
        if training_target_reached(cfg, journal_data):
            stop_reason = 'training_token_target'
            break
        window_started = time.monotonic()
        reserve_seconds = (max(cfg['stop_margin_seconds'], state['evaluation_seconds'] +
                               state['checkpoint_seconds'] + 30)
                           if cfg.get('checkpoint_policy') == 'final_only' else
                           max(cfg['stop_margin_seconds'], state['evaluation_seconds'] +
                               2 * state['checkpoint_seconds'] + 60))
        remaining = job_end - time.time() - reserve_seconds
        stop_tensor = torch.tensor(int(STOP or remaining <= max(state['step_seconds'], 1)), device=accelerator.device)
        if int(reduce_max(stop_tensor, accelerator).item()):
            stop_reason = 'wall_time_or_signal'
            break
        if state['step_seconds'] > 0 and not cfg.get('fixed_horizon', False):
            # Revisit the planned length as actual step/eval/checkpoint costs become known.
            per_update = state['step_seconds'] + (state['evaluation_seconds'] + state['checkpoint_seconds']) / cfg['eval_steps']
            planned = min(cfg['max_steps'], state['global_step'] + max(1, int(remaining / per_update)))
            state['planned_steps'] = broadcast(planned if accelerator.is_main_process else None, accelerator)
        data_started = time.monotonic()
        width, accumulation = window(cfg, journal_data['confirmed_tokens'])
        batches = window_indices(state['cursor'], len(train), cfg['world_size'], accelerator.process_index,
                                 cfg['microbatch_size'], accumulation)
        if not batches:
            stop_reason = 'dataset_tail'
            break
        # Prepare/count every microbatch before processing any of this window.
        from raw_baseline import DataUnavailable
        unavailable = False
        try:
            local = [collate_rows([train[index] for index in indices], cfg['sequence_length']) for indices in batches]
        except DataUnavailable:
            unavailable = True
        waiting = torch.tensor(int(unavailable), device=accelerator.device)
        if int(reduce_max(waiting, accelerator).item()):
            stop_reason = 'data_unavailable'
            break
        local_inputs = sum(int(batch['attention_mask'].sum()) for batch in local)
        local_targets = sum(int((batch['labels'][:, 1:] != -100).sum()) for batch in local)
        counts = accelerator.reduce(torch.tensor([local_inputs, local_targets], dtype=torch.int64,
                                                 device=accelerator.device), reduction='sum')
        inputs, targets = counts.cpu().tolist()
        if targets <= 0:
            raise ValueError('Training window has no prediction targets')
        if inputs != width * cfg['sequence_length']:
            raise ValueError('Token-clock study requires the verified dense raw prefix')
        beta2 = configure_update(optimizers, cfg, width, journal_data['confirmed_tokens'])
        if journal_data['charged_tokens'] + inputs > cfg['max_training_tokens']:
            stop_reason = 'token_budget'
            break
        state['planned_steps'] = planned_endpoint_updates
        timings['data_seconds'] += time.monotonic() - data_started
        journal_started = time.monotonic()
        transaction = on_main(accelerator, lambda: journal.reserve(inputs))
        journal_data = on_main(accelerator, lambda: journal.data)
        timings['journal_seconds'] += time.monotonic() - journal_started
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        nll = torch.zeros((), dtype=torch.float64, device=accelerator.device)
        # Native load_state_dict replaces group dictionaries; resolve current groups per step.
        learning_groups = [group for optimizer in optimizers for group in optimizer.param_groups]
        learning_rates = [float(group['lr']) for group in learning_groups]
        torch.cuda.synchronize()
        started = time.monotonic()
        performance.step_begin()
        staged = performance.stage(local, accelerator.device) if performance.enabled else None
        for micro, batch in enumerate(local):
            if staged is None:
                batch = {name: value.to(accelerator.device) for name, value in batch.items()}
            else:
                batch, ready = staged[micro]
                if ready is not None:
                    torch.cuda.current_stream().wait_event(ready)
            # DDP averages grads across W ranks. Each micro contributes W*NLL/
            # global_window_targets; no extra accumulation division is applied.
            sync = contextlib.nullcontext() if micro == len(local) - 1 else accelerator.no_sync(model)
            with sync, performance.forward_context(model, accelerator, batch):
                loss = causal_loss_sum(model, batch)
                nll += loss.detach().double()
                accelerator.backward(loss * (cfg['world_size'] / targets))
        from recipe_runtime import clip, diagnostic_snapshot, diagnostic_update
        grad_norm, recipe_metrics = clip(model, optimizers, cfg, journal_data['confirmed_tokens'], accelerator)
        diagnostic_samples, layer_metrics = diagnostic_snapshot(model, state['global_step']+1)
        nonfinite = (~(torch.isfinite(grad_norm) & torch.isfinite(nll))).to(torch.int64)
        finite = not bool(reduce_max(nonfinite, accelerator).item())
        optimizer_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        optimizer_events[0].record()
        if finite:
            for optimizer in optimizers:
                optimizer.step()
        optimizer_events[1].record()
        torch.cuda.synchronize()
        optimizer_elapsed = optimizer_events[0].elapsed_time(optimizer_events[1]) / 1000
        timings['optimizer_seconds'] += optimizer_elapsed
        elapsed = time.monotonic() - started
        timings['training_seconds'] += elapsed
        duration = float(reduce_max(torch.tensor(elapsed, device=accelerator.device), accelerator).item())
        total_nll = float(accelerator.reduce(nll, reduction='sum').item())
        # Forward/backward has processed all reserved inputs, even if update skipped.
        journal_started = time.monotonic()
        on_main(accelerator, lambda: journal.confirm(transaction))
        journal_data = on_main(accelerator, lambda: journal.data)
        timings['journal_seconds'] += time.monotonic() - journal_started
        state['cursor'] += width
        state['attempted_updates'] += 1
        state['step_seconds'] = performance.observe(duration, state['step_seconds'])
        if finite and not accelerator.optimizer_step_was_skipped:
            state['global_step'] += 1
            for scheduler in schedulers:
                scheduler.set_tokens(journal_data['confirmed_tokens'])
        else:
            stop_reason = 'nonfinite_or_skipped_update'
            # Tokens remain confirmed; weights/step/LR have not advanced.
            log({'train/total_tokens_seen': journal_data['confirmed_tokens'],
                 'train/budget_tokens_charged': journal_data['charged_tokens'],
                 'train/skipped_update': 1, 'train/attempted_updates': state['attempted_updates']})
            break
        ema.update(journal_data['confirmed_tokens'])
        recipe_metrics.update(layer_metrics)
        recipe_metrics.update(diagnostic_update(model, diagnostic_samples))
        state['interval_nll'] += total_nll
        state['interval_predictions'] += targets
        state['last_step_metrics'] = {
                 **recipe_metrics, 'train/grad_norm': float(grad_norm), 'train/learning_rate': learning_rates[primary_lr_index],
                 **{f"train/lr_{group['name']}": lr for group, lr in zip(learning_groups, learning_rates)},
                 **{f"train/decay_factor_{group['name']}": 1 - lr * group['weight_decay']
                    for group, lr in zip(learning_groups, learning_rates)
                    if not (cfg.get('matrix_optimizer') and group['name']=='muon')},
                 'train/tokens_per_second': inputs / duration, 'train/step_seconds': duration,
                 'train/total_tokens_seen': journal_data['confirmed_tokens'],
                 'train/budget_tokens_charged': journal_data['charged_tokens'],
                 'train/step_input_tokens': inputs, 'train/step_prediction_tokens': targets,
                 'train/global_batch_sequences': width, 'train/gradient_accumulation_steps': accumulation,
                 'train/adamw_beta2': beta2, 'train/reference_update': journal_data['confirmed_tokens'] // Q,
                 'train/h200_hours_so_far_estimate': compute_used()}
        if state['global_step'] % cfg['logging_steps'] == 0:
            log({**state['last_step_metrics'], 'train/loss': state['interval_nll'] / state['interval_predictions']})
            state['interval_nll'], state['interval_predictions'] = 0.0, 0
        window_seconds = float(reduce_max(torch.tensor(time.monotonic() - window_started,
                                  dtype=torch.float64, device=accelerator.device), accelerator).item())
        if accelerator.is_main_process:
            with (run_path / 'timing-steps.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state['global_step'], 'job_id': job_id,
                    'update_seconds': window_seconds, 'compute_seconds': duration,
                    'optimizer_seconds': optimizer_elapsed}) + '\n')
        if cfg.get('lr_transfer'):
            transfer_update_times.append(window_seconds)
            if state['global_step']==32 and cfg['lr_transfer']['kind']!='validation':
                from lr_transfer import projected_allocation
                forecast=on_main(accelerator,lambda:projected_allocation(cfg,state['global_step'],transfer_update_times[-20:],job_end-time.time()))
                log({'train/projected_seconds_required':forecast['required_seconds'],'train/allocation_projection_fits':int(forecast['fits'])})
                if not forecast['fits']:
                    stop_reason='insufficient_throughput'
                    break
        interval = (cfg['eval_steps'] if cfg.get('fixed_horizon', False) else
                    min(cfg['eval_steps'], max(1, state['planned_steps'] // 10)))
        time_interval = max(1, (job_end - job_start - reserve_seconds) / 10)
        should_evaluate = (journal_data['confirmed_tokens'] // Q in evaluation_units() or
                           state['global_step'] - max(0, state['last_eval_step']) >= interval or
                           (not cfg.get('fixed_horizon', False) and
                            time.monotonic() - last_evaluation_time >= time_interval))
        should_evaluate = broadcast(should_evaluate if accelerator.is_main_process else None, accelerator)
        if should_evaluate:
            if not held_out():
                stop_reason = 'nonfinite_or_skipped_update'
                break
            new_best = state.pop('new_best', False)
            if (cfg.get('checkpoint_policy', 'best_and_interval') != 'final_only' and
                    ((new_best and cfg['category'] != 'validation') or 'checkpoint_interval_seconds' not in cfg or
                     time.monotonic() - last_checkpoint_time >= cfg['checkpoint_interval_seconds'])):
                checkpoint(mark_best=new_best)
                last_checkpoint_time = time.monotonic()
            last_evaluation_time = time.monotonic()
        if args.stop_after_step and state['global_step'] >= args.stop_after_step:
            stop_reason = 'validation_pause'
            break
    if stop_reason == 'max_steps' and training_target_reached(cfg, journal_data):
        stop_reason = 'training_token_target'
    if state['interval_predictions']:
        log({**state['last_step_metrics'], 'train/loss': state['interval_nll'] / state['interval_predictions']})
        state['interval_nll'], state['interval_predictions'] = 0.0, 0
    if state['last_eval_step'] != state['global_step'] and not state.get('evaluation_failed'):
        if not held_out():
            stop_reason = 'nonfinite_or_skipped_update'
    ema_results = {}
    for beta in ema.betas:
        if str(beta) in ema.shadows:
            with ema.weights(beta):
                ema_results[str(beta)] = evaluate(model, evaluation, cfg, accelerator)
            log({f'eval/ema_{beta}_perplexity': ema_results[str(beta)]['eval/perplexity']})
    state['ema_evaluations'] = ema_results
    latest = checkpoint(mark_best=state.pop('new_best', False))
    summary = {**state, 'stop_reason': stop_reason, 'initialization': initialization,
               'checkpoint_loaded_for_training': bool(args.resume or getattr(args, 'fork_from_run', None)), 'configuration': cfg, 'data_provenance': provenance, 'wandb_url': wandb_url,
               'total_tokens_seen': journal_data['confirmed_tokens'], 'budget_tokens_charged': journal_data['charged_tokens'],
               'attention_kernel_calls': __import__('attention_backend').CALLS,
               'unused_training_sequences': len(train) - state['cursor'], 'checkpoint': latest,
               'h200_hours_so_far_estimate': compute_used(), 'slurm_job_id': job_id,
               'h200_hours_final': 'pending Slurm accounting reconciliation'}
    summary['performance'] = performance.diagnostics()
    if cfg.get('lr_transfer'):summary['lr_transfer_timing']=timings
    on_main(accelerator, lambda: atomic_json(run_path / 'performance.json', summary['performance']))
    if 'target_training_tokens' in cfg:
        summary['training_token_target_reached'] = training_target_reached(cfg, journal_data)
    if stop_reason != 'validation_pause' and cfg.get('export_on_completion', True):
        export_metadata = summary
        if cfg.get('sweep_id') not in (STUDY, 'horizon-wsd-20261014', 'init-recipe-20261014', 'raw-filtered-3b-20261015', 'raw3b-sota-20261015') and cfg['category'] != 'validation' and (run_path / 'best.json').exists():
            best = on_main(accelerator, lambda: read_json(run_path / 'best.json'))
            accelerator.load_state(run_path / 'checkpoints' / best['checkpoint'])
            summary['export_global_step'] = best['global_step']
            summary['export_tokens_seen'] = best['confirmed_tokens']
            export_metadata = {**summary, 'global_step': best['global_step'], 'training_global_step': state['global_step']}
        export = export_model(accelerator, model, run_path, export_metadata, validate=args.validate_export)
        summary['export'] = export
        summary.setdefault('export_global_step', state['global_step'])
        summary.setdefault('export_tokens_seen', journal_data['confirmed_tokens'])
    summary['ema_exports'] = {}
    if cfg.get('export_on_completion') and ema_results:
        for beta in ema_results:
            with ema.weights(beta):
                summary['ema_exports'][beta] = export_model(accelerator, model, run_path, {**summary, 'observer':beta, 'resumable':False}, validate=args.validate_export)
        on_main(accelerator,lambda:atomic_json(run_path/'export.json',{'path':summary['export'],'reload_verified':args.validate_export}))
    summary['observer_knots']=ema.knots
    summary['mean_snapshot_count']=ema.mean_count
    on_main(accelerator, lambda: atomic_json(run_path / 'summary.json', summary))
    def finish():
        import wandb
        wandb.finish()
    on_main(accelerator, finish)
    accelerator.wait_for_everyone()
    if lock_context is not None:
        lock_context.__exit__(None, None, None)
    accelerator.end_training()
    if stop_reason == 'nonfinite_or_skipped_update' or (
            'target_training_tokens' in cfg and stop_reason != 'validation_pause' and
            not training_target_reached(cfg, journal_data) and
            stop_reason not in ('wall_time_or_signal', 'data_unavailable')):
        return 2
    return 75 if stop_reason in ('wall_time_or_signal', 'data_unavailable') else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/train-pilot.json')
    parser.add_argument('--prep-id', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--root', type=Path, default=os.environ.get('DATA_ROOT', str(DEFAULT_ROOT)))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--fork-from-run')
    parser.add_argument('--stop-after-step', type=int, help='Validation-only deliberate checkpoint pause')
    parser.add_argument('--pause-after-step', type=int, help='Save a faithful checkpoint and pause this logical run')
    parser.add_argument('--validate-export', action='store_true', help='Verify HF reload and weight equality after training')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.resume or args.fork_from_run or args.pause_after_step:
        parser.error('This formal3B run requires fresh random initialization; no resume/fork/pause')
    if bool(args.fork_from_run) != bool(cfg.get('fork_snapshot_tokens')):
        parser.error('Fork configuration and explicit parent must agree')
    if args.fork_from_run and args.resume:
        parser.error('Fork and resume are mutually exclusive')
    if args.resume and args.run_id == 'muon-wsd-filtered-2b-20261009':
        parser.error('This filtered training request requires random initialization; checkpoint resume is unauthorized')
    if args.resume and args.run_id == 'be60p20-lrscaled-2b-20261010':
        parser.error('This BE60P20 2B request requires random initialization; checkpoint resume is unauthorized')
    if args.stop_after_step and cfg['category'] != 'validation':
        parser.error('Validation controls are restricted to validation configurations')
    if args.stop_after_step is not None and not 0 < args.stop_after_step < cfg['max_steps']:
        parser.error('stop-after-step must be within the planned run')
    if args.pause_after_step is not None:
        if args.stop_after_step is not None or not 0 < args.pause_after_step < cfg['max_steps']:
            parser.error('pause-after-step must be within planned steps and cannot combine with stop-after-step')
        args.stop_after_step = args.pause_after_step
    raise SystemExit(run(args, cfg))


if __name__ == '__main__':
    main()
