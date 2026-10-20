"""Verify and publish the user-selected checkpoint; no training or automatic retries."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training_state import DEFAULT_ROOT,PROJECT,ExperimentLedger,read_json,atomic_json,file_digest,locked,digest,TERMINAL

RUN=os.environ.get('HF_PUBLICATION_RUN','hf-best-13717-20261009')
if RUN not in ('hf-best-13717-20261009','hf-best-13717-20261009-recovery-1'):raise ValueError('Unknown publication identity')
PARENT='baseline-muon-wsd-raw-20261009'
MODEL='asciibase64/Llama-3.2-1B-dolma3_mix'
DIRECTORY=DEFAULT_ROOT/'runs/publications'/RUN


def artifact_hashes(path):
    size=path.stat().st_size
    sha256=hashlib.sha256();git_sha1=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            sha256.update(block);git_sha1.update(block)
    return dict(size=size,sha256=sha256.hexdigest(),git_sha1=git_sha1.hexdigest())


def model_card(summary,accounting,initialization,provenance):
    return f'''---
library_name: transformers
pipeline_tag: text-generation
license: llama3.2
datasets:
- allenai/dolma3_mix-150B-1025
tags:
- llama
- pretraining
- from-scratch
- muon
---
# Llama-3.2-1B-dolma3_mix

Built with Llama.

Llama 3.2 1B architecture and tokenizer, trained from **random weights** on the
unfiltered canonical training split of `allenai/dolma3_mix-150B-1025`.
Meta's pretrained model weights were not used for initialization. This is the
best saved checkpoint from the completed Muon + WSD run and its explicitly
authorized one-hour continuation from that same run's own step8500 checkpoint.

| Item | Value |
| --- | --- |
| Checkpoint / optimizer update | {summary['export_global_step']} |
| Cumulative non-padding training input tokens | {summary['total_tokens_seen']:,} |
| Held-out perplexity | {summary['last_evaluation']['eval/perplexity']:.6f}, fixed 1,000-document subset |
| Full reserved 50,000-document PPL for this checkpoint | Not measured |
| Training context | 8192 tokens |
| Parameters | {initialization['num_parameters']:,} |
| Training GPU geometry | 8 NVIDIA H200; 8 CPU threads total |
| Allocation duration across both training jobs | {sum(j['elapsed_seconds'] for j in accounting['jobs'])} seconds |
| Cumulative training allocation cost | {accounting['h200_hours_measured']:.9f} H200-hours |
| Global batch | 16 sequences, 131,072 non-padding input tokens per full update |
| Microbatch / accumulation | 1 sequence per GPU / 2 microbatches |

The reported PPL is `exp(total NLL / valid prediction tokens)` on the same fixed
first 1,000 documents of the reserved evaluation subset. It is not a full-set
measurement. The complete resolved configuration and parameter groups accompany
the weights in `training-details.json` and `resolved-training-config.json`.

Optimizer: native PyTorch Muon for hidden Linear matrices, AdamW for tied
embeddings/output and normalization/bias parameters. Muon peak LR 0.02,
momentum 0.95, Nesterov, five Newton-Schulz steps, weight decay 0.1;
auxiliary AdamW peak LR 0.0003, betas (0.9, 0.95), epsilon 1e-8.
WSD: 170 warmup updates, stable through update 6800, 1700-update linear decay to
0.1 of peak at update 8500. The continuation retained the scheduler and its
terminal LRs (Muon 0.002, auxiliary AdamW 0.00003), optimizer state and RNG.

Dataset revision: `{provenance['dataset_revision']}`. Canonical HF global
document shuffle seed 42, last 50,000 documents reserved before tokenization;
all excluded from training. Evaluation subset indices are recorded in
`training-details.json`. No data filtering was applied to this training run.
Training uses causal attention across EOS-separated documents; structural BOS,
sequence-first targets and padding are ignored in loss. Evaluation packs each
document independently with stride 8191 so each within-document target is scored
once. Tokenizer/config revision:
`{summary['configuration']['model_revision']}`. Initialization seed 42.

Weights are standard sharded safetensors with a Transformers LlamaForCausalLM
configuration. Loading needs no custom model code or FlashAttention3 adapter.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{MODEL}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
```

W&B: [{PARENT}]({summary['wandb_url']}).
The source Llama architecture/tokenizer license is included in `LICENSE.txt`.
'''


def schedule():
    r=subprocess.run(['crontab','-l'],capture_output=True,text=True)
    if r.returncode not in (0,1):raise RuntimeError('Cannot read accounting schedule')
    marker='# codex-hf-best-upload-accounting'
    lines=[x for x in r.stdout.splitlines() if marker not in x]
    cmd=shlex.join(['env','HF_PUBLICATION_RUN='+RUN,'/usr/bin/python3',str(Path(__file__).resolve()),'status'])
    lines.append(f'* * * * * {cmd} >> {shlex.quote(str(DIRECTORY/"accounting.log"))} 2>&1 {marker}')
    subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)


def submit(test_only=False):
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        if (DIRECTORY/'launch.json').exists():raise ValueError('Existing publication submission; use status')
        parent=DEFAULT_ROOT/'runs/experiments'/PARENT
        summary=read_json(parent/'summary.json');best=read_json(parent/'best.json');export=read_json(parent/'export.json')
        if (summary['export_global_step'],best['global_step'])!=(13717,13717):raise ValueError('Best checkpoint changed')
        if summary['export']!=export['path']:raise ValueError('Export pointer mismatch')
        cfg=read_json(PROJECT/'configs/train-smoke.json');cfg.update(world_size=1,max_h200_hours=.5)
        files={str(p):file_digest(p) for p in [PROJECT/'slurm/hf_best_upload.py',PROJECT/'slurm/hf-best-upload.sbatch',parent/'best.json',parent/'export.json',parent/'summary.json']}
        manifest=dict(model_id=MODEL,parent_run=PARENT,export=export['path'],checkpoint=str(parent/'checkpoints'/best['checkpoint']),
            expected_parent_commit='3a73100208da73716a3d9f2bef12139db81fcd18',sealed_files=files,config=cfg,
            user_authorization='Push this best checkpoint to asciibase64/Llama-3.2-1B-dolma3_mix',
            requested_seconds=1800,gpus=1,cpus=8,kind='export verification and publication; no training')
        from slurm.filtering_resume import shared_upper
        if shared_upper(DEFAULT_ROOT)+.5>192:raise ValueError('Insufficient shared budget')
        tag='hf-upload-'+uuid.uuid4().hex
        command=['sbatch','--parsable','--comment='+tag,str(PROJECT/'slurm/hf-best-upload.sbatch'),RUN]
        if test_only:
            r=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,check=True);print(r.stdout+r.stderr);return
        ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,RUN,cfg,1800,1,False)
        atomic_json(DIRECTORY/'authorization.json',manifest)
        state=dict(status='submission_uncertain',reservation=tag,submission_command=command,started_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(DIRECTORY/'launch.json',state)
        with ledger.transaction() as data:data['jobs'][tag].update(submission_tag=tag,submission_started_utc=state['started_utc'],submission_command=command,allocation_kind='validation_and_publication',parent_run=PARENT)
        r=subprocess.run(command,capture_output=True,text=True);state['submission_result']=dict(stdout=r.stdout,stderr=r.stderr,returncode=r.returncode);atomic_json(DIRECTORY/'launch.json',state)
        job=r.stdout.strip().split(';')[0]
        if r.returncode or not re.fullmatch(r'\d+',job):raise ValueError('Uncertain submission; retain reservation and recover before retry')
        ledger.bind(tag,job);state.update(status='submitted',job_id=job);atomic_json(DIRECTORY/'launch.json',state);schedule();print('Submitted export validation/upload job',job)


def status():
    p=DIRECTORY/'launch.json'
    if not p.exists():return
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        s=read_json(p)
        if not s.get('job_id'):raise ValueError('Uncertain submission requires recovery; no retry')
        if s['status']!='reconciled':
            r=subprocess.run(['sacct','-X','-j',s['job_id'],'--noheader','--parsable2','--format=State%30'],capture_output=True,text=True,check=True,timeout=20)
            value=r.stdout.strip().split('|')[0].split()[0] if r.stdout.strip() else 'UNKNOWN';s['slurm_state']=value
            if value in TERMINAL:
                s['accounting']=ExperimentLedger(DEFAULT_ROOT).reconcile(s['job_id']);s['status']='reconciled'
                old=subprocess.run(['crontab','-l'],capture_output=True,text=True,check=True);lines=[x for x in old.stdout.splitlines() if '# codex-hf-best-upload-accounting' not in x]
                subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)
            atomic_json(p,s)
        print(json.dumps(s))


def worker():
    from dataset import require_compute
    require_compute();a=read_json(DIRECTORY/'authorization.json')
    for name,sha in a['sealed_files'].items():
        if file_digest(Path(name))!=sha:raise ValueError('Pinned publication input changed')
    allocation=ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'],RUN)
    if allocation['config_digest']!=digest(a['config']):raise ValueError('Publication resource reservation changed')
    gpu_uuid=Path(os.environ['JOB_RECORD'],'gpus.txt').read_text().strip().split(',')[-1].strip()
    if not gpu_uuid.startswith('GPU-'):raise ValueError('Missing allocated GPU UUID')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu_uuid
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('Allocated GPU does not initialize in CUDA')
    print('CUDA initialized on allocated GPU:',gpu_uuid,flush=True)
    from safetensors import safe_open
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_download
    parent=DEFAULT_ROOT/'runs/experiments'/PARENT;export=Path(a['export']);checkpoint=Path(a['checkpoint'])
    summary=read_json(parent/'summary.json');index=read_json(export/'model.safetensors.index.json')['weight_map']
    compared=0
    with safe_open(checkpoint/'model.safetensors',framework='pt',device='cpu') as reference:
        if set(reference.keys())!=set(index):raise ValueError('Export parameter coverage changed')
        for filename in sorted(set(index.values())):
            with safe_open(export/filename,framework='pt',device='cpu') as shard:
                for key in shard.keys():
                    if index.get(key)!=filename:raise ValueError('Export index points to a different shard')
                    if not torch.equal(shard.get_tensor(key),reference.get_tensor(key)):raise ValueError('Export tensor differs: '+key)
                    compared+=1
    if compared!=len(index):raise ValueError('Export shards omit indexed parameters')
    print('Exact checkpoint/export tensor comparison passed:',compared,flush=True)
    tokenizer=AutoTokenizer.from_pretrained(export,local_files_only=True)
    original=AutoTokenizer.from_pretrained(parent/'assets',local_files_only=True)
    texts=['This model was trained from random initialization.','資料驗證與模型訓練。','def square(x):\n    return x * x']
    if tokenizer(texts,add_special_tokens=False)!=original(texts,add_special_tokens=False):raise ValueError('Export tokenizer differs')
    if (tokenizer.bos_token_id,tokenizer.eos_token_id)!=(128000,128001):raise ValueError('Tokenizer special tokens changed')
    model=AutoModelForCausalLM.from_pretrained(export,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:raise ValueError('Export lost tied output weights')
    base=tokenizer.encode(texts[0],add_special_tokens=False);ids=[tokenizer.bos_token_id]+(base*(8192//len(base)+1))[:8190]+[tokenizer.eos_token_id]
    inputs=torch.tensor([ids],device='cuda');labels=inputs.clone();labels[:,0]=-100
    with torch.inference_mode():loss=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=labels,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('8192-token standard Transformers evaluation is nonfinite')
    validation=dict(exact_tensor_equality=True,compared_tensors=compared,standard_transformers_load=True,tied_output_weights=True,tokenizer_equivalence=True,
        context_tokens=8192,smoke_loss=float(loss),evaluation_scope='Synthetic 8192-token finite-loss/load check; held-out PPL is the existing fixed-subset result',
        job_id=os.environ['SLURM_JOB_ID'],checked_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(DIRECTORY/'validation.json',validation);del model;torch.cuda.empty_cache();print('8192-token inference validation passed',flush=True)
    stage=DIRECTORY/'metadata';stage.mkdir(exist_ok=True)
    accounting=read_json(parent/'resource-accounting.json');init=read_json(parent/'initialization.json');provenance=read_json(parent/'data-provenance.json')
    (stage/'README.md').write_text(model_card(summary,accounting,init,provenance))
    (stage/'NOTICE').write_text('Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.\n')
    license_path=hf_hub_download('meta-llama/Llama-3.2-1B','LICENSE.txt',revision=summary['configuration']['model_revision'])
    (stage/'LICENSE.txt').write_bytes(Path(license_path).read_bytes())
    atomic_json(stage/'resolved-training-config.json',summary['configuration'])
    atomic_json(stage/'training-details.json',dict(global_step=summary['export_global_step'],training_tokens=summary['total_tokens_seen'],training_h200_hours=accounting['h200_hours_measured'],
        training_allocations=accounting['jobs'],initialization=init,provenance=provenance,optimizer_groups=read_json(parent/'optimizer-groups.json'),
        evaluation=summary['last_evaluation'],evaluation_scope='fixed first1000 documents of reserved subset; current full50000 PPL not measured',wandb_url=summary['wandb_url'],
        execution_identity=read_json(parent/'execution-identity.json'),export_validation=validation))
    files={p.name:p for p in export.iterdir() if p.is_file()};files.update({p.name:p for p in stage.iterdir() if p.is_file()})
    hashes={name:artifact_hashes(p) for name,p in files.items()}
    api=HfApi();info=api.model_info(MODEL)
    if info.sha!=a['expected_parent_commit']:raise ValueError('Remote repository changed before upload; review new head')
    if info.private or info.gated:raise ValueError('Expected the existing public ungated target')
    operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(p)) for name,p in files.items()]
    result=api.create_commit(MODEL,operations=operations,commit_message='Upload best Muon + WSD checkpoint at step13717',repo_type='model',parent_commit=info.sha,num_threads=4)
    receipt=dict(model_id=MODEL,revision=result.oid,commit_url=result.commit_url,export=str(export),checkpoint=str(checkpoint),global_step=13717,
        upload_checksums=hashes,validation=validation,status='uploaded; public verification pending',slurm_job_id=os.environ['SLURM_JOB_ID'])
    atomic_json(DIRECTORY/'upload.json',receipt);print('Uploaded revision',result.oid,flush=True)
    public=HfApi(token=False);remote=public.get_paths_info(MODEL,list(files),revision=result.oid)
    if len(remote)!=len(files):raise ValueError('Published file coverage incomplete')
    for item in remote:
        expected=hashes[item.path]
        if item.size!=expected['size']:raise ValueError('Published file size differs: '+item.path)
        if item.lfs:
            if item.lfs.sha256!=expected['sha256']:raise ValueError('Published LFS checksum differs: '+item.path)
        elif item.blob_id!=expected['git_sha1']:raise ValueError('Published Git blob differs: '+item.path)
    config=AutoConfig.from_pretrained(MODEL,revision=result.oid,token=False)
    remote_tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=result.oid,token=False)
    if config.model_type!='llama' or config.vocab_size!=128256 or remote_tokenizer(texts,add_special_tokens=False)!=tokenizer(texts,add_special_tokens=False):
        raise ValueError('Public config/tokenizer load differs')
    receipt.update(status='complete',public_unauthenticated_access=True,remote_file_checksums_verified=True,remote_config_tokenizer_load_verified=True,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    atomic_json(DIRECTORY/'upload.json',receipt);print(json.dumps({k:receipt[k] for k in ['model_id','revision','status','commit_url','public_unauthenticated_access']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('submit','status','worker'));parser.add_argument('--test-only',action='store_true');args=parser.parse_args()
    if args.action=='submit':submit(args.test_only)
    elif args.action=='status':status()
    else:worker()
