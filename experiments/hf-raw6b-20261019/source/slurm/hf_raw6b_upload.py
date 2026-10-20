#!/usr/bin/env python3
"""Publish the verified best4B export with pinned parent HEAD and public checks."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
SOURCE=Path(__file__).resolve().parents[1];LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,digest,file_digest,locked,load_config,read_json
from slurm.submission import find_submission_job
RUN='hf-raw6b-20261019';PARENT='raw6b-normuon-adam2x-20261018';MODEL='asciibase64/Llama-3.2-1B-dolma3_mix'
DIRECTORY=DEFAULT_ROOT/'runs/publications'/RUN
AUDIT=LIVE.parent/PARENT/'audited-results.json'


def now():return dt.datetime.now(dt.timezone.utc).isoformat()


def artifact_hashes(path):
    size=path.stat().st_size;sha=hashlib.sha256();git=hashlib.sha1(b'blob '+str(size).encode()+b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block);git.update(block)
    return dict(size=size,sha256=sha.hexdigest(),git_sha1=git.hexdigest())


def check(require_public_approval=False):
    import math
    plan=read_json(LIVE/'plan.json')
    if plan['model_id']!=MODEL or plan['parent_run']!=PARENT or plan['status']!='approved_for_submission':raise ValueError('Publication not approved')
    seal=read_json(LIVE/'source-seal.json')
    for f,h in seal['files'].items():
        if file_digest(f)!=h:raise ValueError('Publication source/payload changed: '+f)
    for f,h in plan['immutable_inputs'].items():
        if file_digest(f)!=h:raise ValueError('Pinned publication input changed: '+f)
    parent=DEFAULT_ROOT/'runs/experiments'/PARENT
    summary=read_json(parent/'summary.json');extra=read_json(parent/'confirmation-evaluation.json');export=read_json(parent/'export.json')
    result=read_json(LIVE.parent/PARENT/'results.json')
    if (summary['global_step']!=28619 or summary['attempted_updates']!=28619 or
        summary['total_tokens_seen']!=6000082944 or summary['budget_tokens_charged']!=6000082944 or
        summary['stop_reason']!='training_token_target' or summary['checkpoint_loaded_for_training'] or
        summary['initialization']['pretrained_weights_loaded'] or summary['export_global_step']!=28619 or
        summary['export']!=export['path'] or not export['reload_verified'] or
        extra['global_step']!=28619 or extra['export']!=export['path'] or extra['eval/documents']!=9000 or
        extra['eval/prediction_tokens']!=13666812 or extra['evaluation_slots']!=[1000,10000] or
        result['allocation']['state']!='COMPLETED' or result['h200_hours']>80 or
        not result['schedule_audit']['passed']):raise ValueError('6B endpoint verification incomplete')
    for name,expected in plan['export_file_stats'].items():
        p=Path(plan['export'])/name
        if dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)!=expected:raise ValueError('Export changed: '+name)


def progress(stage):
    atomic_json(DIRECTORY/'progress.json',dict(stage=stage,timestamp=now()));print(stage,flush=True)


def submit():
    DIRECTORY.mkdir(parents=True,exist_ok=True)
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        check(require_public_approval=True)
        if (DIRECTORY/'launch.json').exists():raise ValueError('Existing submission; recover/status rather than retry')
        cfg=load_config(LIVE/'validation-config.json');tag=RUN+'-'+uuid.uuid4().hex
        if cfg['category']!='validation' or cfg['world_size']!=1 or cfg['max_h200_hours']!=.5:raise ValueError('Publication uses1H200/30min only')
        ledger=ExperimentLedger(DEFAULT_ROOT);ledger.reserve(tag,RUN,cfg,1800,1,False)
        command=['sbatch','--parsable','--comment='+tag,str(SOURCE/'slurm/hf-raw6b-upload.sbatch'),RUN]
        state=dict(status='submission_uncertain',reservation=tag,submission_tag=tag,submission_started_utc=now(),submission_command=command)
        atomic_json(DIRECTORY/'launch.json',state)
        with ledger.transaction() as d:d['jobs'][tag].update(submission_tag=tag,submission_started_utc=state['submission_started_utc'],submission_command=command,allocation_kind='validation_and_publication',parent_run=PARENT,authorization='User explicitly requested uploading the current best model to HF; use existing public repository and preserve Git history')
        env={**os.environ,'TRAIN_SOURCE_ROOT':str(SOURCE),'DATA_ROOT':str(DEFAULT_ROOT)}
        try:r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=30)
        except (OSError,subprocess.TimeoutExpired) as error:state['submission_error']=str(error);atomic_json(DIRECTORY/'launch.json',state);return
        state['submission_result']=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr);job=r.stdout.strip().split(';')[0]
        if r.returncode==0 and job.isdigit():
            with ledger.transaction() as d:d['jobs'][tag]['returned_job_id']=job
            ledger.bind(tag,job);state.update(job_id=job,status='submitted')
        atomic_json(DIRECTORY/'launch.json',state);print(json.dumps(state))


def write_public_metadata(stage):
    import shutil
    stage.mkdir(parents=True,exist_ok=True)
    for p in (LIVE/'payload-preview').iterdir():
        if p.is_file():shutil.copy2(p,stage/p.name)


def prepare_preview():
    stage=LIVE/'payload-preview';write_public_metadata(stage)
    export=Path(read_json(LIVE/'plan.json')['export']);files={f.name:f for f in export.iterdir() if f.is_file()};files.update({f.name:f for f in stage.iterdir() if f.is_file()})
    manifest=dict(model_id=MODEL,files={name:dict(size=f.stat().st_size,source=str(f),**({'sha256':file_digest(f)} if f.parent==stage or f.stat().st_size<100000 else {})) for name,f in files.items()},status='prepared from explicit upload request')
    atomic_json(LIVE/'payload-manifest.json',manifest)
    print(json.dumps(dict(model_id=MODEL,file_count=len(files),total_bytes=sum(f.stat().st_size for f in files.values()),files=list(sorted(files))),indent=2))


def worker():
    from dataset import require_compute
    require_compute();check(require_public_approval=True)
    if (DIRECTORY/'upload.json').exists() or (DIRECTORY/'upload-intent.json').exists():raise ValueError('Existing publication intent; verify/recover instead of creating another commit')
    cfg=load_config(LIVE/'validation-config.json');allocation=ExperimentLedger(DEFAULT_ROOT).allocation(os.environ['SLURM_JOB_ID'],RUN)
    if allocation['config_digest']!=digest(cfg):raise ValueError('Publication allocation binding changed')
    from transformers import AutoConfig,AutoTokenizer
    from huggingface_hub import HfApi,CommitOperationAdd,hf_hub_url
    import requests
    plan=read_json(LIVE/'plan.json');parent=DEFAULT_ROOT/'runs/experiments'/PARENT;export=Path(plan['export'])
    summary=read_json(parent/'summary.json');provenance=read_json(parent/'data-provenance.json')
    config=AutoConfig.from_pretrained(export,local_files_only=True);tokenizer=AutoTokenizer.from_pretrained(export,local_files_only=True)
    original=AutoTokenizer.from_pretrained(parent/'assets',local_files_only=True)
    texts=['Training from random initialization.','資料驗證與模型訓練。','def square(x):\n    return x * x']
    if tokenizer(texts,add_special_tokens=False)!=original(texts,add_special_tokens=False):raise ValueError('Export tokenizer differs')
    if config.model_type!='llama' or config.vocab_size!=128256:raise ValueError('Export architecture differs')
    index=read_json(export/'model.safetensors.index.json')['weight_map']
    if not index or any(not (export/name).is_file() for name in set(index.values())):raise ValueError('Incomplete weight index')
    stage=DIRECTORY/'metadata';write_public_metadata(stage)
    files={f.name:f for f in export.iterdir() if f.is_file()};files.update({f.name:f for f in stage.iterdir() if f.is_file()})
    allowed={'config.json','generation_config.json','model.safetensors.index.json','model-00001-of-00003.safetensors','model-00002-of-00003.safetensors','model-00003-of-00003.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','README.md','resolved-training-config.json','training-details.json','training-metadata.json'}
    if set(files)!=allowed:raise ValueError('Unexpected public payload file set')
    progress('hashing_verified_export');hashes={name:artifact_hashes(path) for name,path in files.items()}
    approved=read_json(LIVE/'payload-manifest.json')
    if approved['model_id']!=MODEL or set(approved['files'])!=set(files):raise ValueError('Payload file list differs from reviewed preview')
    for name,expected in approved['files'].items():
        if hashes[name]['size']!=expected['size'] or ('sha256' in expected and hashes[name]['sha256']!=expected['sha256']):raise ValueError('Payload differs from reviewed preview: '+name)
    atomic_json(DIRECTORY/'payload-checksums.json',hashes)
    api=HfApi();info=api.model_info(MODEL)
    if info.sha!=plan['expected_parent_commit'] or info.private or info.gated:raise ValueError('Remote parent HEAD/visibility changed; no upload made')
    atomic_json(DIRECTORY/'upload-intent.json',dict(model_id=MODEL,expected_parent=info.sha,files=hashes,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=now()))
    progress('uploading')
    result=api.create_commit(MODEL,repo_type='model',operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)) for name,path in files.items()],
        parent_commit=info.sha,num_threads=4,commit_message='Upload6B intradoc NorMuon checkpoint:10k PPL12.90785,step28619')
    receipt=dict(model_id=MODEL,revision=result.oid,commit_url=result.commit_url,parent_revision=info.sha,parent_run=PARENT,global_step=28619,training_tokens=6000082944,
        combined_10k_perplexity=12.9078477387367,status='uploaded; verification pending',slurm_job_id=os.environ['SLURM_JOB_ID'],local_export=str(export))
    atomic_json(DIRECTORY/'upload.json',receipt);progress('verifying_public_revision')
    public=HfApi(token=False);remote=public.get_paths_info(MODEL,list(files),revision=result.oid)
    if len(remote)!=len(files):raise ValueError('Remote file coverage differs')
    for item in remote:
        expected=hashes[item.path]
        if item.size!=expected['size']:raise ValueError('Remote file size mismatch: '+item.path)
        if item.lfs:
            if item.lfs.sha256!=expected['sha256']:raise ValueError('Remote LFS SHA256 mismatch: '+item.path)
        elif item.blob_id!=expected['git_sha1']:raise ValueError('Remote Git blob checksum mismatch: '+item.path)
    rc=AutoConfig.from_pretrained(MODEL,revision=result.oid,token=False);rt=AutoTokenizer.from_pretrained(MODEL,revision=result.oid,token=False)
    lc,pc=config.to_dict(),rc.to_dict()
    for key in ['_name_or_path','_commit_hash']:lc.pop(key,None);pc.pop(key,None)
    if lc!=pc or tokenizer(texts,add_special_tokens=False)!=rt(texts,add_special_tokens=False):raise ValueError('Public config/tokenizer reload mismatch')
    for name in set(index.values()):
        response=requests.head(hf_hub_url(MODEL,name,revision=result.oid),allow_redirects=True,timeout=60);response.raise_for_status()
        if response.status_code!=200:raise ValueError('Public weight access unavailable')
    latest=public.model_info(MODEL,revision=result.oid)
    if latest.private or latest.gated or latest.sha!=result.oid:raise ValueError('Public revision inaccessible')
    receipt.update(status='complete',remote_file_checksums_verified=True,remote_config_tokenizer_load_verified=True,public_unauthenticated_access=True,public_weight_shard_access_verified=True,finished_utc=now())
    atomic_json(DIRECTORY/'upload.json',receipt);progress('complete');print(json.dumps(receipt),flush=True)


def status():
    with locked(DIRECTORY/'.submission.lock',blocking=False):
        s=read_json(DIRECTORY/'launch.json');ledger=ExperimentLedger(DEFAULT_ROOT)
        if not s.get('job_id'):s['job_id']=find_submission_job(s,DEFAULT_ROOT);ledger.bind(s['reservation'],s['job_id'])
        if s['status']!='reconciled':
            r=subprocess.run(['sacct','-X','-j',s['job_id'],'-n','-P','-o','JobIDRaw,State'],capture_output=True,text=True,timeout=20,check=True)
            states=[x.split('|')[1].split()[0].rstrip('+') for x in r.stdout.splitlines() if x.split('|')[0]==s['job_id']];s['slurm_state']=states[0] if states else 'UNKNOWN'
            if s['slurm_state'] in TERMINAL:s.update(accounting=ledger.reconcile(s['job_id']),status='reconciled')
            atomic_json(DIRECTORY/'launch.json',s)
        print(json.dumps(s))
        for name in ['progress.json','upload.json']:
            p=DIRECTORY/name
            if p.exists():print(name,json.dumps(read_json(p)))

if __name__=='__main__':
    {'submit':submit,'worker':worker,'status':status,'check':check,'preview':prepare_preview}[sys.argv[1]]()
