#!/usr/bin/env python3
"""Allocated-node-only pinned teacher download and tokenizer equivalence proof."""
import os
if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
    raise SystemExit('Use sbatch')
import json,sys,hashlib,time
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent
proposal=json.loads((HERE.parent/'pretrained-teacher-proposal.json').read_text())
spec=proposal['teacher']; root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=False)
def fetch(name):
    if time.time()>int(os.environ['TRAIN_END_EPOCH'])-60:raise RuntimeError('Download stopping margin')
    return Path(hf_hub_download(spec['model_id'],name,revision=spec['revision'],local_dir=root))
index=json.loads(fetch('model.safetensors.index.json').read_text())
names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']
names+=sorted(set(index['weight_map'].values()))
for name in names:
    fetch(name);print('Prepared '+name,flush=True)
teacher=AutoTokenizer.from_pretrained(root,local_files_only=True,use_fast=True)
student_path=Path('/work/asciibase64/Llama-3.2-1B-training/preprocessed/dolma-raw-baseline-20261008/raw/tokenizer')
student=AutoTokenizer.from_pretrained(student_path,local_files_only=True,use_fast=True)
assert teacher.get_vocab()==student.get_vocab(),'Token ID maps differ'
assert len(teacher)==len(student)==128256
for attr in ['bos_token_id','eos_token_id','pad_token_id','add_bos_token','add_eos_token']:
    assert getattr(teacher,attr,None)==getattr(student,attr,None),attr
tb=json.loads(teacher.backend_tokenizer.to_str());sb=json.loads(student.backend_tokenizer.to_str())
for key in ['model','normalizer','pre_tokenizer','post_processor','decoder','added_tokens']:
    assert tb.get(key)==sb.get(key),'Tokenizer backend differs: '+key
probes=['Hello world!','繁體中文測試。',' a\n\n b\t café','1234567890','<|begin_of_text|>x<|end_of_text|>','🦙\u200d💻']
for text in probes:
    for special in [False,True]:assert teacher.encode(text,add_special_tokens=special)==student.encode(text,add_special_tokens=special)
files={}
for name in names+['model.safetensors.index.json']:
    path=root/name;h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024*1024),b''):h.update(data)
    files[name]={'size':path.stat().st_size,'sha256':h.hexdigest()}
receipt={'model_id':spec['model_id'],'revision':spec['revision'],'root':str(root),'files':files,
    'equivalent_tokenization_verified':True,'tokenizer_checks':'full vocabulary and backend components, special-token behavior, multilingual probes',
    'student_tokenizer_directory':str(student_path),'job_id':os.environ['SLURM_JOB_ID'],'weights_loaded':False}
(root/'teacher-manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Pinned teacher and tokenizer verification complete',flush=True)
