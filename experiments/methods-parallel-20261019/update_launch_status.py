"""Small read-only Slurm/run-metadata snapshot; writes local launch-status.json.

No submission, model loading, data processing or ledger mutation.
"""
import datetime,hashlib,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/work/asciibase64/Llama-3.2-1B-training')
def read(p):return json.loads(p.read_text())
def tail_json(p):
 if not p.exists():return []
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-32000));lines=f.read().decode(errors='replace').splitlines()
 values=[]
 for line in lines:
  try:values.append(json.loads(line))
  except json.JSONDecodeError:pass
 return values

def main():
 matrix=read(HERE/'matrix.json');ledger=read(ROOT/'experiment-ledger.json');items=[]
 for row in matrix['jobs']:
  receipt=read(ROOT/'runs/parallel-submissions'/HERE.name/(row['key']+'.json'))
  job=receipt.get('returned_job_id');run=ROOT/'runs/experiments'/row['run_id']
  item=dict(key=row['key'],job_id=job,run_id=row['run_id'],config_digest=receipt['config_digest'],
            submission_phase=receipt['phase'],requested_seconds=receipt['requested_seconds'],
            source_root=receipt.get('source_root',str(HERE/'source')),
            execution_seal_sha256=receipt['seal_sha256'])
  for name,field in [('ppl-gate.json','preflight'),('parallel-method-gate.json','method_gate')]:
   p=ROOT/'logs'/f'job-{job}'/name
   if p.exists():item[field]=dict(passed=read(p).get('passed'),path=str(p))
  for name,field in [('initialization.json','initialization'),('wandb.json','wandb')]:
   p=run/name
   if p.exists():item[field]=read(p)
  metrics=tail_json(run/'metrics.jsonl')
  train=[x for x in metrics if 'train/total_tokens_seen' in x]
  if train:
   item['last_training_metrics']={k:v for k,v in train[-1].items() if k in ('train/global_step','train/total_tokens_seen','train/loss','train/grad_norm','train/tokens_per_second','timestamp')}
  evals=[x for x in metrics if 'eval/perplexity' in x]
  if evals:item['last_evaluation']=evals[-1]
  allocations=[v for v in ledger['jobs'].values() if v['run_id']==row['run_id']]
  item['ledger_jobs']=[x.get('job_id') for x in allocations]
  item['measured_h200_hours']=sum(x.get('h200_hours',0) for x in allocations)
  item['reserved_or_measured_h200_hours']=sum(x.get('h200_hours',x['reserved_h200_hours']) for x in allocations)
  items.append(item)
 ids=','.join(x['job_id'] for x in items if x['job_id'])
 raw=subprocess.check_output(['sacct','-j',ids,'-X','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode,ElapsedRaw,Start,End,NodeList%50'],text=True,timeout=30)
 for line in raw.splitlines():
  f=line.split('|');item=next(x for x in items if x['job_id']==f[0])
  item.update(slurm_state=f[1],exit_code=f[2],elapsed_seconds=int(f[3]),start=f[4],end=f[5],node=f[6])
  item['phase']='training' if item.get('last_training_metrics') else ('initialization' if item.get('preflight',{}).get('passed') else 'preflight_or_startup')
  if f[1]!='RUNNING':item['phase']=f[1].lower()
 state=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=42,
  historical_control='ppl-next-20261017-B-seed42',original_cap_h200_hours=128,
  replacement_authorization=read(HERE/'startup-failures/replacement-authorization.json'),
  reserved_or_measured_h200_hours=sum(x['reserved_or_measured_h200_hours'] for x in items),
  initial_failed_h200_hours=read(HERE/'startup-failures/accounting.json')['total_failed_h200_hours'],
  source_seal_sha256=hashlib.sha256((HERE/'execution-seal.json').read_bytes()).hexdigest(),jobs=items)
 if (HERE/'dion-retry1/authorization.json').exists():state['dion_retry_authorization']=read(HERE/'dion-retry1/authorization.json')
 (HERE/'launch-status.json').write_text(json.dumps(state,indent=2)+'\n')
 for x in items:
  print(json.dumps({k:v for k,v in x.items() if k in ('key','job_id','slurm_state','node','phase','last_training_metrics','wandb')},ensure_ascii=False))
 print(json.dumps(dict(reserved_or_measured_h200_hours=state['reserved_or_measured_h200_hours'])))
if __name__=='__main__':main()
