"""Read small run metadata only; no model/data loading or bootstrap on login."""
import hashlib,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/work/asciibase64/Llama-3.2-1B-training')
def read(path):return json.loads(path.read_text())
def main():
 from control_reference import validate
 reference=validate()
 matrix=read(HERE/'matrix.json');results={reference['key']:{**reference,'status':'historical_reference'}}
 ledger=read(ROOT/'experiment-ledger.json')
 for row in matrix['jobs']:
  run=ROOT/'runs/experiments'/row['run_id'];record={'key':row['key'],'status':'not_started'}
  allocations=[v for v in ledger['jobs'].values() if v.get('run_id')==row['run_id']]
  if allocations:
   record.update(status='submitted_without_summary',allocations=allocations)
   if all('h200_hours' in v for v in allocations):
    record.update(allocation_accounting='reconciled',h200_hours=sum(v['h200_hours'] for v in allocations))
   else:record['allocation_accounting']='not_reconciled'
  if (run/'summary.json').exists():
   summary=read(run/'summary.json')
   expected=hashlib.sha256(json.dumps(read(HERE/row['config']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
   assert summary['config_digest']==expected
   record.update(status='incomplete',inputs=summary['total_tokens_seen'],
     steps=summary['global_step'],stop_reason=summary['stop_reason'],wandb_url=summary.get('wandb_url'))
   if summary.get('training_token_target_reached') and summary['total_tokens_seen']==1000079360 and summary['global_step']==4775:
    record['status']='endpoint_without_confirmation'
    if (run/'confirmation-evaluation.json').exists():
     c=read(run/'confirmation-evaluation.json');s=summary['last_evaluation']
     assert c['global_step']==summary['global_step']==summary['last_eval_step']
     assert c['export']==summary['export']
     assert c['preparation_fingerprint']==summary['preparation_fingerprint']
     assert s['eval/prediction_tokens']==1479302 and c['eval/prediction_tokens']==13666812
     record.update(status='endpoint_and_confirmation_measured',nll_1k=s['eval/nll']/s['eval/prediction_tokens'],
       nll_9k=c['eval/nll']/c['eval/prediction_tokens'],
       ppl_10k=math.exp((s['eval/nll']+c['eval/nll'])/(s['eval/prediction_tokens']+c['eval/prediction_tokens'])))
   record.setdefault('allocation_accounting','not_reconciled')
   if (run/'resource-accounting.json').exists():
    cost=read(run/'resource-accounting.json')
    if cost.get('all_jobs_reconciled'):
     record['allocation_accounting']='reconciled';record['h200_hours']=cost['h200_hours_measured']
  results[row['key']]=record
 contrasts=[]
 for arm,control in matrix['comparisons']:
  for seed in matrix['seeds']:
   a,b=results[f'{arm}-seed{seed}'],results[f'{control}-seed{seed}']
   if 'nll_9k' in a and 'nll_9k' in b:
    contrasts.append(dict(arm=arm,control=control,seed=seed,delta_nll_9k=a['nll_9k']-b['nll_9k'],
       interpretation='point estimate conditional on seed42; historical-control comparisons are not contemporaneous; no seed-variance estimate'))
 print(json.dumps(dict(results=results,contrasts=contrasts),indent=2))
if __name__=='__main__':main()
