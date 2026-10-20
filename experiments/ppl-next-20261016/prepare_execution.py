"""Generate small immutable configs and metadata; no model/data processing."""
from pathlib import Path
import json,sys,copy,ast
live=Path(__file__).resolve().parent;source=live/'source';repo=live.parents[1]
sys.path.insert(0,str(source))
from training_state import atomic_json,load_config,digest,DEFAULT_ROOT
from ppl_contracts import *
if (live/'controller-state.json').exists():raise RuntimeError('Do not rewrite an active campaign')
base=json.loads((repo/'experiments/raw3b-bs64-wsd40-20261016/config.json').read_text())
base.update(category='pilot',sweep_id=STUDY,sweep_stage='medium',sweep_purpose='Two parallel mask trials, then PE-only/NorMuon-only on selected mask',max_h200_hours=16,max_steps=END,scheduler_steps=END,target_training_tokens=TARGET,max_training_tokens=TARGET,batch_switch_tokens=SWITCH*Q,learning_rate=ADAM_LR,eval_steps=400,parent_runs=['raw3b-bs64-wsd40-20261016'])
base['muon']['learning_rate']=MUON_LR
base['phase_schedule']=[dict(start=a,end=b,peak_ratio=r,warmup_steps=170,hold_schedule=base['hold_schedule']) for a,b,r in [(0,SPLIT,1.),(SPLIT,END,.1)]]
for stage in (1,2):
 for arm in (('dense','intradoc') if stage==1 else ('pe','nor')):
  for mask in ((arm,) if stage==1 else ('dense','intradoc')):
   cfg=copy.deepcopy(base);cfg['ppl']=dict(stage=stage,arm=arm,mask=mask,matrix='ns' if stage==1 else arm)
   path=live/'configs'/(f's{stage}-{arm}'+(f'-{mask}' if stage==2 else '')+'.json')
   atomic_json(path,cfg);load_config(path)
first=load_config(live/'configs/s1-dense.json');rows=batch_plan(first)
assert rows[-1]['end_tokens']==TARGET and all(r['width'] in (16,32,48,64) for r in rows)
assert len(rows)==count_updates(first)
atomic_json(live/'batch-plan.json',dict(updates=len(rows),input_tokens=TARGET,observer_knots=observer_knots(first),rows=rows))
old=json.loads((repo/'experiments/raw3b-bs64-wsd40-20261016/plan.json').read_text())
atomic_json(live/'execution-plan.json',dict(study=STUDY,stage1=['dense','intradoc'],stage2=['pe','nor'],stage2_mask='strict lower raw endpoint PPL winner from stage1',parallel_jobs_per_stage=2,gpus_per_job=8,cpus_per_job=8,max_concurrent_gpus=16,max_concurrent_cpus=16,seconds_per_job=7200,preflight_seconds_cap_inside_job=600,maximum_jobs=4,maximum_h200_hours=64,planned_updates=len(rows),target_inputs_per_run=TARGET,automatic_stage2=True,automatic_retry=False,training_checkpoint_load=False,excluded_nodes=old['excluded_nodes']))
ledger=json.loads((DEFAULT_ROOT/'experiment-ledger.json').read_text())
atomic_json(live/'ledger-before.json',dict(run_digests={k:v['config_digest'] for k,v in ledger['runs'].items()},measured_costs={k:v['h200_hours'] for k,v in ledger['jobs'].items() if 'h200_hours' in v},total_h200_hours_measured=ledger['total_h200_hours_measured']))
progress=json.loads((DEFAULT_ROOT/'preprocessed'/PREP/'raw/progress.json').read_text());prefix=[];n=0
for row in progress['train_files']:
 prefix.append(row);n+=row['rows']
 if n>=TARGET//8192:break
atomic_json(live/'raw-prefix.json',dict(files=prefix,rows=n,target_training_rows=TARGET//8192))
for p in source.rglob('*.py'):ast.parse(p.read_text())
print(json.dumps(dict(configurations=6,planned_updates=len(rows),observer_knots=observer_knots(first),source_python_files=len(list(source.rglob('*.py'))),published_prefix_rows=n)))
