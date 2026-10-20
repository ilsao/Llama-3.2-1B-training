"""Build only small, locked experiment files. Never submits or loads a model."""
import copy,csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path('/work/asciibase64/Llama-3.2-1B-training')
ARMS={
 'N0':('native','Current intradoc NS5 NorMuon + AdamW x2 control'),
 'D1':('dion1','Official Dion3 full fraction=1 bridge; fixed NS5'),
 'D025':('dion025','Official Dion3 fraction=.25; fixed NS5; LR x2 vs D1'),
 'A1':('adamuon','AdaMuon v3 sign / elementwise adaptivity; replicated implementation'),
 'S1':('soap','Full SOAP matrix preconditioning, frequency 10'),
 'K1':('kd','Frozen existing 6B-trained teacher, .75 CE + .25 forward KL, T=1'),
 'F1':('sf','SF-NorMuon + SF-AdamW with averaged-weight feedback'),
 'F2':('observer','Same inner updates and constant LR, average for eval only'),
 'C1':('cce','Exact CCE loss kernel, same N0 optimizer and schedule'),
}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 base=json.loads((REPO/'experiments/ppl-next-20261017/configs/B-seed42.json').read_text())
 teacher_root=ROOT/'runs/experiments/raw6b-normuon-adam2x-20261018/export-step-28619-2c5a9317'
 receipt=ROOT/'runs/publications/hf-raw6b-20261019/payload-checksums.json'
 hashes=json.loads(receipt.read_text())
 files={k:v for k,v in hashes.items() if k!='training-metadata.json' and (teacher_root/k).exists()}
 # Metadata was sanitized for publication; pin the original local record separately.
 p=teacher_root/'training-metadata.json'
 files[p.name]={'size':p.stat().st_size,'sha256':sha(p)}
 teacher={'export':str(teacher_root),'files':files,'model_id':'asciibase64/Llama-3.2-1B-dolma3_mix',
          'revision':'13125ad091ca25c914e1528af6dce22d5f6efa3a',
          'historical_training_tokens':6000082944,'historical_training_h200_hours':64.70666666666668}
 write(HERE/'source/teacher-manifest.json',teacher)
 teacher_spec={'export':str(teacher_root),'manifest_sha256':sha(HERE/'source/teacher-manifest.json')}
 locked={};jobs=[]
 for arm,(method,description) in ARMS.items():
  for seed in (42,):
   key=f'{arm}-seed{seed}';cfg=copy.deepcopy(base)
   cfg.update(seed=seed,sweep_id=HERE.name,sweep_stage='independent',sweep_purpose=description,
              parent_runs=['ppl-next-20261017-B-seed42'],
              parallel_method={'name':method,'fixed_recipe':True,'training_checkpoint_load':False})
   cfg['ppl']['arm']=arm
   if method=='kd':cfg['parallel_method'].update(teacher=teacher_spec,teacher_weight=.25,temperature=1.,logit_chunk=128)
   if method in ('sf','observer'):
    cfg['parallel_method'].update(feedback_beta=.9,average_weight='lr_squared',evaluation_weights='average_x',
             schedule='single token warmup then constant; no phase restart or cooldown',matrix_momentum=.95)
   locked[key]=cfg
   write(HERE/'configs'/f'{key}.json',cfg)
   if arm=='N0':continue  # Historical results supply the control; this config is reference-only.
   jobs.append(dict(key=key,arm=arm,seed=seed,method=method,run_id=HERE.name+'-'+key.lower(),
                    config='configs/'+key+'.json',gpus=8,cpus=8,seconds=7200,max_h200_hours=16,
                    training_inputs=1000079360,updates=4775,depends_on=[],
                    teacher_forward_inputs=1000079360 if method=='kd' else 0))
 write(HERE/'source/locked-configs.json',locked)
 from control_reference import build
 reference=build(locked['N0-seed42'])
 write(HERE/'reference-control.json',reference)
 write(HERE/'matrix.json',dict(study=HERE.name,status='prepared_not_submitted',jobs=jobs,seeds=[42],references=[reference],
      preparation='dolma-raw-baseline-20261008',max_h200_hours_all=128,max_h200_hours_one_seed=128,
      max_simultaneous_gpus_all=64,max_simultaneous_cpus_all=64,
      max_simultaneous_gpus_one_seed=64,max_simultaneous_cpus_one_seed=64,
      allocation_includes=['startup','numerical_gate','training','evaluation','checkpoint','export','idle'],
      job_dependency_policy='none; independent sbatch jobs, no promotion/resume/controller',
      resource_note='Per-job limit is 8 GPUs / 8 CPUs. No aggregate concurrency throttle is installed.',
      comparisons=[['D1','N0'],['D025','D1'],['A1','N0'],['S1','N0'],['K1','N0'],['F1','F2'],['F2','N0'],['C1','N0']]))
 with (HERE/'budget.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['key','gpus','cpus','seconds','max_h200_hours','training_inputs','teacher_forward_inputs'])
  writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in jobs)
 print(json.dumps({'jobs':len(jobs),'h200_hours_cap':128,'submitted':False}))
if __name__=='__main__':main()
