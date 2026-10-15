"""Seal bounded recovery without changing any of the approved research configs."""
import ast
import copy
from pathlib import Path
import socket
import sys

SOURCE=Path(__file__).resolve().parent;LIVE=SOURCE.parent
sys.path.insert(0,str(SOURCE))
from training_state import DEFAULT_ROOT,atomic_json,read_json,file_digest,load_config


def definition(path,name):
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    if name=='build_optimizers':
        # Only the variant=True branch changed; native=False is identical.
        node=copy.deepcopy(node)
        for n in ast.walk(node):
            if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='variant':
                n.body=[ast.Pass()]
    return ast.dump(node,include_attributes=False)


def prepare():
    state_path=LIVE/'controller-state-r3.json'
    if state_path.exists() and 'normuon-check-r3' in read_json(state_path)['jobs']:
        raise ValueError('Cannot reseal submitted recovery')
    old=LIVE/'source-r1'
    unchanged=['train.py','optimizers.py','training_state.py','raw_baseline.py',
               'attention_backend.py','performance.py','recipe_runtime.py','batch1b_contracts.py']
    for name in unchanged:
        if file_digest(SOURCE/name)!=file_digest(old/name):raise ValueError('Native payload changed: '+name)
    definitions=['TokenScheduler','adamw_class','configure_update','verify_checkpoint_files',
                 'publish_snapshot','fork_checkpoint','build_optimizers']
    for name in definitions:
        if definition(SOURCE/'batch1b_runtime.py',name)!=definition(old/'batch1b_runtime.py',name):
            raise ValueError('Native runtime changed: '+name)
    evidence=read_json(LIVE/'validation-progress-r1.json')
    if evidence.get('native_pass') is not True or len(evidence['checks'])!=13:
        raise ValueError('Missing prior native validation success')
    evidence.update(unchanged_native_payload=True,
        native_files={name:file_digest(SOURCE/name) for name in unchanged},
        equivalent_runtime_definitions=definitions,
        original_failed_allocation_remains_failed=True)
    atomic_json(LIVE/'native-evidence-r3.json',evidence)
    cfg=read_json(LIVE/'configs/normuon.json')
    cfg.update(sweep_id='batch-normuon-1b-20261016-normuon-check-r3',sweep_stage='correctness-recovery',
        category='validation',world_size=1,microbatch_size=1,gradient_accumulation_steps=2,
        max_steps=100,scheduler_steps=100,warmup_steps=1,eval_steps=10,eval_documents=16,
        max_training_tokens=65536,target_training_tokens=65536,max_h200_hours=.26,
        stop_margin_seconds=120,wandb_mode='offline',export_on_completion=False,parent_runs=[])
    cfg.pop('phase_schedule')
    atomic_json(LIVE/'configs/normuon-check-r3.json',cfg)
    load_config(LIVE/'configs/normuon-check-r3.json')
    atomic_json(LIVE/'recovery-r3-authorization.json',dict(approved=True,
        approval_evidence='User twice explicitly requested 修正並繼續 after554476 failed',
        original_approval='Implement the plan:exactly nine1B endpoints and146.5H cap',
        correctness_gpus=1,correctness_cpus=8,correctness_seconds=900,
        training_seconds=7140,training_gpus=8,complete_gate_required=True,
        no_training_retry=True,no_cleanup_upload=True))
    ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
    plan=read_json(LIVE/'plan.json')
    plan.update(historical_jobs={k:v.get('h200_hours') for k,v in ledger['jobs'].items()},
        historical_run_digests={k:v['config_digest'] for k,v in ledger['runs'].items()},
        research_config_sha256={name:file_digest(LIVE/'configs'/f'{name}.json') for name in plan['endpoints']},
        recovery_validation_seconds=900,recovery_validation_gpus=1,
        worst_requested_total_h200_hours=sum(j.get('h200_hours',j['reserved_h200_hours'])
            for j in ledger['jobs'].values() if j['run_id'].startswith('batch-normuon-1b-20261016-'))+142.8+.25,
        complete_native_and_normuon_gate=True)
    atomic_json(LIVE/'plan-r3.json',plan)
    if not state_path.exists():
        previous=read_json(LIVE/'controller-state-r2.json')
        atomic_json(state_path,dict(stage='recovery_prepared',jobs=previous['jobs'],blocked_lanes={}))
    controller=SOURCE/'slurm/batch1b_controller.py'
    atomic_json(LIVE/'submission-owner-r3.json',dict(hostname=socket.gethostname().split('.')[0],
        script=str(controller),sha256=file_digest(controller)))
    files={str(p.relative_to(SOURCE)):file_digest(p) for p in SOURCE.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.sbatch','.txt','.json')}
    inputs=list((LIVE/'configs').glob('*.json'))+list((LIVE/'tests-r3').glob('*.py'))
    inputs += [LIVE/name for name in ('protocol.md','recovery-r3-protocol.md','plan.json','plan-r3.json',
        'submission-owner-r3.json','raw-prefix.json','recovery-r3-authorization.json','native-evidence-r3.json',
        'validation-progress-r1.json','source-seal-r1.json')]
    atomic_json(LIVE/'source-seal-r3.json',dict(files=files,inputs={str(p):file_digest(p) for p in inputs}))
    print('Sealed recovery;unchanged native payload;full validation gate retained;no submissions')


if __name__=='__main__':prepare()
