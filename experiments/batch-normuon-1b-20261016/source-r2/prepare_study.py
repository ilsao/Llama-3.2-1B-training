"""Lightweight creation/sealing of the exact approved experiment metadata."""
from pathlib import Path
import socket
import sys

SOURCE = Path(__file__).resolve().parent
LIVE = SOURCE.parent
ROOT = LIVE.parents[1]
sys.path.insert(0, str(SOURCE))
from training_state import DEFAULT_ROOT, atomic_json, file_digest, load_config, read_json
from batch1b_contracts import STUDY, TARGET, build_configs, count_updates, Q


def prepare():
    if (LIVE/'controller-state.json').exists():
        raise ValueError('Cannot reseal a study after any submission')
    parent = ROOT/'experiments/raw3b-sota-wsd20-20261015/config.json'
    calibration = ROOT/'experiments/lr-transfer-horizons-20261015/calibration-result.json'
    configs = build_configs(read_json(parent))
    for name, cfg in configs.items():
        atomic_json(LIVE/'configs'/f'{name}.json', cfg)
        load_config(LIVE/'configs'/f'{name}.json')
    validation = dict(configs['control'])
    validation.update(category='validation', sweep_stage='validation', snapshot_tokens=[],
                      wandb_mode='offline', export_on_completion=False)
    atomic_json(LIVE/'configs/validation.json', validation)
    setup = dict(validation)
    setup.update(sweep_id=STUDY+'-setup', world_size=1, max_h200_hours=.5,
                 sweep_stage='dependency-setup', stop_margin_seconds=60)
    setup.pop('phase_schedule')
    atomic_json(LIVE/'configs/setup.json', setup)
    for name in ('validation','setup'):
        load_config(LIVE/'configs'/f'{name}.json')
    (LIVE/'receipts').mkdir(exist_ok=True)
    ledger = read_json(DEFAULT_ROOT/'experiment-ledger.json')
    progress = read_json(DEFAULT_ROOT/'preprocessed/dolma-raw-baseline-20261008/raw/progress.json')
    files, rows = [], 0
    for item in progress['train_files']:
        files.append(item)
        rows += item['rows']
        if rows >= TARGET//8192:
            break
    if rows < TARGET//8192:
        raise ValueError('Insufficient published raw prefix')
    atomic_json(LIVE/'raw-prefix.json', dict(files=files, required_rows=TARGET//8192))
    plan = dict(study=STUDY, endpoints=list(configs),
                physical_training_tokens=5267259392, reservation_limit_h200_hours=146.5,
                historical_jobs={key:value.get('h200_hours') for key,value in ledger['jobs'].items()},
                historical_run_digests={key:value['config_digest'] for key,value in ledger['runs'].items()},
                excluded_nodes=read_json(ROOT/'experiments/raw3b-sota-wsd20-20261015/plan.json')['excluded_nodes'],
                approved_by='User: Implement the plan.', initialization='random residual_half seed42',
                no_automatic_retry_resume=True, no_cleanup_upload=True,
                branch_updates={name:dict(inherited_updates=cfg.get('fork_snapshot_tokens',0)//Q,
                    remaining_updates=count_updates(cfg,cfg.get('fork_snapshot_tokens',0)))
                    for name,cfg in configs.items()})
    atomic_json(LIVE/'plan.json', plan)
    controller = SOURCE/'slurm/batch1b_controller.py'
    atomic_json(LIVE/'submission-owner.json', dict(hostname=socket.gethostname().split('.')[0],
                script=str(controller), sha256=file_digest(controller)))
    inputs = [LIVE/'protocol.md', LIVE/'plan.json', LIVE/'submission-owner.json',
              LIVE/'raw-prefix.json', parent, calibration, *sorted((LIVE/'configs').glob('*.json')),
              *sorted((LIVE/'tests').glob('*.py'))]
    source_files = [p for p in SOURCE.rglob('*') if p.is_file() and
                    '__pycache__' not in p.parts and p.suffix in ('.py','.sh','.sbatch','.txt','.json')]
    atomic_json(LIVE/'source-seal.json', dict(files={str(p.relative_to(SOURCE)):file_digest(p) for p in source_files},
                inputs={str(p):file_digest(p) for p in inputs}))
    print('Prepared',STUDY,'endpoints',len(configs),'sealed source files',len(source_files))
    for name, row in plan['branch_updates'].items():
        print(name,row)


if __name__ == '__main__':
    prepare()
