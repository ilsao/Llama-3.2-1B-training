"""Create small immutable authorization, provenance and source-seal metadata."""
from pathlib import Path
import socket
import sys

SOURCE = Path(__file__).resolve().parent
LIVE = SOURCE.parent
sys.path.insert(0, str(SOURCE))
from training_state import DEFAULT_ROOT, atomic_json, digest, file_digest, load_config, read_json
from batch3b_contracts import STUDY, PREP, TARGET, Q, PLANNED_UPDATES, evaluation_units


def prepare():
    if (LIVE / 'controller-state.json').exists():
        raise ValueError('Never reseal a submitted or reserved experiment')
    cfg = load_config(LIVE / 'config.json')
    sha = digest(cfg)
    (SOURCE / 'batch3b_authorization.py').write_text(
        '"""Exact user-approved fresh formal13;one6h/48H allocation only."""\n'
        f'RUN_ID={STUDY!r}\nCONFIG_DIGEST={sha!r}\n\n'
        'def authorized(run_id,cfg,resume=False):\n'
        '    from training_state import digest\n'
        '    return run_id==RUN_ID and not resume and digest(cfg)==CONFIG_DIGEST\n')
    ledger = read_json(DEFAULT_ROOT / 'experiment-ledger.json')
    previous = sorted(k for k, v in ledger['runs'].items() if v['category'] == 'full')
    if len(previous) != 12 or STUDY in ledger['runs']:
        raise ValueError('Expected exactly twelve preserved prior formal runs')
    progress = read_json(DEFAULT_ROOT / 'preprocessed' / PREP / 'raw/progress.json')
    files = []
    rows = 0
    for item in progress['train_files']:
        files.append(item)
        rows += item['rows']
        if rows >= TARGET // 8192:
            break
    if rows < TARGET // 8192:
        raise ValueError('Insufficient immutable raw prefix')
    atomic_json(LIVE / 'raw-prefix.json', dict(files=files, required_rows=TARGET // 8192,
                                             published_rows=rows, preparation=PREP))
    atomic_json(LIVE / 'authorization.json', dict(approved=True, run_id=STUDY,
        configuration_digest=sha, formal_run_number=13,
        approval_evidence='User requested 幫我用 steps 後40% 跑一次3B and selected Warmup也按steps：每段170次實際optimizer updates',
        maximum_seconds=21600, requested_seconds=21540, maximum_h200_hours=48,
        gpus=8, cpus=8, single_allocation=True, random_weights_only=True,
        automatic_retry=False, resume=False, cleanup=False, hf_upload=False))
    old_plan = read_json(LIVE.parent / 'raw3b-sota-wsd20-20261015/plan.json')
    atomic_json(LIVE / 'plan.json', dict(run_id=STUDY, configuration_digest=sha,
        formal_run_number=13, previous_formal_runs=previous,
        historical_run_digests={k:v['config_digest'] for k,v in ledger['runs'].items()},
        historical_job_costs={k:v.get('h200_hours') for k,v in ledger['jobs'].items()},
        excluded_nodes=old_plan['excluded_nodes'], preparation=PREP,
        total_input_tokens=TARGET, reference_horizon_units=22889, reference_unit_tokens=Q,
        actual_optimizer_updates=PLANNED_UPDATES, first_phase_updates=13355, second_phase_updates=957,
        decay_clock='optimizer_updates', warmup_clock='optimizer_updates',
        phase_step_knots=[(8014,11575,12465,13355),(13929,14184,14248,14312)],
        evaluation_units=evaluation_units(),
        switch_tokens=11444 * Q, requested_seconds=21540, maximum_seconds=21600,
        maximum_h200_hours=48, gpus=8, cpus=8, full_evaluation_documents=50000,
        full_evaluation_prediction_tokens=76930208, automatic_retry=False, automatic_resume=False))
    controller = SOURCE / 'slurm/batch3b_controller.py'
    atomic_json(LIVE / 'submission-owner.json', dict(hostname=socket.gethostname().split('.')[0],
        script=str(controller), sha256=file_digest(controller)))
    inputs = [LIVE / name for name in ('config.json', 'protocol.md', 'authorization.json',
                                      'plan.json', 'raw-prefix.json', 'submission-owner.json')]
    inputs += list((LIVE / 'tests').glob('*.py'))
    inputs += [LIVE.parent / 'lr-transfer-horizons-20261015/calibration-result.json',
               LIVE.parent / 'raw3b-bs64-wsd40-20261016/receipt.json',
               LIVE.parent / 'raw3b-sota-20261015/receipt.json',
               LIVE.parent / 'raw3b-sota-wsd20-20261015/receipt.json',
               LIVE.parent / 'batch-normuon-1b-20261016/receipts/bs-50-fixed.json']
    atomic_json(LIVE / 'source-seal.json', dict(
        files={str(p.relative_to(SOURCE)):file_digest(p) for p in SOURCE.rglob('*')
               if p.is_file() and '__pycache__' not in p.parts},
        inputs={str(p):file_digest(p) for p in inputs}))
    print('Prepared formal13,config digest', sha, ';no jobs submitted')


if __name__ == '__main__':
    prepare()
