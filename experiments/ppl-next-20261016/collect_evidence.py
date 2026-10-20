"""Read small training metadata only; no torch, model, dataset or submissions.

Run from the repository root with Python's standard library. Outputs are confined
to this report directory; original runs, controllers and ledgers are read-only.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import shutil
import subprocess
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
NAMES = (
    'raw-filtered-3b-20261015-raw',
    'raw3b-sota-20261015',
    'raw3b-sota-wsd20-20261015',
    'raw3b-bs64-wsd40-20261016',
    'raw3b-bs64-stepwsd40-20261016',
)


def read(path):
    return json.loads(path.read_text())


def seal(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def collect():
    ledger_path = ROOT / 'experiment-ledger.json'
    ledger = read(ledger_path)
    result = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
                  ledger=seal(ledger_path),
                  measured_h200_hours=ledger['total_h200_hours_measured'],
                  reserved_or_measured_h200_hours=ledger['total_h200_hours_reserved_or_measured'],
                  formal_run_count=sum(v['category'] == 'full' for v in ledger['runs'].values()),
                  runs=[], curves=[], historical_recipe=[])
    for name in NAMES:
        directory = ROOT / 'runs/experiments' / name
        summary = read(directory / 'summary.json')
        full = read(directory / 'full-evaluation.json')
        cfg = summary['configuration']
        with (directory / 'metrics.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream]
        evaluations = [x for x in rows if 'eval/perplexity' in x]
        training = [x for x in rows if 'train/grad_norm' in x]
        jobs = [v for v in ledger['jobs'].values() if v['run_id'] == name]
        row = dict(run_id=name, job_id=summary['slurm_job_id'],
                   tokens=summary['total_tokens_seen'], updates=summary['global_step'],
                   allocation_h200_hours=sum(v.get('h200_hours', 0) or 0 for v in jobs),
                   full_ppl=full['eval/perplexity'],
                   full_mean_nll=full['eval/nll'] / full['eval/prediction_tokens'],
                   full_documents=full['eval/documents'],
                   full_prediction_tokens=full['eval/prediction_tokens'],
                   subset_ppl=summary['last_evaluation']['eval/perplexity'],
                   tail_ratio=cfg['wsd_decay_ratio'],
                   phase_schedule=cfg.get('phase_schedule'),
                   schedule_clock=cfg.get('schedule_clock', 'optimizer_updates'),
                   warmup_clock=cfg.get('warmup_clock', cfg.get('schedule_clock', 'optimizer_updates')),
                   muon_lr=cfg['muon']['learning_rate'], adamw_lr=cfg['learning_rate'],
                   checkpoint_loaded=summary['checkpoint_loaded_for_training'],
                   initialization_signature=summary['initialization']['sample_signature'],
                   preparation_fingerprint=summary['preparation_fingerprint'],
                   stop_reason=summary['stop_reason'], wandb=summary['wandb_url'],
                   full_length_buckets={k:v for k,v in full.items() if k.startswith('eval/length_')},
                   sampled_training_windows=[],
                   sources=[seal(directory / p) for p in ('summary.json', 'metrics.jsonl', 'full-evaluation.json')])
        for low, high in [(0, 1), (1, 1.5), (1.5, 2.5), (2.5, 3.1)]:
            part = [x for x in training if low * 1e9 <= x['train/total_tokens_seen'] < high * 1e9]
            row['sampled_training_windows'].append(dict(
                start_b=low, end_b=high, logged_samples=len(part),
                median_preclip_norm=statistics.median(x['train/grad_norm'] for x in part),
                fraction_samples_over_clip=sum(x['train/grad_norm'] > cfg['max_grad_norm'] for x in part)/len(part),
                median_tokens_per_second=statistics.median(x['train/tokens_per_second'] for x in part)))
        result['runs'].append(row)
        result['curves'].extend(dict(run_id=name, step=x['train/global_step'],
                                     tokens=x['eval/total_training_tokens_seen'],
                                     ppl=x['eval/perplexity'],
                                     mean_nll=x['eval/nll']/x['eval/prediction_tokens']) for x in evaluations)
    batch = read(REPO / 'experiments/batch-normuon-1b-20261016/results.json')
    fields = ('job_id', 'tokens', 'global_step', 'mean_nll', 'perplexity', 'delta_mean_nll_vs_control', 'wandb_url')
    result['batch1b'] = [dict(candidate=k, **{f:v[f] for f in fields}) for k,v in batch['endpoints'].items()]
    state = read(REPO / 'experiments/init-recipe-20261014/controller-state.json')
    for name in ('short-baseline', 'promote-adam20', 'promote-muon13', 'combo-merged-reranked',
                 'seed43-baseline', 'seed43-candidate', 'filtered', 'batch8', 'batch32',
                 'wd003', 'wd005', 'clip-late', 'clip-groups'):
        endpoint = state['jobs'][name]['endpoint']
        ev = endpoint['evaluation']
        result['historical_recipe'].append(dict(candidate=name, tokens=endpoint['tokens'],
            ppl=ev['eval/perplexity'], mean_nll=ev['eval/nll']/ev['eval/prediction_tokens']))
    best = next(x for x in result['runs'] if x['run_id'] == 'raw3b-bs64-wsd40-20261016')
    sota = next(x for x in result['runs'] if x['run_id'] == 'raw3b-sota-20261015')
    result['comparisons_vs_best'] = [dict(run_id=x['run_id'],
        delta_nll=x['full_mean_nll']-best['full_mean_nll'],
        relative_ppl_percent=100*(x['full_ppl']/best['full_ppl']-1)) for x in result['runs']]
    result['best_vs_sota'] = dict(relative_ppl_reduction_percent=100*(1-best['full_ppl']/sota['full_ppl']),
        relative_cost_reduction_percent=100*(1-best['allocation_h200_hours']/sota['allocation_h200_hours']),
        long_document_share_of_nll_gain=(sota['full_length_buckets']['eval/length_4096_inf_nll']-
            best['full_length_buckets']['eval/length_4096_inf_nll']) /
            ((sota['full_mean_nll']-best['full_mean_nll'])*best['full_prediction_tokens']))
    assert all(x['tokens'] == 3000107008 and x['full_documents'] == 50000 and
               x['full_prediction_tokens'] == 76930208 and not x['checkpoint_loaded'] for x in result['runs'])
    assert len({x['preparation_fingerprint'] for x in result['runs']}) == 1
    assert len({x['initialization_signature'] for x in result['runs']}) == 1
    assert all(math.isclose(math.exp(x['full_mean_nll']), x['full_ppl'], rel_tol=1e-12) for x in result['runs'])
    return result


if __name__ == '__main__':
    evidence = collect()
    (OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    with (OUT / 'eval-curves.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run_id', 'step', 'tokens', 'ppl', 'mean_nll'])
        writer.writeheader()
        writer.writerows(evidence['curves'])
    if shutil.which('gnuplot'):
        subprocess.run(['gnuplot', str(OUT / 'plot_curves.gp')], cwd=REPO, check=True)
    else:
        print('Curve data written; SVG requires gnuplot.')
    print(json.dumps({k:evidence[k] for k in ['measured_h200_hours', 'formal_run_count', 'best_vs_sota', 'historical_recipe']}, ensure_ascii=False))
