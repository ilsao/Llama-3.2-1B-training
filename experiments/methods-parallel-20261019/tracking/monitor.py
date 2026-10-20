#!/usr/bin/env python3
"""Bounded metadata monitoring; never submits, cancels, resumes or loads models."""
import argparse
import datetime as dt
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[1]
ROOT = Path('/work/asciibase64/Llama-3.2-1B-training')
sys.path[:0] = [str(STUDY), str(STUDY/'source')]
from training_state import ExperimentLedger, atomic_json, locked, read_json

TARGET = 1000079360
EXPIRY = dt.datetime(2026, 10, 19, 22, tzinfo=ZoneInfo('Asia/Taipei')).timestamp()
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
LABELS = {'N0': '歷史 NorMuon 控制', 'D1': 'Dion3 full', 'D025': 'Dion3 fraction .25',
          'A1': 'AdaMuon', 'S1': 'SOAP', 'K1': '舊本地教師蒸餾',
          'F1': 'Schedule-Free feedback', 'F2': '平均權重 observer',
          'C1': 'CCE exact', 'K3': '官方 3B 教師 pipeline'}
PREFIXES = ('methods-parallel-20261019', 'kd-acceleration-', 'kd-pipeline-', 'kd3b-online-pipeline-', 'kd3b-io-validation-', 'kd3b-io8-validation-')
CRON_MARKER = '# llama-methods-tracking-20261019'


def optional(path):
    return read_json(path) if path.exists() else {}


def metrics(path):
    # Only small scalar training logs, never model/data/checkpoint artifacts.
    if not path.exists():
        return [], []
    if path.stat().st_size > 16*1024*1024:
        raise ValueError('Scalar log exceeds login-node monitoring bound: '+str(path))
    evaluations, latest = [], []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):continue  # concurrent writer's final row
                raise
            if 'eval/perplexity' in row:
                evaluations.append(dict(tokens=row['eval/total_training_tokens_seen'],
                    step=row['train/global_step'], ppl=row['eval/perplexity'],
                    nll=row['eval/nll'], predictions=row['eval/prediction_tokens']))
            if 'train/tokens_per_second' in row:
                latest = {k: row[k] for k in ('train/global_step', 'train/total_tokens_seen',
                    'train/loss', 'train/tokens_per_second', 'train/grad_norm', 'timestamp') if k in row}
    return evaluations, latest


def matched(left, right):
    """Exact token/subset/count matches only; never interpolate PPL curves."""
    if not left['identity'] or left['identity'] != right['identity']:
        return []
    other = {(x['tokens'], x['predictions']): x for x in right['evaluations']}
    return [dict(tokens=x['tokens'], ppl=x['ppl'], control_ppl=other[(x['tokens'],x['predictions'])]['ppl'],
                 delta_nll=x['nll']/x['predictions']-other[(x['tokens'],x['predictions'])]['nll']/x['predictions'])
            for x in left['evaluations'] if (x['tokens'],x['predictions']) in other]


def endpoint(summary, confirmation):
    if not (summary.get('training_token_target_reached') and summary.get('total_tokens_seen') == TARGET
            and summary.get('global_step') == 4775 and confirmation):
        return None
    selection = summary['last_evaluation']
    if not (confirmation['global_step'] == summary['last_eval_step'] == 4775
            and confirmation['export'] == summary['export']
            and confirmation['preparation_fingerprint'] == summary['preparation_fingerprint']
            and selection['eval/prediction_tokens'] == 1479302
            and confirmation['eval/prediction_tokens'] == 13666812):
        raise ValueError('Endpoint confirmation does not match student checkpoint/subset')
    return math.exp((selection['eval/nll']+confirmation['eval/nll'])/15146114)


def query(ids):
    if not ids:return {}, None
    try:
        result = subprocess.run(['sacct','-j',','.join(ids),'-X','--noheader','--parsable2',
            '--format=JobIDRaw,State,ElapsedRaw,Start,End,NodeList%50'],
            text=True,capture_output=True,check=True,timeout=15)
        rows = {}
        for line in result.stdout.splitlines():
            f=line.split('|')
            if f[0] in ids:
                rows[f[0]]=dict(state=f[1], elapsed_seconds=int(f[2]), start=f[3], end=f[4], node=f[5])
        return rows, None
    except (OSError,subprocess.SubprocessError,ValueError) as exc:
        return {}, str(exc)


def snapshot():
    from control_reference import validate
    reference = validate()
    ledger = read_json(ROOT/'experiment-ledger.json')
    registry = [('N0',reference['run_id'])]
    registry += [(x['arm'],x['run_id']) for x in read_json(STUDY/'matrix.json')['jobs']]
    registry += [('K3','kd3b-online-pipeline-20261019-seed42')]
    records = {}
    for arm, run_id in registry:
        run = ROOT/'runs/experiments'/run_id
        cfg = optional(run/'resolved-config.json')
        provenance = optional(run/'data-provenance.json')
        evaluations, latest = metrics(run/'metrics.jsonl')
        summary = optional(run/'summary.json')
        journal = optional(run/'token-journal.json')
        allocations = sorted([j for j in ledger['jobs'].values() if j['run_id']==run_id],
                             key=lambda j: int(j.get('job_id','0')))
        last_job = allocations[-1] if allocations else {}
        identity = [provenance.get(k) for k in ('fingerprint','evaluation_indices_sha256','evaluation_manifest_sha256')]
        identity += [cfg.get('seed'),cfg.get('eval_documents'),cfg.get('sequence_length'),cfg.get('ppl',{}).get('mask')]
        if any(x is None for x in identity):identity = None
        record = dict(arm=arm, label=LABELS[arm], run_id=run_id, job_id=last_job.get('job_id'),
            state='HISTORICAL' if arm=='N0' else last_job.get('state','UNKNOWN'),
            inputs=journal.get('confirmed_tokens',summary.get('total_tokens_seen',0)),
            charged_inputs=journal.get('charged_tokens',summary.get('budget_tokens_charged',0)),
            pending_input_reservations=journal.get('pending',{}),
            evaluations=evaluations,latest_training=latest,identity=identity,
            ppl_10k=reference['ppl_10k'] if arm=='N0' else endpoint(summary,optional(run/'confirmation-evaluation.json')),
            allocation_ids=[j.get('job_id') for j in allocations],
            h200_hours_measured=sum(j.get('h200_hours',0) for j in allocations),
            h200_hours_reserved_or_measured=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in allocations),
            accounting_reconciled=all('h200_hours' in j for j in allocations),
            wandb_url=optional(run/'wandb.json').get('url'))
        records[arm]=record
    live, slurm_error = query([r['job_id'] for a,r in records.items() if a!='N0' and r['job_id']])
    for arm, row in records.items():
        if arm!='N0' and row['job_id'] in live:
            row['slurm']=live[row['job_id']]
            row['state']=row['slurm']['state']
        row['matched_vs_N0']=matched(row,records['N0']) if arm!='N0' else []
    # Reconcile only terminated allocations already in the shared ledger.
    errors=[]
    for arm,row in records.items():
        if arm=='N0':continue
        for job in row['allocation_ids']:
            if job in live and live[job]['state'].split()[0] in TERMINAL and 'h200_hours' not in ledger['jobs'][job]:
                try:ExperimentLedger(ROOT).reconcile(job)
                except Exception as exc:errors.append(dict(job_id=job,error=str(exc)))
    ledger=read_json(ROOT/'experiment-ledger.json')
    for arm,row in records.items():
        allocations=[j for j in ledger['jobs'].values() if j['run_id']==row['run_id']]
        row.update(h200_hours_measured=sum(j.get('h200_hours',0) for j in allocations),
            h200_hours_reserved_or_measured=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in allocations),
            accounting_reconciled=all('h200_hours' in j for j in allocations))
    jobs=[j for j in ledger['jobs'].values() if j['run_id'].startswith(PREFIXES)]
    all_done=all(r['state'].split()[0] in TERMINAL and r['accounting_reconciled']
                 for a,r in records.items() if a!='N0')
    return dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        updated_taipei=dt.datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        runs=records,slurm_query_error=slurm_error,accounting_errors=errors,
        study_budget=dict(cap_H200_hours=128,measured_H200_hours=sum(j.get('h200_hours',0) for j in jobs),
            reserved_or_measured_H200_hours=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in jobs)),
        all_terminal_and_reconciled=all_done,
        monitoring='Every 5 minutes until all tracked jobs terminate and reconcile, or 2026-10-19 22:00 Asia/Taipei. No job submissions/cancellations/resumes.',
        interpretation='Single seed42; historical quality comparisons only. No interpolation or cross-budget endpoint ranking; no confidence interval or speedup significance claim.')


def render(data):
    lines=[f"更新：{data['updated_taipei']}；每5分鐘追蹤，至全部結束並記帳或台北22:00。",'',
           '| ID | 方法 | 狀態 | 已確認 inputs (M) | 最後1k PPL @ inputs(M) | 完成1B的10k PPL | H200-hours實測／預留含實測 |',
           '|---|---|---|---:|---|---:|---:|']
    for arm,row in data['runs'].items():
        last=row['evaluations'][-1] if row['evaluations'] else None
        quality=f"{last['ppl']:.4f} @ {last['tokens']/1e6:.3f}" if last else '—'
        end=f"{row['ppl_10k']:.4f}" if row['ppl_10k'] is not None else '—'
        lines.append(f"| {arm} | {row['label']} | {row['state']} | {row['inputs']/1e6:.3f} | {quality} | {end} | {row['h200_hours_measured']:.3f} / {row['h200_hours_reserved_or_measured']:.3f} |")
    k3=data['runs']['K3']
    points=k3['evaluations']
    if points:
        point=points[-1]
        lines += ['',f"新蒸餾最新評估點：{point['tokens']:,} inputs；以下只列精確相同token與subset的1k結果。",'',
                  '| 方法 | PPL ↓ | 相對N0 NLL差 |','|---|---:|---:|']
        for arm,row in data['runs'].items():
            pair=[x for x in matched(k3,row) if x['tokens']==point['tokens']]
            baseline=[x for x in row['matched_vs_N0'] if x['tokens']==point['tokens']]
            if pair:
                delta='0' if arm=='N0' else f"{baseline[-1]['delta_nll']:+.6f}" if baseline else '—'
                lines.append(f"| {arm} | {pair[-1]['control_ppl']:.4f} | {delta} |")
    b=data['study_budget']
    lines += ['',f"整批已結算 {b['measured_H200_hours']:.6f} H200-hours；含目前完整預留 {b['reserved_or_measured_H200_hours']:.6f} / 128。",
              'N0成本保留於原歷史帳本，未重複計入這批；其他成本包含失敗allocation。',
              '各方法最後1k數值可能對應不同tokens，不可直接排序。10k欄僅接受完整1B終點且checkpoint／subset一致的1k+9k token加權結果。',
              '只有seed42，未估計跨seed變異或統計顯著性；歷史成本／速度不能當作同期加速對照。','',
              f"[新蒸餾 W&B]({k3['wandb_url']})"]
    if data['slurm_query_error']:lines += ['', 'Slurm查詢暫時失敗，狀態取帳本；不推論完成：'+data['slurm_query_error']]
    if data['accounting_errors']:lines += ['', '有allocation尚未成功結算，保留完整預留。']
    return '\n'.join(lines)+'\n'


def retire_cron(reason):
    current=subprocess.run(['crontab','-l'],text=True,capture_output=True,check=True,timeout=5).stdout
    kept=[line for line in current.splitlines() if not line.endswith(CRON_MARKER)]
    if len(kept)!=len(current.splitlines()):
        subprocess.run(['crontab','-'],input='\n'.join(kept)+'\n',text=True,check=True,timeout=5)
    atomic_json(HERE/'monitor-retired.json',dict(reason=reason,
        timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),other_cron_entries_preserved=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tick',action='store_true')
    args=parser.parse_args()
    if args.tick and (time.time()>EXPIRY or (HERE/'finished.json').exists()):
        retire_cron('all_terminal' if (HERE/'finished.json').exists() else 'monitoring_deadline')
        return
    with locked(HERE/'.monitor.lock',blocking=False):
        data=snapshot()
        atomic_json(HERE/'status.json',data)
        temp=HERE/'status.md.tmp'
        temp.write_text(render(data));temp.replace(HERE/'status.md')
        if data['all_terminal_and_reconciled']:
            atomic_json(HERE/'finished.json',dict(updated_utc=data['updated_utc'],reason='all_terminal_and_reconciled'))
            if args.tick:retire_cron('all_terminal_and_reconciled')
        print(json.dumps(dict(updated_taipei=data['updated_taipei'],states={a:r['state'] for a,r in data['runs'].items()},
                              all_terminal_and_reconciled=data['all_terminal_and_reconciled'])))


if __name__=='__main__':main()
