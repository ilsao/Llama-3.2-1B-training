#!/usr/bin/env python3
"""Read scalar status and reconcile terminal allocations; never submit or resume."""
import datetime
import json
import math
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

LIVE=Path(__file__).resolve().parent
REPO=LIVE.parents[1]
sys.path[:0]=[str(LIVE/'source'),str(REPO/'slurm')]
from training_state import DEFAULT_ROOT,ExperimentLedger,atomic_json,read_json,locked,TERMINAL
from kd3b_hidden8 import RUN,PREFIXES
MARKER='# kd3b-hidden8-20261019-status'


def optional(path):return read_json(path) if path.exists() else {}


def main():
    with locked(LIVE/'.monitor.lock',blocking=False):
        launch=read_json(LIVE/'submission.json')
        ids=[x['job_id'] for x in launch['roles'].values()]
        result=subprocess.run(['sacct','-X','-j',','.join(ids),'--noheader','--parsable2',
            '--format=JobIDRaw,State,ElapsedRaw,Start,End,NodeList%128'],capture_output=True,text=True,timeout=20,check=True)
        jobs={}
        for line in result.stdout.splitlines():
            f=line.split('|')
            if f[0] in ids:jobs[f[0]]=dict(state=f[1],elapsed_seconds=int(f[2]),start=f[3],end=f[4],node=f[5])
        terminal=len(jobs)==2 and all(j['state'].split()[0] in TERMINAL for j in jobs.values())
        accounting={}
        if terminal:
            ledger=ExperimentLedger(DEFAULT_ROOT)
            for job in ids:accounting[job]=ledger.reconcile(job)
            atomic_json(LIVE/'accounting.json',accounting)
        run=DEFAULT_ROOT/'runs/experiments'/RUN
        metrics=run/'metrics.jsonl';latest={};evaluations=[]
        if metrics.exists():
            if metrics.stat().st_size>32*1024*1024:raise ValueError('Scalar log exceeds monitoring limit')
            for line in metrics.read_text().splitlines():
                try:row=json.loads(line)
                except json.JSONDecodeError:continue
                if 'train/tokens_per_second' in row:
                    latest={k:row[k] for k in ('train/global_step','train/total_tokens_seen','train/loss',
                        'train/tokens_per_second','train/grad_norm','train/teacher_inference_inputs',
                        'train/hidden_wait_seconds_cumulative_max_rank','timestamp') if k in row}
                if 'eval/perplexity' in row:
                    evaluations.append({k:row[k] for k in ('train/global_step','eval/total_training_tokens_seen',
                        'eval/perplexity','eval/nll','eval/prediction_tokens')})
        ledger=read_json(DEFAULT_ROOT/'experiment-ledger.json')
        historical={k:j for k,j in ledger['jobs'].items() if j['run_id'].startswith(PREFIXES)}
        costs=dict(measured_h200_hours=sum(j.get('h200_hours',0) for j in historical.values()),
            reserved_or_measured_h200_hours=sum(j.get('h200_hours',j['reserved_h200_hours']) for j in historical.values()),
            study_cap_h200_hours=191.419)
        summary=optional(run/'summary.json');confirmation=optional(run/'confirmation-evaluation.json')
        ppl10k=None
        if summary.get('training_token_target_reached') and confirmation:
            first=summary['last_evaluation']
            if (summary['total_tokens_seen']!=3000107008 or summary['global_step']!=14312 or
                    confirmation['global_step']!=14312 or confirmation['export']!=summary['export'] or
                    first['eval/prediction_tokens']!=1479302 or confirmation['eval/prediction_tokens']!=13666812):
                raise ValueError('Endpoint evaluation identity mismatch')
            ppl10k=math.exp((first['eval/nll']+confirmation['eval/nll'])/15146114)
        status=dict(timestamp=datetime.datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),run_id=RUN,jobs=jobs,
            terminal=terminal,latest_training=latest,evaluations=evaluations,
            student_journal=optional(run/'token-journal.json'),teacher_progress=optional(run/'transport/producer/progress.json'),
            teacher_journal=optional(run/'transport/producer/teacher-input-journal.json'),
            teacher_done=optional(run/'transport/producer/done.json'),wandb=optional(run/'wandb.json'),
            summary=summary,endpoint_10k_perplexity=ppl10k,costs=costs,
            previous_experiments_status=str(REPO/'experiments/methods-parallel-20261019/tracking/status.json'))
        atomic_json(LIVE/'status.json',status)
        lines=[f"Updated {status['timestamp']}",f"Run {RUN}",
            f"Jobs: {json.dumps(jobs)}",f"Student: {json.dumps(latest)}",
            f"Teacher: {json.dumps(status['teacher_progress'])}",f"Budget: {json.dumps(costs)}",
            f"Endpoint10k PPL: {ppl10k}",f"W&B: {status['wandb'].get('url','initializing')}"]
        (LIVE/'status.md').write_text('\n\n'.join(lines)+'\n')
        if terminal:
            cron=subprocess.run(['crontab','-l'],capture_output=True,text=True,timeout=10)
            if cron.returncode==0 and MARKER in cron.stdout:
                kept='\n'.join(x for x in cron.stdout.splitlines() if MARKER not in x)+'\n'
                subprocess.run(['crontab','-'],input=kept,text=True,check=True,timeout=10)
        print(json.dumps(dict(timestamp=status['timestamp'],jobs=jobs,latest=latest,terminal=terminal)))


if __name__=='__main__':main()
