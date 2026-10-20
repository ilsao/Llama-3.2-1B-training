#!/usr/bin/env python3
"""Small read-only Slurm/run receipts, plus a local status snapshot."""
import datetime
import json
import subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
RUN=Path('/work/asciibase64/Llama-3.2-1B-training/runs/experiments/kd6b-r5-20261020-seed42')


def main():
    submission=json.loads((HERE/'submission.json').read_text());job=submission['job_id']
    assert job.isdigit()
    result=subprocess.run(['sacct','-j',job,'-X','--noheader','--parsable2',
          '--format=JobIDRaw,State,ElapsedRaw,Start,End,NodeList'],capture_output=True,text=True,timeout=30,check=True)
    state=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=job,
               slurm=result.stdout.strip(),configuration_digest=submission['configuration_digest'])
    for name in ('starting.json','owner-update-gate.json','serializer-preflight.json','teacher-attention-gate.json',
                 'reader-preflight.json','training-started.json','wandb.json','progress.json','evaluations.json',
                 'summary.json','complete.json','combined10k.json','export.json'):
        if (RUN/name).exists():state[name]=json.loads((RUN/name).read_text())
    state['errors']={p.name:json.loads(p.read_text()) for p in RUN.glob('abort-rank*.json')}
    for role in ('teacher','student'):
        p=RUN/f'{role}-input-journal.json'
        if p.exists():state[role+'_journal']=json.loads(p.read_text())
    if (HERE/'accounting.json').exists():state['accounting']=json.loads((HERE/'accounting.json').read_text())
    (HERE/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    summary={k:v for k,v in state.items() if k in ('slurm','progress.json','errors','wandb.json','complete.json','combined10k.json','accounting')}
    summary['preflight_receipts']=[k for k in state if 'gate' in k or 'preflight' in k]
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
