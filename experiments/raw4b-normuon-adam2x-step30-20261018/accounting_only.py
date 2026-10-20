"""Reconcile only existing job565259. No submit/retry/resume/control commands."""
from pathlib import Path
import sys,subprocess
LIVE=Path(__file__).resolve().parent
sys.path.insert(0,str(LIVE/'source'))
from training_state import DEFAULT_ROOT,ExperimentLedger,TERMINAL,atomic_json,read_json,locked
JOB='565259'
RUN='raw4b-normuon-adam2x-step30-20261018'

def main():
    with locked(LIVE/'.controller.lock',blocking=False):
        state=read_json(LIVE/'controller-state.json');item=state['job']
        if item.get('job_id')!=JOB:raise ValueError('Only existing job565259 is authorized')
        if item.get('status') in ('complete','failed'):return
        query=subprocess.run(['sacct','-X','-n','-P','-j',JOB,'--format=JobIDRaw,State'],capture_output=True,text=True,timeout=30,check=True)
        rows=[r.split('|') for r in query.stdout.splitlines() if r.startswith(JOB+'|')]
        if len(rows)!=1:return
        status=rows[0][1].split()[0].rstrip('+');item['slurm_state']=status
        if status in TERMINAL:
            ledger=ExperimentLedger(DEFAULT_ROOT)
            data=read_json(ledger.path)
            if data['jobs'][JOB]['run_id']!=RUN:raise ValueError('Existing allocation identity changed')
            baseline=read_json(LIVE/'ledger-before.json')
            for name,sha in baseline['run_digests'].items():
                if data['runs'].get(name,{}).get('config_digest')!=sha:raise ValueError('Historical identity missing')
            for job,cost in baseline['measured_costs'].items():
                if data['jobs'].get(job,{}).get('h200_hours')!=cost:raise ValueError('Historical cost missing')
            item['accounting']=ledger.reconcile(JOB)
            item['status']='complete' if status=='COMPLETED' and (LIVE/'receipt.json').exists() else 'failed'
            state['status']=item['status']
            if item['status']=='complete':
                receipt=read_json(LIVE/'receipt.json')
                if receipt['job_id']!=JOB or receipt['run_id']!=RUN:raise ValueError('Receipt identity changed')
                atomic_json(LIVE/'results.json',{**receipt,'h200_hours':item['accounting']['h200_hours'],'allocation':item['accounting']})
            snapshot=read_json(ledger.path)
            if snapshot.get('runs') and snapshot.get('jobs'):atomic_json(LIVE/'ledger-latest.json',snapshot)
        atomic_json(LIVE/'controller-state.json',state)
        print(JOB,status,item['status'])

if __name__=='__main__':
    try:main()
    except BlockingIOError:pass
