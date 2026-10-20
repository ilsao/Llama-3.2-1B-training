"""Bounded Slurm submission recovery; metadata only, never resubmits."""
import datetime as dt
import getpass
from pathlib import Path
import re
import subprocess


def find_submission_job(item, root):
    if item.get('returned_job_id'):
        return item['returned_job_id']
    query_start = (dt.datetime.fromisoformat(item['submission_started_utc'])
                   - dt.timedelta(minutes=1)).astimezone().strftime('%Y-%m-%dT%H:%M:%S')
    commands = [
        ['squeue', '--noheader', '--user', getpass.getuser(), '--format=%i|%k'],
        ['sacct', '-X', '--noheader', '--parsable2', '--user', getpass.getuser(),
         '--starttime', query_start, '--format=JobIDRaw,Comment%128'],
    ]
    matches = set()
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode:
            raise RuntimeError('Slurm recovery query failed; reservation remains charged')
        for row in result.stdout.splitlines():
            fields = row.strip().split('|')
            if len(fields) >= 2 and fields[1].strip() == item['submission_tag']:
                if not re.fullmatch(r'\d+', fields[0].strip()):
                    raise ValueError('Unexpected tagged Slurm job ID; reservation remains charged')
                matches.add(fields[0].strip())
    # nano4 does not retain job comments in sacct (AccountingStoreFlags=null).
    # Startup scontrol allocation receipts retain the exact submission tag.
    for receipt in (Path(root) / 'logs').glob('job-*/allocation.txt'):
        comment = re.search(r'\bComment=(\S+)', receipt.read_text())
        job_id = receipt.parent.name[len('job-'):]
        if comment and comment[1] == item['submission_tag'] and re.fullmatch(r'\d+', job_id):
            matches.add(job_id)
    if len(matches) != 1:
        raise ValueError(f'Recovery found {len(matches)} matching jobs; reservation remains charged; do not retry submission')
    job_id = matches.pop()
    return job_id
