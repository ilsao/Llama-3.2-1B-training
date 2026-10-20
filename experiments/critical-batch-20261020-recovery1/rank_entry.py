"""Set private compiler/cache paths BEFORE importing torch or triton."""
import os
from pathlib import Path
import runpy
import sys


def private_paths(root, job, rank):
    if not str(job).isdigit() or not str(rank).isdigit():
        raise ValueError('Numeric Slurm job and local rank required')
    base=Path(root)/f'cbs-recovery-{job}'/f'rank-{rank}'
    return {key:str(base/name) for key,name in [
        ('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton'),
        ('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]}


def main():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME'):
        raise SystemExit('Use sbatch')
    rank=os.environ['LOCAL_RANK'];job=os.environ['SLURM_JOB_ID']
    root='/work/asciibase64/Llama-3.2-1B-training/caches'
    paths=private_paths(root,job,rank)
    for key,value in paths.items():Path(value).mkdir(parents=True,exist_ok=True);os.environ[key]=value
    record=Path(os.environ['JOB_RECORD'])/f'private-cache-rank{rank}.json'
    import json
    record.write_text(json.dumps(paths,indent=2)+'\n')
    entry=sys.argv[1]
    if entry not in ('compile_gate.py','runner.py'):raise ValueError('Unexpected entry')
    runpy.run_path(str(Path(__file__).parent/entry),run_name='__main__')


if __name__=='__main__':main()
