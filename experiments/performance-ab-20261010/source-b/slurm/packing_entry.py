#!/usr/bin/env python3
"""Explicit parallel execution overlay for immutable pack/verify preparation."""
import argparse
import datetime as dt
import faulthandler
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dataset as pipeline
from preparation_parallel import configure_processes,configure_threads,worker_count
from filtering_reader import build_index
from filtering_batch_reader import BatchRaw

SOURCES=('packing_parallel.py','verification_parallel.py','slurm/packing_entry.py','slurm/packing_checks.py',
         'filtering_reader.py','filtering_batch_reader.py','filtering_parallel.py','filtering_resources.py',
         'slurm/filtering_entry.py','slurm/filtering_parallel_checks.py','slurm/filtering_batch_checks.py')


def identity(prep):
    return dict(preparation_fingerprint=prep.key,split_sha256=pipeline.sha256(prep.path/'split/manifest.json'),
        select_sha256=pipeline.sha256(prep.path/'select/manifest.json'),
        overlay_hashes={p:pipeline.sha256(pipeline.PROJECT/p) for p in SOURCES},
        execution_resources=dict(cpus=int(os.environ['SLURM_CPUS_PER_TASK']),workers=worker_count(prep.config),
            library_threads_per_worker=1,maximum_outstanding_tasks=2*worker_count(prep.config),raw_shard_cache_per_worker=0))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pack','verify'));parser.add_argument('--prep-id',required=True)
    parser.add_argument('--root',type=Path,default=pipeline.DEFAULT_ROOT)
    parser.add_argument('--config',type=Path,default=pipeline.PROJECT/'configs/dataset.json')
    args=parser.parse_args();pipeline.require_compute();configure_threads();configure_processes()
    faulthandler.enable();faulthandler.dump_traceback_later(300,repeat=True)
    def stop(signum,frame):pipeline.STOP=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    root=pipeline.setup_storage(args.root);cfg=pipeline.read_json(args.config);pipeline.validate_config(cfg)
    if args.prep_id!='dolma-dclm10-005':raise ValueError('This recovery overlay is for the immutable filtering preparation')
    prep=pipeline.Preparation(root,args.prep_id,cfg)
    with (root/'.retention.lock').open('a') as retention,(prep.path/'.lock').open('a') as lock:
        fcntl.flock(retention,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if prep.done(args.stage):return 0
        prep.manifest('select' if args.stage=='pack' else 'pack');prep.event(args.stage,'started')
        evidence=identity(prep);base=prep.directory('packing-access')/'parallel-v1';base.mkdir(exist_ok=True)
        directory=base/pipeline.fingerprint(evidence);directory.mkdir(exist_ok=True)
        pipeline.atomic_json(directory/('job-'+os.environ['SLURM_JOB_ID']+'.json'),dict(identity=evidence,
            stage=args.stage,config=cfg,job_id=os.environ['SLURM_JOB_ID'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print('parallel pack/verify resources:',json.dumps(evidence['execution_resources']),flush=True)
        raw=None
        try:
            index=build_index(prep);raw=BatchRaw(index,prep.path/'filtering-access/batch-reader-v1/indexes',
                dict(preparation_fingerprint=prep.key,split_sha256=evidence['split_sha256']))
            from slurm.filtering_entry import validate
            from huggingface_hub import hf_hub_download
            model=hf_hub_download(cfg['classifier_id'],cfg['classifier_filename'],revision=cfg['classifier_revision'])
            validate(prep,raw,model,directory/'bounded-reader-validation.json')
            from slurm.filtering_parallel_checks import prepare_indexes
            prepare_indexes(raw)
            from slurm.packing_checks import run
            run(prep,raw,directory,evidence,args.stage)
            if args.stage=='pack':
                from packing_parallel import packing
                packing(prep,directory,evidence)
            else:
                from verification_parallel import verify
                verify(prep,raw,index,directory,evidence)
            prep.event(args.stage,'completed');return 0
        except pipeline.StopRequested as error:
            prep.event(args.stage,'checkpointed');print(str(error),file=sys.stderr);return 75
        except Exception:
            prep.event(args.stage,'failed');raise
        finally:
            if raw is not None:raw.close()


if __name__=='__main__':sys.exit(main())
