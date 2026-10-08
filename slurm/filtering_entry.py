#!/usr/bin/env python3
"""Verified bounded-reader overlay for immutable scoring/selection preparation."""
import argparse
import datetime as dt
import faulthandler
import fcntl
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dataset as pipeline
from filtering_reader import IndexedRaw, build_index
from preparation_parallel import configure_processes, configure_threads, ordered_executor, worker_count


def identity(prep):
    return dict(preparation_fingerprint=prep.key,
        split_sha256=pipeline.sha256(prep.path / 'split/manifest.json'),
        overlay_hashes={name:pipeline.sha256(pipeline.PROJECT/name) for name in
                        ['filtering_reader.py','slurm/filtering_entry.py']})


def validate(prep, raw, model_path, receipt):
    """Compare real reserved originals, native shard selection and official scores."""
    import pyarrow as pa
    import datasets
    evidence_identity=identity(prep)
    if receipt.exists():
        old=pipeline.read_json(receipt)
        if old.get('passed') is True and old['identity']==evidence_identity:
            return old
        raise ValueError('Existing reader validation differs; do not overwrite provenance')
    details=prep.manifest('split')['details']; samples=[]; sources={}
    # Independent original evaluation archive authenticates global row mapping/full fields.
    for item in details['evaluation_raw_files']:
        pipeline.stop_if_requested()
        with pa.memory_map(item['path'],'r') as src:
            table=pa.ipc.open_stream(src).read_all()
            for row in table.to_pylist():
                if sources.get(row['source'],0)<2:
                    samples.append(row);sources[row['source']]=sources.get(row['source'],0)+1
        if len(sources)==6 and all(v==2 for v in sources.values()): break
    if set(sources)!=pipeline.SOURCES:
        raise ValueError('Need independently archived samples from all six sources')
    for saved in samples:
        actual=raw[int(saved['row_index'])]
        if actual['document_json']!=saved['document_json'] or pipeline.original_document(actual)!=json.loads(saved['document_json']):
            raise ValueError('Indexed reader changed original document/global row mapping')
    # Cross-file bounds/order/duplicates/column projection versus native Dataset.select.
    sample_files=sorted({raw.locate(int(row['row_index']))[0] for row in samples})[:3]
    pieces=[datasets.Dataset.from_file(raw.files[s]['path']) for s in sample_files]
    native=datasets.concatenate_datasets(pieces)
    local_indices=[]; global_indices=[]; count=0
    for s,piece in zip(sample_files,pieces):
        global_start=raw.ends[s-1] if s else 0
        for offset in [0,len(piece)-1,len(piece)//2]:
            local_indices.append(count+offset);global_indices.append(global_start+offset)
        count+=len(piece)
    local_indices=local_indices[::-1]+local_indices[:2]; global_indices=global_indices[::-1]+global_indices[:2]
    if list(native.select(local_indices))!=list(raw.select(global_indices)):
        raise ValueError('Bounded selection differs from native cross-file select')
    if native.select(local_indices)['text']!=raw.select(global_indices)['text']:
        raise ValueError('Column projection differs from native select')
    del native,pieces
    # Compare 1/7 CPU-worker official classification outputs, then saved score receipt.
    texts=raw.select([int(s['row_index']) for s in samples])['text']
    pipeline.initialize_scorer(model_path)
    serial=pipeline.score_texts(texts)
    workers=worker_count(prep.config,scoring=True)
    with ordered_executor(workers,initializer=pipeline.initialize_scorer,initargs=(model_path,),check=pipeline.stop_if_requested) as ordered:
        parallel=[x for row in ordered(pipeline.score_texts,[[text] for text in texts]) for x in row]
    if parallel!=serial: raise ValueError('Official scores differ between 1/7 workers')
    score_db=prep.path/'score/scores.sqlite'; checked=0
    if score_db.exists():
        with contextlib.closing(pipeline.readonly(score_db)) as db:
            saved=db.execute('SELECT row_index,score FROM scores ORDER BY row_index DESC LIMIT 16').fetchall()
        if saved:
            computed=pipeline.score_texts(raw.select([r[0] for r in saved])['text'])
            if computed!=[r[1] for r in saved]: raise ValueError('Existing committed scores differ with bounded reader')
            checked=len(saved)
    evidence=dict(passed=True,identity=evidence_identity,archived_documents=len(samples),sources=sources,
        native_selection=True,duplicate_and_order_preservation=True,official_worker_equivalence=[1,workers],
        existing_scores_verified=checked,maximum_cached_shards=raw.maximum_open,
        slurm_job_id=os.environ['SLURM_JOB_ID'],completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    pipeline.atomic_json(receipt,evidence);raw.close()
    print('bounded-reader validation passed',json.dumps(evidence),flush=True)
    return evidence


def score(prep, raw, model_path):
    cfg=prep.config; workers=worker_count(cfg,scoring=True)
    path=prep.directory('score')/'scores.sqlite'; progress_path=prep.path/'score/progress.json'
    expected=prep.manifest('clean')['details']['source_counts']['common_crawl']
    with contextlib.closing(pipeline.readonly(prep.path/'clean/documents.sqlite')) as clean, pipeline.database(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS scores (row_index INTEGER PRIMARY KEY, position INTEGER NOT NULL, score REAL NOT NULL)')
        last=pipeline.progress(db,'last_row',-1)
        initial=db.execute('SELECT COUNT(*) FROM scores').fetchone()[0]; count=initial
        if count and db.execute('SELECT MAX(row_index) FROM scores').fetchone()[0]!=last:
            raise ValueError('Committed score cursor disagrees with score rows')
        query=clean.execute("SELECT row_index,position FROM winners WHERE source='common_crawl' AND row_index>? ORDER BY row_index",(last,))
        started=time.monotonic();last_log=started;elapsed=0;processed=0
        def publish(status):
            seconds=time.monotonic()-started;rate=(count-initial)/seconds if seconds>0 else 0
            value=dict(status=status,job_id=os.environ['SLURM_JOB_ID'],completed_documents=count,
                expected_documents=expected,last_row=last,session_documents=count-initial,session_seconds=seconds,
                session_documents_per_second=rate,remaining_seconds_estimate=(expected-count)/rate if rate>0 else None,
                maximum_cached_shards=raw.maximum_open,identity=identity(prep),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            pipeline.atomic_json(progress_path,value); print('score progress:',json.dumps(value),flush=True)
        publish('running')
        try:
            with ordered_executor(workers,initializer=pipeline.initialize_scorer,initargs=(model_path,),check=pipeline.stop_if_requested) as ordered:
                while True:
                    pipeline.stop_if_requested(); rows=query.fetchmany(cfg['document_batch_size']*workers)
                    if not rows:break
                    texts=raw.select([r[0] for r in rows])['text'];size=cfg['document_batch_size']
                    values=[v for part in ordered(pipeline.score_texts,[texts[i:i+size] for i in range(0,len(texts),size)]) for v in part]
                    if len(values)!=len(rows):raise ValueError('Missing official scores')
                    with db:
                        db.executemany('INSERT INTO scores VALUES (?,?,?)',[(row,pos,v) for (row,pos),v in zip(rows,values)])
                        pipeline.set_progress(db,'last_row',rows[-1][0])
                        pipeline.set_progress(db,'scored_documents',count+len(rows))
                    count+=len(rows);last=rows[-1][0]
                    if time.monotonic()-last_log>=30:publish('running');last_log=time.monotonic()
            if count!=expected:raise ValueError('Not every eligible Common Crawl document was scored')
            publish('building_ranking_index')
            # Preserve original score DESC / canonical-position ASC tie-breaking index.
            deadline=int(os.environ.get('PREP_STOP_EPOCH','0'))
            db.set_progress_handler(lambda:int(pipeline.STOP or (deadline and time.time()>=deadline)),100000)
            db.execute('CREATE INDEX IF NOT EXISTS score_ranking ON scores(score DESC,position ASC)')
        except pipeline.StopRequested:
            publish('paused');raise
        except sqlite3.OperationalError as error:
            if 'interrupted' in str(error).lower():publish('paused');raise pipeline.StopRequested('Ranking index checkpointed') from error
            publish('failed');raise
        except Exception:
            publish('failed');raise
    prep.finish('score',[path,model_path],dict(scored_documents=count,workers=workers,
        input='full text; official strip and splitlines join',classifier_sha256=pipeline.sha256(Path(model_path)),device='CPU',
        score='official DCLM high-quality probability; no rounding or clamping',
        raw_access='bounded canonical shard index; two cached shards',reader_validation_sha256=pipeline.sha256(prep.path/'filtering-access/validation.json')))
    publish('complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['score','select']);parser.add_argument('--prep-id',required=True)
    parser.add_argument('--root',type=Path,default=pipeline.DEFAULT_ROOT)
    parser.add_argument('--config',type=Path,default=pipeline.PROJECT/'configs/dataset.json')
    args=parser.parse_args();pipeline.require_compute();configure_threads();configure_processes()
    faulthandler.enable();faulthandler.dump_traceback_later(300,repeat=True)
    def stop(signum,frame):pipeline.STOP=True
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    root=pipeline.setup_storage(args.root);cfg=pipeline.read_json(args.config);pipeline.validate_config(cfg)
    import pyarrow as pa
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    prep=pipeline.Preparation(root,args.prep_id,cfg)
    with (root/'.retention.lock').open('a') as retention,(prep.path/'.lock').open('a') as lock:
        fcntl.flock(retention,fcntl.LOCK_SH|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if prep.done(args.stage):return 0
        prep.manifest('clean' if args.stage=='score' else 'score');prep.event(args.stage,'started')
        provenance=prep.directory('filtering-access')/('job-'+os.environ['SLURM_JOB_ID']+'.json')
        pipeline.atomic_json(provenance,dict(identity=identity(prep),stage=args.stage,
            config=cfg,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        try:
            raw=IndexedRaw(build_index(prep))
            from huggingface_hub import hf_hub_download
            model=hf_hub_download(cfg['classifier_id'],cfg['classifier_filename'],revision=cfg['classifier_revision'])
            validate(prep,raw,model,prep.path/'filtering-access/validation.json')
            if args.stage=='score':score(prep,raw,model)
            else:
                pipeline.load_raw=lambda _:raw
                pipeline.stage_select(prep)
            raw.close();prep.event(args.stage,'completed');return 0
        except pipeline.StopRequested as error:
            prep.event(args.stage,'checkpointed');print(str(error),file=sys.stderr);return 75
        except Exception:
            prep.event(args.stage,'failed');raise


if __name__=='__main__':
    sys.exit(main())
