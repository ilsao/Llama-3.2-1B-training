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
from preparation_parallel import configure_processes, configure_threads, ordered_executor
from filtering_resources import worker_count, training_worker_count, evaluation_worker_count


def identity(prep):
    return dict(preparation_fingerprint=prep.key,
        execution_resources=dict(allocated_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','1')),
            training_workers=training_worker_count(prep.config),evaluation_workers=evaluation_worker_count(prep.config)),
        split_sha256=pipeline.sha256(prep.path / 'split/manifest.json'),
        overlay_hashes={name:pipeline.sha256(pipeline.PROJECT/name) for name in
                        ['filtering_reader.py','filtering_batch_reader.py','slurm/filtering_entry.py','slurm/filtering_batch_checks.py','slurm/filtering_batch_fixture.py','filtering_parallel.py','filtering_resources.py','slurm/filtering_parallel_checks.py']})


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
    local_indices=[]; global_indices=[]; expected=[]
    # Compare one native mapping at a time, with one optimized mapping beside it.
    # Reassemble reversed cross-shard/duplicate requests after obtaining references.
    raw.close(); original_bound=raw.max_open; raw.max_open=1
    try:
        for shard in sample_files:
            piece=datasets.Dataset.from_file(raw.files[shard]['path'])
            start=raw.ends[shard-1] if shard else 0
            offsets=[0,len(piece)-1,len(piece)//2]
            references=list(piece.select(offsets))
            actual=list(raw.select([start+i for i in offsets]))
            if actual!=references: raise ValueError('Native shard selection differs')
            expected.extend(references);global_indices.extend(start+i for i in offsets)
            raw.close();del piece,references,actual
        indices=global_indices[::-1]+global_indices[:2]
        reference=expected[::-1]+expected[:2]
        if list(raw.select(indices))!=reference:
            raise ValueError('Bounded selection differs from native cross-file select')
        if raw.select(indices)['text']!=[row['text'] for row in reference]:
            raise ValueError('Column projection differs from native select')
    finally:
        raw.close();raw.max_open=original_bound
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
        raw_access='bounded canonical shard index; two cached shards',reader_validation_sha256=pipeline.sha256(getattr(prep,'reader_validation_path',prep.path/'filtering-access/validation.json'))))
    publish('complete')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['score','select','verify']);parser.add_argument('--prep-id',required=True)
    parser.add_argument('--root',type=Path,default=pipeline.DEFAULT_ROOT)
    parser.add_argument('--config',type=Path,default=pipeline.PROJECT/'configs/dataset.json')
    parser.add_argument('--training-workers',type=int,choices=(7,11,15),default=7)
    parser.add_argument('--evaluation-workers',type=int,choices=(7,11),default=7)
    args=parser.parse_args();pipeline.require_compute();configure_threads();configure_processes()
    if (args.training_workers>7 or args.evaluation_workers>7) and (args.stage!='select' or args.prep_id!='dolma-dclm10-005'):
        raise ValueError('16-CPU override is only authorized for this preparation training tokenization')
    os.environ['FILTERING_TRAIN_WORKERS']=str(args.training_workers)
    os.environ['FILTERING_EVAL_WORKERS']=str(args.evaluation_workers)
    pipeline.worker_count=worker_count
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
        prep.manifest('clean' if args.stage=='score' else 'pack' if args.stage=='verify' else 'score');prep.event(args.stage,'started')
        provenance=prep.directory('filtering-access')/('job-'+os.environ['SLURM_JOB_ID']+'.json')
        pipeline.atomic_json(provenance,dict(identity=identity(prep),stage=args.stage,
            config=cfg,slurm_job_id=os.environ['SLURM_JOB_ID'],started_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print('execution resources:',json.dumps(identity(prep)['execution_resources']),flush=True)
        try:
            index=build_index(prep)
            evidence_identity=identity(prep)
            if args.stage in ('select','verify'):
                from filtering_batch_reader import BatchRaw
                sidecars=prep.path/'filtering-access/batch-reader-v1/indexes'
                raw=BatchRaw(index,sidecars,dict(preparation_fingerprint=prep.key,
                    split_sha256=evidence_identity['split_sha256']))
            else:
                raw=IndexedRaw(index)
            from huggingface_hub import hf_hub_download
            model=hf_hub_download(cfg['classifier_id'],cfg['classifier_filename'],revision=cfg['classifier_revision'])
            receipt_dir=prep.path/'filtering-access'/('batch-reader-v1/'+pipeline.fingerprint(evidence_identity) if args.stage in ('select','verify') else 'score-'+pipeline.fingerprint(evidence_identity))
            receipt_dir.mkdir(parents=True,exist_ok=True)
            validate(prep,raw,model,receipt_dir/'validation.json')
            prep.reader_validation_path=receipt_dir/'validation.json'
            if args.stage=='score':score(prep,raw,model)
            elif args.stage=='verify':
                from slurm.filtering_batch_checks import saved_tokens
                saved_tokens(prep,raw)
                pipeline.load_raw=lambda _:raw
                pipeline.stage_verify(prep)
            else:
                from slurm.filtering_batch_checks import saved_tokens,progress_writer
                from slurm.filtering_parallel_checks import prepare_indexes,fixture as parallel_fixture,benchmark as parallel_benchmark
                from filtering_parallel import ParallelRaw,prepared_documents
                from slurm.filtering_batch_fixture import run as fixture
                fixture(receipt_dir/('fixture-'+os.environ['SLURM_JOB_ID']))
                saved=saved_tokens(prep,raw)
                pipeline.atomic_json(receipt_dir/('saved-tokens-'+os.environ['SLURM_JOB_ID']+'.json'),saved)
                index_receipt=prepare_indexes(raw)
                pipeline.atomic_json(receipt_dir/('indexes-'+os.environ['SLURM_JOB_ID']+'.json'),index_receipt)
                parallel_fixture(receipt_dir/('parallel-fixture-'+os.environ['SLURM_JOB_ID']))
                parallel_benchmark(prep,raw,receipt_dir,evidence_identity)
                adapter=ParallelRaw(prep,raw)
                original,writer=progress_writer(prep,adapter,evidence_identity)
                original_tokenize=pipeline.tokenize_documents
                pipeline.load_raw=lambda _:adapter
                pipeline.tokenize_documents=prepared_documents
                pipeline.atomic_json=writer
                try:
                    pipeline.stage_select(prep)
                finally:
                    pipeline.atomic_json=original
                    pipeline.tokenize_documents=original_tokenize
                    adapter.close()
            raw.close();prep.event(args.stage,'completed');return 0
        except pipeline.StopRequested as error:
            prep.event(args.stage,'checkpointed');print(str(error),file=sys.stderr);return 75
        except Exception:
            prep.event(args.stage,'failed');raise


if __name__=='__main__':
    sys.exit(main())
