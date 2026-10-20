"""Slurm-only correctness and matched-data throughput gates for worker prefetch."""
import contextlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import dataset as pipeline
from filtering_batch_reader import BatchRaw
from filtering_parallel import FixtureTokenizer,OrderedBatches,ParallelRaw,prepared_documents
from preparation_parallel import ordered_executor
from filtering_resources import worker_count,training_worker_count,evaluation_worker_count
from slurm.filtering_batch_checks import rss_upper_bytes


def prepare_indexes(raw):
    started=time.monotonic();raw.close()
    for shard in range(len(raw.files)):
        pipeline.stop_if_requested()
        raw.open(shard);raw.close()
    return dict(shards=len(raw.files),seconds=time.monotonic()-started)


def fixture(directory):
    pipeline.require_compute()
    import pyarrow as pa
    import numpy as np
    import sqlite3
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    schema=pa.schema([('id',pa.string()),('text',pa.string()),('document_json',pa.string())])
    docs=[dict(id=str(i),text='Fixture '+str(i),document_json=json.dumps({'id':str(i),'text':'Fixture '+str(i)})) for i in range(352)]
    files=[]
    for number,piece in enumerate((docs[:160],docs[160:])):
        path=directory/f'raw-{number}.arrow'
        with pa.OSFile(str(path),'wb') as sink,pa.ipc.new_stream(sink,schema) as writer:
            for offset in range(0,len(piece),19):writer.write_batch(pa.RecordBatch.from_pylist(piece[offset:offset+19],schema=schema))
        files.append(dict(**pipeline.artifact(path),rows=len(piece)))
    raw=BatchRaw(dict(row_count=len(docs),files=files),directory/'indexes',dict(fixture='parallel'))
    prepare_indexes(raw)
    workers=3;cfg=dict(preprocessing_workers=workers,document_shard_tokens=24,max_training_tokens=1800,common_crawl_keep_fraction=.1)
    def prep_at(name,budget=1800):
        path=directory/name;path.mkdir();(path/'select').mkdir();(path/'split').mkdir()
        np.save(path/'split/evaluation.npy',np.arange(288,352,dtype=np.uint64))
        with pipeline.database(path/'select/candidates.sqlite') as db:
            db.execute('CREATE TABLE candidates(row_index INTEGER PRIMARY KEY,position INTEGER UNIQUE,source TEXT)')
            db.executemany('INSERT INTO candidates VALUES (?,?,?)',[(i,i,'finemath') for i in range(288)])
            pipeline.set_progress(db,'complete',True);pipeline.set_progress(db,'common_crawl_kept',0);pipeline.set_progress(db,'boundary',None);db.commit()
        config=dict(cfg,max_training_tokens=budget)
        def finish(stage,paths,details):pipeline.atomic_json(path/'select/result.json',details)
        return SimpleNamespace(path=path,config=config,directory=lambda s:path/s,finish=finish,
            manifest=lambda s:{'details':dict(scored_documents=0,train_document_count=288,ranges=[dict(start=0,end=352,source='finemath')])})
    @contextlib.contextmanager
    def inline(*args,**kwargs):yield lambda f,items:map(f,items)
    observed=[]
    def execute(prep,parallel=False,interrupt=False):
        adapter=ParallelRaw(prep,raw,fixture=True) if parallel else raw
        real=pipeline.atomic_json
        def write(path,value):
            real(path,value)
            if interrupt and Path(path)==prep.path/'select/state.json':raise pipeline.StopRequested('native fixture checkpoint')
        try:
            with patch.object(pipeline,'load_raw',return_value=adapter),patch.object(pipeline,'load_tokenizer',return_value=FixtureTokenizer()),patch.object(pipeline,'ordered_executor',inline),patch.object(pipeline,'atomic_json',write),patch.object(pipeline,'_TOKENIZER',FixtureTokenizer()):
                if parallel:
                    with patch.object(pipeline,'tokenize_documents',prepared_documents):pipeline.stage_select(prep)
                else:pipeline.stage_select(prep)
        finally:
            if parallel:observed.append(adapter.stats);adapter.close()
    normal,worker,resume,tight1,tight2=(prep_at('reference'),prep_at('parallel'),prep_at('resumed'),prep_at('budget-reference',59),prep_at('budget-parallel',59))
    execute(normal);execute(worker,True)
    try:execute(resume,True,True)
    except pipeline.StopRequested:pass
    else:raise AssertionError('Fixture did not interrupt')
    execute(resume,True);execute(tight1);execute(tight2,True)
    def compare(a,b):
        states=[pipeline.read_json(p.path/'select/state.json') for p in (a,b)]
        for key in ('train_tokens','train_documents','last_position','budget_skipped_documents','next_evaluation_slot','evaluation_tokens'):
            assert states[0][key]==states[1][key],key
        for key in ('train_files','eval_files'):
            values=[[row for row,_ in pipeline.iter_arrow(s[key])] for s in states];assert values[0]==values[1]
            assert list(pipeline.expected_sequences(iter((row,None) for row in values[0]),8192,9,key=='eval_files'))==list(pipeline.expected_sequences(iter((row,None) for row in values[1]),8192,9,key=='eval_files'))
    compare(normal,worker);compare(normal,resume);compare(tight1,tight2)
    # Direct stream telemetry demonstrates actual read/compute overlap per worker.
    metadata=[[(i,i,'finemath',-1) for i in range(start,start+32)] for start in range(0,288,32)]
    session=OrderedBatches(dict(row_count=len(raw),files=raw.files),raw.directory,raw.identity,cfg,True);timeline=[]
    try:
        for result in session.ordered(metadata):timeline.append(dict(session.metrics[(len(timeline))%workers]))
        bounds=session.stats
    finally:session.close()
    overlap=any(b['read_start']<a['tokenize_end'] and b['read_end']>a['tokenize_start']
                for a in timeline for b in timeline if a['worker']==b['worker'] and b['sequence']>a['sequence'])
    assert overlap,'Prefetch did not overlap tokenization';assert bounds['max_batches_per_worker']<=2
    raw.close();result=dict(passed=True,worker_count=workers,ordered_tokens_equal=True,interrupted_resume_equal=True,
        whole_document_budget_and_skips_equal=True,packing_masks_equal=True,read_compute_overlap=overlap,
        credits=bounds,timeline=timeline,observed_sessions=observed,slurm_job_id=os.environ['SLURM_JOB_ID'])
    pipeline.atomic_json(directory/'result.json',result);return result


def benchmark(prep,raw,directory,identity):
    path=directory/'parallel-performance.json'
    if path.exists():
        result=pipeline.read_json(path)
        if result['identity']!=identity or not result['passed']:raise ValueError('Parallel performance receipt changed or failed')
        return result
    state=pipeline.read_json(prep.path/'select/state.json')
    with contextlib.closing(pipeline.readonly(prep.path/'select/candidates.sqlite')) as db:
        rows=db.execute('SELECT row_index,position,source FROM candidates WHERE position>? ORDER BY position LIMIT 4096',(state['last_position'],)).fetchall()
    if len(rows)!=4096:raise ValueError('Need4096matched documents')
    batches=[[(row,pos,source,-1) for row,pos,source in rows[offset:offset+32]] for offset in range(0,len(rows),32)]
    workers=worker_count(prep.config);tokenizer=pipeline.load_tokenizer(prep);measurements={};reference=[]
    started=time.monotonic()
    with ordered_executor(workers,initializer=pipeline.initialize_tokenizer,initargs=(prep.config,),check=pipeline.stop_if_requested) as ordered:
        for metadata in batches:
            documents=list(raw.select([m[0] for m in metadata]))
            tasks=((original,*m) for original,m in zip(documents,metadata))
            reference.extend(pipeline.tokenize_documents(ordered,tokenizer,tasks,workers))
    duration=time.monotonic()-started;tokens=sum(len(doc['tokens']) for doc in reference)
    measurements['coordinator_reader']=dict(seconds=duration,tokens=tokens,tokens_per_second=tokens/duration)
    counts=sorted(set((workers,training_worker_count(prep.config),evaluation_worker_count(prep.config))))
    for count in counts:
        raw.close();started=time.monotonic();session=OrderedBatches(dict(row_count=len(raw),files=raw.files),raw.directory,raw.identity,prep.config,workers=count)
        offset=0
        try:
            for documents in session.ordered(batches):
                if documents!=reference[offset:offset+len(documents)]:raise ValueError('Parallel benchmark changed tokens or order')
                offset+=len(documents)
            stats=session.stats
        finally:session.close()
        duration=time.monotonic()-started
        if offset!=len(reference):raise ValueError('Parallel benchmark dropped documents')
        measurements[f'worker_reader_prefetch_{count}']=dict(seconds=duration,tokens=tokens,tokens_per_second=tokens/duration,workers=count)
    measurements['worker_reader_prefetch']=measurements[f'worker_reader_prefetch_{counts[-1]}']
    speedup=measurements['coordinator_reader']['seconds']/duration;memory=rss_upper_bytes(counts[-1])
    result=dict(identity=identity,measurements=measurements,speedup=speedup,memory_upper_bytes=memory,
        passed=speedup>=1.1 and memory<160*1024**3 and stats['max_batches_per_worker']<=2,
        input_sha256=pipeline.fingerprint(rows),stats=stats,scaling_speedup=measurements[f'worker_reader_prefetch_{workers}']['seconds']/duration,
        slurm_job_id=os.environ['SLURM_JOB_ID'],timestamp=time.time())
    pipeline.atomic_json(path,result);print('parallel benchmark:',json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Parallel speed/memory/credit gate failed; original production state unchanged')
    return result
