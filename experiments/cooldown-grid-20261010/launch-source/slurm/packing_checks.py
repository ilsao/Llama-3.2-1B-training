"""Slurm-only original/parallel equivalence, resume, corruption and speed gates."""
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace
from unittest.mock import patch

import dataset as pipeline
import packing_parallel as parallel
import verification_parallel as verification
from filtering_batch_reader import BatchRaw
from slurm.filtering_parallel_checks import prepare_indexes


def fixture(directory):
    import numpy as np
    import pyarrow as pa
    import datasets
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    n=352;reserved=64;length=8
    original=[dict(id=str(i),text=f'Fixture {i}',extra=[i,None]) for i in range(n)]
    raw_rows=[dict(id=d['id'],text=d['text'],document_json=json.dumps(d)) for d in original]
    raw_files=[]
    for start in range(0,n,128):
        path=directory/f'raw-{start}.arrow';table=pa.Table.from_pylist(raw_rows[start:start+128])
        with pa.OSFile(str(path),'wb') as sink,pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table,max_chunksize=19)
        raw_files.append(dict(**pipeline.artifact(path),rows=len(table)))
    index=dict(row_count=n,files=raw_files);raw_identity=dict(fixture='packing')
    raw=BatchRaw(index,directory/'indexes',raw_identity);prepare_indexes(raw)
    shuffled=datasets.Dataset.from_dict({'id':list(range(n))}).shuffle(seed=42)
    perm=np.asarray(shuffled['id'],dtype=np.uint64);inverse=np.empty(n,dtype=np.uint64);inverse[perm]=np.arange(n,dtype=np.uint64)
    evaluation=perm[-reserved:];ranges=[dict(start=0,end=64,source='common_crawl'),dict(start=64,end=n,source='stack_edu')]
    source=lambda row:'common_crawl' if row<64 else 'stack_edu'
    cfg=dict(preprocessing_workers=3,score_workers=3,sequence_length=length,sequence_shard_rows=3,
        evaluation_documents=reserved,evaluation_subset_documents=16,shuffle_seed=42,common_crawl_keep_fraction=.1,max_training_tokens=1000000)
    def prepare(name):
        path=directory/name;path.mkdir()
        for stage in pipeline.STAGES:(path/stage).mkdir()
        key='fixture-'+name
        def manifest(stage,full=False):
            result=pipeline.read_json(path/stage/'manifest.json')
            for item in result['artifacts']:pipeline.check_artifact(item,full)
            return result
        def finish(stage,paths,details,records=None):
            parent=pipeline.STAGES[pipeline.STAGES.index(stage)-1] if stage!='download' else None
            pipeline.atomic_json(path/stage/'manifest.json',dict(stage=stage,fingerprint=key,
                parent_sha256=pipeline.sha256(path/parent/'manifest.json') if parent else None,
                artifacts=records if records is not None else [pipeline.artifact(Path(p)) for p in paths],details=details))
        prep=SimpleNamespace(path=path,config=dict(cfg),key=key,manifest=manifest,finish=finish,directory=lambda s:path/s)
        finish('download',[i['path'] for i in raw_files],{})
        for name,array in (('permutation',perm),('inverse',inverse),('evaluation',evaluation),('evaluation_subset',evaluation[:16])):np.save(path/'split'/f'{name}.npy',array)
        archived=[]
        for slot,row in enumerate(evaluation):
            # Match dataset.archive_evaluation_originals exactly: no flattened
            # id/text columns exist in the immutable evaluation original archive.
            row=int(row);archived.append(dict(document_json=raw_rows[row]['document_json'],row_index=row,shuffle_position=int(inverse[row]),
                evaluation_slot=slot,source=source(row),text_sha256=hashlib.sha256(original[row]['text'].encode()).digest()))
        archive_path=path/'split/evaluation-raw.arrow';table=pa.Table.from_pylist(archived)
        with pa.OSFile(str(archive_path),'wb') as sink,pa.ipc.new_stream(sink,table.schema) as writer:writer.write_table(table)
        archive=dict(**pipeline.artifact(archive_path),rows=reserved)
        split=dict(row_count=n,train_document_count=n-reserved,ranges=ranges,evaluation_raw_files=[archive])
        finish('split',[str(p) for p in (path/'split').iterdir()],split)
        with pipeline.database(path/'clean/documents.sqlite') as db:
            db.execute('CREATE TABLE winners(digest BLOB PRIMARY KEY,row_index INTEGER,position INTEGER,source TEXT)')
            db.executemany('INSERT INTO winners VALUES(?,?,?,?)',[(hashlib.sha256(original[int(r)]['text'].encode()).digest(),int(r),p,source(int(r))) for p,r in enumerate(perm[:-reserved])])
            db.execute('CREATE UNIQUE INDEX winner_rows ON winners(row_index)')
            cc=db.execute("SELECT row_index,position FROM winners WHERE source='common_crawl'").fetchall()
            noncc=db.execute("SELECT row_index,position,source FROM winners WHERE source!='common_crawl'").fetchall()
        clean=dict(eligible_documents=n-reserved,source_counts={'common_crawl':len(cc),'stack_edu':len(noncc)})
        finish('clean',[str(path/'clean/documents.sqlite')],clean)
        with pipeline.database(path/'score/scores.sqlite') as db:
            db.execute('CREATE TABLE scores(row_index INTEGER PRIMARY KEY,position INTEGER,score REAL)')
            db.executemany('INSERT INTO scores VALUES(?,?,?)',[(r,p,(r%5)/5) for r,p in cc])
            db.execute('CREATE INDEX score_ranking ON scores(score DESC,position ASC)')
            kept=db.execute('SELECT row_index,position FROM scores ORDER BY score DESC,position ASC LIMIT ?',(len(cc)//10,)).fetchall()
        finish('score',[str(path/'score/scores.sqlite')],dict(scored_documents=len(cc)))
        candidates=sorted(noncc+[(r,p,'common_crawl') for r,p in kept],key=lambda d:d[1])
        with pipeline.database(path/'select/candidates.sqlite') as db:
            db.execute('CREATE TABLE candidates(row_index INTEGER PRIMARY KEY,position INTEGER UNIQUE,source TEXT)')
            db.executemany('INSERT INTO candidates VALUES(?,?,?)',candidates)
        sizes=[2,3,4,7,8,9,13,33,257]
        def document(row,pos,src,slot=-1):
            size=sizes[row%len(sizes)];tokens=[1]+[100+row]*(size-2)+[9]
            return dict(row_index=row,shuffle_position=pos,id=original[row]['id'],source=src,
                text_sha256=hashlib.sha256(original[row]['text'].encode()).digest(),tokens=tokens,evaluation_slot=slot)
        train=[document(r,p,s) for r,p,s in candidates]
        eval_docs=[document(int(r),n-reserved+i,source(int(r)),i) for i,r in enumerate(evaluation)]
        files={}
        for phase,docs in (('train',train),('eval',eval_docs)):
            files[phase]=[]
            for i in range(0,len(docs),23):
                files[phase].append(pipeline.write_arrow(path/'select'/f'{phase}-{i}.arrow',docs[i:i+23],pipeline.document_schema()))
        details=dict(train_files=files['train'],eval_files=files['eval'],train_tokens=sum(len(d['tokens']) for d in train),
            train_documents=len(train),evaluation_tokens=sum(len(d['tokens']) for d in eval_docs),bos_token_id=1,eos_token_id=9,pad_token_id=9)
        finish('select',[str(path/'select/candidates.sqlite')]+[i['path'] for i in files['train']+files['eval']],details)
        (path/'packing-access/parallel-v1').mkdir(parents=True)
        return prep
    reference=prepare('reference');candidate=prepare('parallel');resumed=prepare('resumed');legacy=prepare('legacy-resumed')
    pipeline.stage_pack(reference)
    identity=dict(fixture='packing-equivalence')
    def execute(prep,label):
        target=directory/label;target.mkdir();parallel.packing(prep,target,identity)
    execute(candidate,'parallel-output')
    original_atomic=pipeline.atomic_json
    for prep,label,first_parallel in ((resumed,'resumed',True),(legacy,'legacy-resumed',False)):
        checkpoint=prep.path/'pack/state.json'
        def stop(path,value):
            original_atomic(path,value)
            if Path(path)==checkpoint:raise pipeline.StopRequested('fixture faithful checkpoint')
        try:
            with patch.object(pipeline,'atomic_json',stop):
                if first_parallel:execute(prep,label+'-first')
                else:pipeline.stage_pack(prep)
        except pipeline.StopRequested:pass
        else:raise AssertionError('Fixture did not checkpoint')
        execute(prep,label+'-continuation')
    def compare(prep):
        a=reference.manifest('pack')['details'];b=prep.manifest('pack')['details']
        for key in ('training_tokens','evaluation_input_tokens','evaluation_prediction_tokens','subset_prediction_tokens','train_cursor','eval_cursor','carry'):
            if a[key]!=b[key]:raise ValueError('Fixture counters/cursors/carry differ: '+key)
        for key in ('train_files','eval_files'):
            x=[r for r,_ in pipeline.iter_arrow(a[key])];y=[r for r,_ in pipeline.iter_arrow(b[key])]
            if x!=y:raise ValueError('Fixture packed tokens/masks/spans/order differ')
            if [r['sha256'] for r in a[key]]!=[r['sha256'] for r in b[key]]:raise ValueError('Fixture Arrow bytes/shard boundaries differ')
    for prep in (candidate,resumed,legacy):compare(prep)
    with patch.object(pipeline,'load_raw',return_value=raw):pipeline.stage_verify(reference)
    verify_dir=directory/'verification';verify_dir.mkdir()
    report=verification.verify(candidate,raw,index,verify_dir,identity)
    reference_report=pipeline.read_json(reference.path/'verify/report.json')
    for key in ('training','evaluation','evaluation_documents','evaluation_raw_documents','subset_documents'):
        if report[key]!=reference_report[key]:raise ValueError('Full original/parallel verification reports differ')
    # The checksum is updated too: reconstruction must independently catch this.
    selected=candidate.manifest('select')['details'];packed=candidate.manifest('pack')['details'];plans=parallel.training_plans(selected['train_files'],[0,0],{},length,3)
    task=next(parallel.make_tasks(selected['train_files'],plans,'train',cfg,9,actual=packed['train_files']))
    rows=[r for r,_ in pipeline.iter_arrow([task['actual']])];rows[0]['labels'][1]=123456
    broken=pipeline.write_arrow(directory/'corrupted-mask.arrow',rows,pipeline.sequence_schema(length));task['actual']=broken
    try:parallel.sequence_task(task)
    except ValueError:pass
    else:raise AssertionError('Updated-checksum mask corruption escaped reconstruction')
    # Rank membership corruption is checked independently of document selection.
    cc_count=candidate.manifest('clean')['details']['source_counts']['common_crawl'];keep=cc_count//10
    with pipeline.readonly(candidate.path/'score/scores.sqlite') as db:
        boundary=db.execute('SELECT score,position FROM scores ORDER BY score DESC,position ASC LIMIT 1 OFFSET ?',(keep-1,)).fetchone()
    with pipeline.database(candidate.path/'select/candidates.sqlite') as db:
        db.execute("DELETE FROM candidates WHERE source='common_crawl' AND row_index=(SELECT MIN(row_index) FROM candidates WHERE source='common_crawl')")
    with parallel.Workers(cfg,dict(preparation_path=str(candidate.path),score_boundary=boundary)) as workers:
        try:list(workers.run(verification.ranking_task,[(0,n)]))
        except ValueError:pass
        else:raise AssertionError('Missing quality winner escaped verification')
    raw.close();result=dict(passed=True,byte_identical_shards=True,canonical_masks_spans_order=True,
        original_resume=True,parallel_resume=True,full_original_verification_equal=True,
        updated_checksum_mask_corruption_rejected=True,quality_membership_corruption_rejected=True,
        workers=3,job_id=os.environ['SLURM_JOB_ID'])
    pipeline.atomic_json(directory/'result.json',result);return result


def saved_prefix(prep,directory):
    state=pipeline.read_json(prep.path/'pack/state.json') if (prep.path/'pack/state.json').exists() else dict(train_files=[],train_cursor=[0,0],carry={})
    selected=prep.manifest('select')['details'];cfg=prep.config;count=len(state['train_files'])
    if not count:return dict(passed=True,shards=0)
    plans=parallel.training_plans(selected['train_files'],[0,0],{},cfg['sequence_length'],cfg['sequence_shard_rows'])
    recent=[]
    for number,plan in enumerate(itertools.islice(plans,count)):
        recent.append((number,plan));recent=recent[-2:]
    if not recent or recent[-1][0]+1!=count:raise ValueError('Saved shard count differs from canonical plan')
    last=recent[-1][1]
    if last['end']!=state['train_cursor'] or last['carry_after']!=state['carry']:
        raise ValueError('Saved cursor/carry differs from full canonical reconstruction')
    tasks=[]
    for number,plan in recent:
        task=next(parallel.make_tasks(selected['train_files'],[plan],'train',cfg,selected['pad_token_id']))
        task['actual']=state['train_files'][number];tasks.append(task)
    with parallel.Workers(cfg) as workers:list(workers.run(parallel.sequence_task,tasks))
    result=dict(passed=True,shards=count,reconstructed_recent_shards=len(tasks),saved_cursor=state['train_cursor'])
    pipeline.atomic_json(Path(directory)/'saved-prefix.json',result);return result


def benchmark(prep,directory):
    cfg=prep.config;selected=prep.manifest('select')['details'];state=pipeline.read_json(prep.path/'pack/state.json')
    if state['train_complete']:return dict(passed=True,skipped='Training packing already complete')
    plans=list(itertools.islice(parallel.training_plans(selected['train_files'],state['train_cursor'],state['carry'],cfg['sequence_length'],cfg['sequence_shard_rows']),32))
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);measurements={};results={}
    for mode in ('serial','parallel'):
        output=directory/mode;output.mkdir()
        tasks=list(parallel.make_tasks(selected['train_files'],plans,'train',cfg,selected['pad_token_id'],output=output))
        started=time.monotonic()
        if mode=='serial':values=[parallel.sequence_task(t) for t in tasks]
        else:
            with parallel.Workers(cfg) as workers:values=list(workers.run(parallel.sequence_task,tasks));stats=workers.stats
        seconds=time.monotonic()-started;tokens=sum(v['input_tokens'] for v in values)
        measurements[mode]=dict(seconds=seconds,tokens=tokens,tokens_per_second=tokens/seconds);results[mode]=values
    for left,right in zip(results['serial'],results['parallel']):
        for key in ('cursor','carry','input_tokens','prediction_tokens','subset_prediction_tokens'):
            if left[key]!=right[key]:raise ValueError('Matched real-data packing changed '+key)
        if left['file']['sha256']!=right['file']['sha256']:raise ValueError('Matched real-data Arrow bytes changed')
    speedup=measurements['serial']['seconds']/measurements['parallel']['seconds']
    memory=1024*(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss+workers.count*resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    result=dict(passed=speedup>=1.1 and memory<160*1024**3,measurements=measurements,speedup=speedup,
        memory_upper_bytes=memory,stats=stats,byte_identical=True,
        timing_scope='matched preplanned shards; includes read/pack/write/hash and worker startup')
    pipeline.atomic_json(directory/'result.json',result)
    if not result['passed']:raise ValueError('Packing speed/memory gate failed; original packing state unchanged')
    return result


def run(prep,raw,directory,identity,stage):
    directory=Path(directory);receipt=directory/'checks.json'
    if receipt.exists():
        result=pipeline.read_json(receipt)
        if result['identity']!=identity or not result['passed']:raise ValueError('Packing gate identity changed')
        return result
    fixture_result=fixture(directory/('fixture-'+os.environ['SLURM_JOB_ID']))
    prefix=saved_prefix(prep,directory)
    performance=benchmark(prep,directory/('benchmark-'+os.environ['SLURM_JOB_ID'])) if stage=='pack' else dict(passed=True,skipped='Verification only')
    result=dict(passed=True,identity=identity,fixture=fixture_result,saved_prefix=prefix,performance=performance,job_id=os.environ['SLURM_JOB_ID'])
    pipeline.atomic_json(receipt,result);print('parallel packing/verification gates passed:',json.dumps(result),flush=True)
    return result
